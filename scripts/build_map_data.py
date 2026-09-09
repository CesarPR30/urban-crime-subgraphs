"""Genera los artefactos estáticos que consume el mapa de verificación.

Paso 0 del proyecto: comprobar visualmente que el CSV de crímenes se lee bien y
que los puntos caen donde deben. No forma parte todavía del pipeline definitivo
(`crimepipe`); es el smoke-test previo a la Fase A.

Uso:
    python scripts/build_map_data.py [--csv RUTA] [--out RUTA] [opciones]

Salidas en `dashboard/public/data/`:
    crimes.bin              posiciones y atributos empaquetados (8 B/punto)
    crimes_meta.json        diccionarios, bbox, cuantización, matriz mes×tipo
    validation_report.json  reporte de validación (§3.1)

Y en `data/interim/`:
    crimes_canonical.csv    el CSV proyectado al esquema canónico de 7 columnas

## Por qué binario

Medio millón de puntos en JSON son ~12 MB de texto que hay que parsear en el
hilo principal. Empaquetados a enteros son ~3.9 MB que el navegador expone como
`TypedArray` sobre el `ArrayBuffer` recibido, con coste de parseo cero.

Disposición del búfer, con `n` = número de puntos:

    offset      tipo      campo
    0           Uint16    lat    cuantizada sobre el bbox
    2n          Uint16    lon    cuantizada sobre el bbox
    4n          Uint16    place  índice en el diccionario de lugares
    6n          Uint8     cat    índice en el diccionario de tipos
    7n          Uint8     month  índice en el diccionario de meses

Los `Uint16` van primero para que sus offsets queden alineados a 2 bytes, que es
lo que exige el constructor de `Uint16Array` sobre un `ArrayBuffer`.

La cuantización reparte 65 536 niveles sobre la extensión de los datos: para el
bbox de Chicago eso son ~0.6 m, tres órdenes de magnitud por debajo de lo que
distingue el ojo a cualquier zoom. El pipeline real trabaja sobre el CSV crudo
con doble precisión; esta pérdida solo afecta al dibujo.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from array import array
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pipeline.config import available_datasets, load_config  # noqa: E402
from pipeline.ingest.crimes import (  # noqa: E402
    CANONICAL_FIELDS,
    CrimeRecord,
    fingerprint,
    load_crimes,
)

logger = logging.getLogger("build_map_data")

QUANT_MAX = 65535


def _index(values: list[str], seen: dict[str, int], v: str) -> int:
    i = seen.get(v)
    if i is None:
        i = seen[v] = len(values)
        values.append(v)
    return i


def pack(
    records: list[CrimeRecord],
    bbox: tuple[float, float, float, float],
    focus: tuple[str, ...] | None = None,
):
    """Empaqueta los registros al búfer binario descrito en el docstring.

    Cuantiza cada coordenada a `Uint16` sobre la extensión de los datos::

        q = round( (v - v_min) / (v_max - v_min) * 65535 )

    Devuelve `(buffer, meta)`.
    """
    lat0, lon0, lat1, lon1 = bbox
    dlat = (lat1 - lat0) or 1.0
    dlon = (lon1 - lon0) or 1.0

    categories: list[str] = []
    months: list[str] = []
    places: list[str] = []
    ci: dict[str, int] = {}
    mi: dict[str, int] = {}
    pi: dict[str, int] = {}

    n = len(records)
    q_lat = array("H", bytes(2 * n))
    q_lon = array("H", bytes(2 * n))
    q_place = array("H", bytes(2 * n))
    q_cat = array("B", bytes(n))
    q_month = array("B", bytes(n))

    for i, r in enumerate(records):
        q_lat[i] = round((r.lat - lat0) / dlat * QUANT_MAX)
        q_lon[i] = round((r.lon - lon0) / dlon * QUANT_MAX)
        q_place[i] = _index(places, pi, r.lugar or "—")
        q_cat[i] = _index(categories, ci, r.tipo)
        q_month[i] = _index(months, mi, r.mes)

    # Los meses se descubren en el orden del CSV; reindexar a orden cronológico
    # para que el selector temporal tenga sentido como eje.
    order = sorted(range(len(months)), key=lambda i: months[i])
    remap = {old: new for new, old in enumerate(order)}
    months = [months[i] for i in order]
    for i in range(n):
        q_month[i] = remap[q_month[i]]

    if len(categories) > 256:
        raise ValueError(f"{len(categories)} categorías no caben en Uint8")
    if len(places) > QUANT_MAX + 1:
        raise ValueError(f"{len(places)} lugares no caben en Uint16")

    for a in (q_lat, q_lon, q_place):
        if sys.byteorder != "little":
            a.byteswap()

    buf = b"".join(a.tobytes() for a in (q_lat, q_lon, q_place, q_cat, q_month))

    # Matriz de conteos mes × tipo. 24×31 enteros que evitan al navegador
    # recorrer medio millón de puntos cada vez que cambia un filtro: cualquier
    # total visible es una suma sobre la submatriz seleccionada.
    matrix = [[0] * len(categories) for _ in months]
    for i in range(n):
        matrix[q_month[i]][q_cat[i]] += 1

    meta = {
        "n": n,
        # Firma del orden de `records`. `snapping.bin` lleva la misma y el
        # dashboard las compara antes de combinar los dos artefactos: se
        # indexan por posición, así que mezclarlos tras cargas distintas
        # pintaría cada crimen en el nodo de otro.
        "fingerprint": fingerprint(records),
        "bbox": [lat0, lon0, lat1, lon1],
        "quant_max": QUANT_MAX,
        "categories": categories,
        "months": months,
        "places": places,
        # Sin recorte explícito de categorías, el foco son todas las que
        # quedaron: el alcance ya se decidió antes (en Lima, sobre `tipo_hecho`
        # vía `keep_where`), y marcar unas pocas con estrella sugeriría un
        # recorte que no existe.
        "focus_categories": [c for c in (focus or ()) if c in ci] or categories,
        "matrix": matrix,
        "layout": [
            {"field": "lat", "type": "Uint16", "offset": 0},
            {"field": "lon", "type": "Uint16", "offset": 2 * n},
            {"field": "place", "type": "Uint16", "offset": 4 * n},
            {"field": "cat", "type": "Uint8", "offset": 6 * n},
            {"field": "month", "type": "Uint8", "offset": 7 * n},
        ],
    }
    return buf, meta


def write_canonical(records: list[CrimeRecord], path: Path) -> None:
    """Materializa el dataset ya proyectado al esquema canónico.

    Es la evidencia en disco de la reducción: entren las columnas que entren,
    sale siempre este archivo con las mismas siete.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(CANONICAL_FIELDS)
        for r in records:
            w.writerow([r.lat, r.lon, r.fecha.isoformat(sep=" "),
                        r.crimen, r.tipo, r.lugar or "", r.uid or ""])


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--dataset",
        default=None,
        help="ciudad a procesar (bloque de `datasets:` en config.yaml). "
             "Por defecto, `active_dataset`.",
    )
    p.add_argument("--config", type=Path, default=None)
    p.add_argument("--csv", type=Path, default=None, help="anula data.crimes_csv")
    p.add_argument("--out", type=Path, default=None,
                   help="anula dashboard/public/data/<dataset>")
    p.add_argument("--from", dest="month_from", default=None, help="YYYY-MM")
    p.add_argument("--to", dest="month_to", default=None, help="YYYY-MM")
    p.add_argument(
        "--all-categories",
        action="store_true",
        help="retener todos los tipos (por defecto: solo los que analiza la tesis)",
    )
    p.add_argument(
        "--dedupe",
        choices=("auto", "id", "content", "none"),
        default=None,
        help="clave de deduplicación. Por defecto, la del dataset en config.yaml",
    )
    p.add_argument("--no-canonical", action="store_true",
                   help="no escribir <interim>/crimes_canonical.csv")
    args = p.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s  %(levelname)-7s %(name)s  %(message)s"
    )
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass

    # Todo lo que define el recorte —CSV, ventana, categorías, perfil de fuente,
    # limpieza— sale del config del dataset. Antes vivía aquí como constantes,
    # lo que obligaba a editar el script para mirar otra ciudad y, peor, dejaba
    # que este script y `crimepipe` cargasen el mismo CSV con criterios
    # distintos: los dos artefactos se indexan por posición y solo casan si
    # salieron de la misma carga.
    cfg = load_config(args.config, args.dataset)
    csv_path = args.csv or cfg.path(cfg.data.crimes_csv)
    out_dir = args.out or cfg.dashboard_data
    interim = cfg.path(cfg.data.interim_dir)
    focus = tuple(cfg.data.categories or ())

    if not csv_path.exists():
        logger.error("No existe el CSV: %s", csv_path)
        return 1
    logger.info("Dataset %s (%s) -> %s", cfg.dataset, cfg.dataset_label, out_dir)

    records, report = load_crimes(
        csv_path,
        categories=None if args.all_categories else (focus or None),
        month_from=args.month_from or cfg.data.month_from,
        month_to=args.month_to or cfg.data.month_to,
        dedupe=args.dedupe or cfg.data.dedupe,
        profile=cfg.data.source_profile,
        cleaning=cfg.data.cleaning,
    )
    print()
    print(report.summary())
    print()

    if not records:
        logger.error("Ningún registro sobrevivió. Revisa los filtros.")
        return 1

    out_dir.mkdir(parents=True, exist_ok=True)
    buf, meta = pack(records, report.bbox, focus)
    meta["dataset"] = cfg.dataset
    # La tabla de §7.3, si esta ciudad tiene una. El dashboard omite las
    # columnas de contraste cuando es `null`, en vez de restar contra los
    # numeros de otra ciudad.
    meta["reference"] = cfg.reference.model_dump() if cfg.reference else None
    meta["label"] = cfg.dataset_label
    meta["sublabel"] = cfg.dataset_sublabel
    meta["source_csv"] = csv_path.name
    meta["window"] = f"{meta['months'][0]} .. {meta['months'][-1]}"

    bin_path = out_dir / "crimes.bin"
    bin_path.write_bytes(buf)
    (out_dir / "crimes_meta.json").write_text(
        json.dumps(meta, separators=(",", ":")), encoding="utf-8"
    )
    (out_dir / "validation_report.json").write_text(
        json.dumps(report.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8"
    )

    # Artefacto JSON antiguo: ya no se usa y confundiría al servirlo.
    legacy = out_dir / "crimes_points.json"
    if legacy.exists():
        legacy.unlink()

    if not args.no_canonical:
        canon = interim / "crimes_canonical.csv"
        write_canonical(records, canon)
        logger.info("Escrito %s (%.1f MB, %d columnas)",
                    canon, canon.stat().st_size / 1e6, len(CANONICAL_FIELDS))

    logger.info(
        "Escrito %s (%.2f MB · %d puntos · %.1f B/punto)",
        bin_path, len(buf) / 1e6, meta["n"], len(buf) / meta["n"],
    )
    logger.info(
        "Diccionarios: %d tipos, %d meses, %d lugares · matriz %d×%d",
        len(meta["categories"]), len(meta["months"]), len(meta["places"]),
        len(meta["months"]), len(meta["categories"]),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
