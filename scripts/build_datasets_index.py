"""Índice de datasets que consume el selector del dashboard.

El dashboard es estático: no puede leer `config.yaml` ni listar directorios. Sin
un índice tendría que llevar los nombres de las ciudades escritos en el HTML, y
añadir una ciudad significaría tocar el JavaScript. Este script traduce el
`datasets:` del config a un JSON que el navegador sí puede leer.

Solo se listan los datasets **con artefactos en disco**: un dataset declarado en
el config pero todavía sin procesar aparecería en el selector y llevaría a un
mapa vacío, que es peor que no ofrecerlo.

Uso:
    python scripts/build_datasets_index.py

Salida:
    dashboard/public/data/datasets.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pipeline.config import available_datasets, load_config  # noqa: E402

logger = logging.getLogger("build_datasets_index")

DEFAULT_OUT = ROOT / "dashboard" / "public" / "data" / "datasets.json"

#: Lo mínimo para que el mapa arranque. El resto de artefactos (hotspots,
#: similitud, POIs) son opcionales y el dashboard ya sabe funcionar sin ellos.
REQUIRED = ("crimes.bin", "crimes_meta.json", "validation_report.json")

#: Artefactos opcionales cuya presencia se anuncia para que el selector pueda
#: avisar de que una ciudad todavía no tiene el pipeline completo.
OPTIONAL = (
    "hotspots.geojson", "similarity.json", "evaluation.json",
    "param_sweep.json", "snapping.bin", "pois.bin",
    "casestudies.json", "benchmark.json",
)


def describe(name: str, config: Path | None) -> dict | None:
    """Entrada del índice para un dataset, o `None` si no tiene artefactos."""
    cfg = load_config(config, name)
    d = cfg.dashboard_data
    missing = [f for f in REQUIRED if not (d / f).exists()]
    if missing:
        logger.warning(
            "%s: sin artefactos (falta %s). Ejecuta "
            "`python scripts/build_map_data.py --dataset %s`",
            name, ", ".join(missing), name,
        )
        return None

    meta = json.loads((d / "crimes_meta.json").read_text(encoding="utf-8"))
    return {
        "id": name,
        "label": cfg.dataset_label or name,
        "sublabel": cfg.dataset_sublabel,
        "dir": name,
        "place": cfg.network.place,
        "n": meta.get("n"),
        "window": meta.get("window"),
        "months": len(meta.get("months") or ()),
        "categories": len(meta.get("categories") or ()),
        "has": sorted(f for f in OPTIONAL if (d / f).exists()),
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, default=None)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = p.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)-7s %(message)s")
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass

    names = list(available_datasets(args.config))
    if not names:
        logger.error("config.yaml no declara `datasets:`.")
        return 1

    entries = [e for e in (describe(n, args.config) for n in names) if e]
    if not entries:
        logger.error("Ningún dataset tiene artefactos. Nada que indexar.")
        return 1

    # El primero es el que abre el dashboard cuando la URL no pide otro.
    default = load_config(args.config).dataset
    if not any(e["id"] == default for e in entries):
        default = entries[0]["id"]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps({"default": default, "datasets": entries},
                   indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    logger.info("Escrito %s con %d datasets (por defecto: %s)",
                args.out, len(entries), default)
    for e in entries:
        logger.info("  %-8s %-22s %9s hechos · %s · %d artefactos opcionales",
                    e["id"], e["label"], f'{e["n"]:,}', e["window"], len(e["has"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
