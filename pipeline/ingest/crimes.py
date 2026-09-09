"""Carga y validación del CSV de crímenes (PROJECT_SPEC §3.1).

Acepta *cualquier* CSV que tenga columnas equivalentes a las cinco canónicas
(`lat`, `lon`, `fecha`, `crimen`, `tipo`), resolviendo alias sin distinguir
mayúsculas, espacios ni guiones bajos.

**Proyección al esquema canónico.** El cargador no arrastra el CSV de entrada:
lo *proyecta* a un esquema fijo y reducido. Da igual que la fuente traiga 6
columnas o 22 — salen siempre las mismas seis (`lat`, `lon`, `fecha`, `crimen`,
`tipo`, más `lugar` e `id` si existen) y todo lo demás se descarta de forma
explícita y auditable. Es lo que hace que el resto del pipeline sea portable a
otra ciudad: nada aguas abajo conoce el esquema del proveedor.

Implementación en stdlib puro a propósito: este módulo es la puerta de entrada
del pipeline y debe poder ejecutarse sin instalar nada. Cuando en la Fase B se
fije el entorno con `uv`, la salida se materializa a Parquet vía pandas/pyarrow
sin tocar la lógica de resolución de alias ni de validación que vive aquí.
"""

from __future__ import annotations

import csv
import hashlib
import logging
import re
import time
import unicodedata
from collections import Counter
from dataclasses import dataclass, field, replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable, Iterator, Sequence

from .profiles import SourceProfile, find_fallback_points, get_profile

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Vocabulario de columnas
# --------------------------------------------------------------------------- #

#: Alias aceptados por columna canónica, ya normalizados (ver `_normalize_key`).
#:
#: `log` es alias real de longitud: aparece así en algunos CSV de origen.
#: NO quitar.
COLUMN_ALIASES: dict[str, tuple[str, ...]] = {
    "lat": ("lat", "latitude", "latitud", "y", "lat hecho"),
    "lon": ("lon", "log", "lng", "long", "longitude", "longitud", "x", "long hecho"),
    "fecha": (
        "fecha",
        "date",
        "datetime",
        "fecha hora",
        "occurred at",
        # Portal de datos abiertos de Chicago
        "date of occurrence",
        "occurrence date",
        # PNP / DGIS (Lima). La ISO ya trae fecha y hora del *hecho*; el epoch
        # en milisegundos de `fecha_hora_hecho` diría lo mismo con más trabajo.
        "fecha hora hecho iso utc",
        "fecha hora hecho",
    ),
    # `crimen` es la descripción específica del incidente; `tipo` es su categoría.
    # En el export completo de Chicago eso es `Description` / `Primary Type`. En
    # exports reducidos solo existe `PRIMARY DESCRIPTION`, y ambas caen ahí.
    "crimen": (
        "crimen",
        "crime",
        "delito",
        "incidente",
        "incident",
        # Portal de datos abiertos de Chicago
        "description",
        "secondary description",
        "primary description",
        "primary type",
        # PNP / DGIS (Lima)
        "subtipo hecho",
    ),
    "tipo": (
        "tipo",
        "type",
        "crime type",
        "tipo delito",
        "categoria",
        # Portal de datos abiertos de Chicago
        "primary type",
        "primary description",
        # PNP / DGIS (Lima)
        "tipo hecho",
    ),
}

#: Columnas opcionales: si están, se conservan; si no, no pasa nada.
OPTIONAL_ALIASES: dict[str, tuple[str, ...]] = {
    "lugar": ("lugar", "location description", "place", "descripcion lugar",
              "distrito hecho"),
    # Identificador estable del incidente. Es la clave correcta para deduplicar:
    # dos incidentes reales pueden coincidir en fecha, tipo y coordenada (Chicago
    # geocodifica al centroide de la manzana) pero nunca en `id`.
    "id": ("id", "incident id", "row id", "case number", "numero de caso",
           "id dgc"),
}

#: El esquema al que se proyecta *cualquier* CSV de entrada. Todo lo que no
#: esté aquí se descarta. Ver el docstring del módulo.
CANONICAL_FIELDS: tuple[str, ...] = (
    "lat", "lon", "fecha", "crimen", "tipo", "lugar", "id",
)

#: Formatos de fecha probados en orden.
DATE_FORMATS: tuple[str, ...] = (
    "%m/%d/%Y %I:%M:%S %p",
    "%m/%d/%Y %H:%M:%S",
    "%m/%d/%Y %H:%M",
    "%m/%d/%Y",
    "%Y-%m-%d %H:%M:%S",
    "%Y-%m-%dT%H:%M:%S",
    "%Y-%m-%d",
    "%d/%m/%Y %H:%M:%S",
    "%d/%m/%Y",
    # ISO 8601 con offset explícito (`2019-01-12T05:00:00+00:00`), que es como
    # la PNP publica la hora del hecho.
    "%Y-%m-%dT%H:%M:%S%z",
)

#: Razones de descarte reportadas en el ValidationReport (§3.1).
DISCARD_REASONS: tuple[str, ...] = (
    "lat_invalida",
    "lon_invalida",
    "fecha_invalida",
    "crimen_vacio",
    "tipo_vacio",
)

#: Razones de filtrado: la fila es válida pero queda fuera del alcance.
FILTER_REASONS: tuple[str, ...] = (
    "fuera_de_categoria",
    "fuera_de_ventana",
)

#: Razones de exclusión por decisión sobre la *fuente*: la fila está bien
#: formada y dentro del alcance, pero el proveedor la marca como no fiable o el
#: detector de centroides la sitúa en un punto de relleno. Se contabiliza aparte
#: para que ni la calidad del CSV ni el recorte del estudio queden contaminados
#: por decisiones que no son ni una cosa ni la otra (§3.1).
SOURCE_REASONS: tuple[str, ...] = (
    "marcada_por_la_fuente",
    "centroide_de_relleno",
)


# --------------------------------------------------------------------------- #
# Objetos de dominio
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class CrimeRecord:
    """Un crimen ya validado y normalizado."""

    lat: float
    lon: float
    fecha: datetime
    crimen: str
    tipo: str
    lugar: str | None = None
    uid: str | None = None

    @property
    def mes(self) -> str:
        """Mes calendario en formato `YYYY-MM` (unidad de agregación temporal)."""
        return f"{self.fecha.year:04d}-{self.fecha.month:02d}"


@dataclass
class ValidationReport:
    """Resumen auditable de una carga de CSV (§3.1).

    Distingue dos cosas que no hay que mezclar:

    * **descartes** — filas que el CSV trae mal (coordenada ilegible, fecha
      imposible, campo vacío). Miden la calidad de la fuente.
    * **filtros** — filas correctas que quedan fuera del alcance del estudio
      (categoría no analizada, mes fuera de la ventana). Miden el recorte.

    Un descarte es un problema; un filtro es una decisión. `valid_rows` cuenta
    las filas bien formadas *antes* de filtrar, para que la calidad del CSV se
    pueda auditar con independencia de la ventana de análisis.
    """

    source: str
    total_rows: int = 0
    valid_rows: int = 0
    discarded_rows: int = 0
    duplicate_rows: int = 0
    kept_rows: int = 0
    filtered_rows: int = 0
    source_excluded_rows: int = 0
    column_mapping: dict[str, str] = field(default_factory=dict)
    columns_in: list[str] = field(default_factory=list)
    columns_dropped: list[str] = field(default_factory=list)
    dedupe_key: str = "none"
    discards_by_reason: dict[str, int] = field(
        default_factory=lambda: {r: 0 for r in DISCARD_REASONS}
    )
    filters_applied: dict[str, object] = field(default_factory=dict)
    filters_by_reason: dict[str, int] = field(
        default_factory=lambda: {r: 0 for r in FILTER_REASONS}
    )
    source_profile: str = "generic"
    source_rules: dict[str, object] = field(default_factory=dict)
    source_by_reason: dict[str, int] = field(
        default_factory=lambda: {r: 0 for r in SOURCE_REASONS}
    )
    fallback_points: list[dict[str, object]] = field(default_factory=list)
    """Los centroides de relleno detectados, con su peso. Van al reporte para
    que la decisión sea revisable: si el detector se comiera una esquina real,
    se vería aquí antes que en el mapa."""
    months: dict[str, int] = field(default_factory=dict)
    categories: dict[str, int] = field(default_factory=dict)
    bbox: tuple[float, float, float, float] | None = None
    """(lat_min, lon_min, lat_max, lon_max) sobre las filas válidas."""
    elapsed_s: float = 0.0

    def to_dict(self) -> dict[str, object]:
        return {
            "source": self.source,
            "total_rows": self.total_rows,
            "valid_rows": self.valid_rows,
            "discarded_rows": self.discarded_rows,
            "duplicate_rows": self.duplicate_rows,
            "filtered_rows": self.filtered_rows,
            "source_excluded_rows": self.source_excluded_rows,
            "kept_rows": self.kept_rows,
            "valid_fraction": (
                round(self.valid_rows / self.total_rows, 6) if self.total_rows else 0.0
            ),
            "dedupe_key": self.dedupe_key,
            "source_profile": self.source_profile,
            "source_rules": self.source_rules,
            "source_by_reason": self.source_by_reason,
            "fallback_points": self.fallback_points,
            "columns_in": self.columns_in,
            "columns_kept": list(self.column_mapping),
            "columns_dropped": self.columns_dropped,
            "column_mapping": self.column_mapping,
            "discards_by_reason": self.discards_by_reason,
            "filters_applied": self.filters_applied,
            "filters_by_reason": self.filters_by_reason,
            "months": dict(sorted(self.months.items())),
            "categories": dict(
                sorted(self.categories.items(), key=lambda kv: -kv[1])
            ),
            "bbox": list(self.bbox) if self.bbox else None,
            "elapsed_s": round(self.elapsed_s, 3),
        }

    def summary(self) -> str:
        lines = [
            f"Fuente           : {self.source}",
            f"Filas totales    : {self.total_rows:,}",
            f"Filas válidas    : {self.valid_rows:,}",
            f"Filas descartadas: {self.discarded_rows:,}",
            f"Filas duplicadas : {self.duplicate_rows:,}  (clave: {self.dedupe_key})",
            f"Filas filtradas  : {self.filtered_rows:,}",
            f"Filas excluidas  : {self.source_excluded_rows:,}  "
            f"(perfil: {self.source_profile})",
            f"Filas retenidas  : {self.kept_rows:,}",
            f"Columnas         : {len(self.columns_in)} de entrada "
            f"-> {len(self.column_mapping)} canónicas "
            f"({len(self.columns_dropped)} descartadas)",
            "Mapeo de columnas:",
        ]
        lines += [f"    {k:<7} <- {v!r}" for k, v in self.column_mapping.items()]
        if self.columns_dropped:
            lines.append("Columnas descartadas:")
            lines.append("    " + ", ".join(self.columns_dropped))
        if self.source_rules:
            lines.append("Exclusiones de la fuente (perfil %s):" % self.source_profile)
            lines += [f"    {k:<22} {v}" for k, v in self.source_rules.items()]
            lines += [f"    {k:<22} {v:,}" for k, v in self.source_by_reason.items()]
            for fp in self.fallback_points[:10]:
                lines.append(
                    f"      centroide {fp['point']}  {fp['rows']:,} hechos, "
                    f"{fp['min_addresses']}+ direcciones"
                )
            if len(self.fallback_points) > 10:
                lines.append(f"      … y {len(self.fallback_points) - 10} más")
        lines.append("Descartes por razón (calidad del CSV):")
        lines += [f"    {k:<18} {v:,}" for k, v in self.discards_by_reason.items()]
        if self.filters_applied:
            lines.append("Filtros aplicados (alcance del estudio):")
            lines += [f"    {k:<18} {v}" for k, v in self.filters_applied.items()]
            lines += [f"    {k:<18} {v:,}" for k, v in self.filters_by_reason.items()]
        lines.append(f"Meses retenidos  : {len(self.months)}")
        lines += [f"    {m} {c:>7,}" for m, c in sorted(self.months.items())]
        if self.bbox:
            la0, lo0, la1, lo1 = self.bbox
            lines.append(f"BBox            : lat [{la0:.5f}, {la1:.5f}]  lon [{lo0:.5f}, {lo1:.5f}]")
        lines.append(f"Tiempo           : {self.elapsed_s:.2f} s")
        return "\n".join(lines)


class MissingColumnError(ValueError):
    """Falta una columna obligatoria en el CSV de entrada."""


# --------------------------------------------------------------------------- #
# Resolución de columnas
# --------------------------------------------------------------------------- #


def _normalize_key(name: str) -> str:
    """Normaliza un encabezado para comparar contra los alias.

    Minúsculas, sin acentos, sin puntuación; guiones bajos y espacios múltiples
    colapsados a un solo espacio. Así `"DATE  OF OCCURRENCE"`, `"date_of_occurrence"`
    y `"Date Of Occurrence"` resuelven todos a `"date of occurrence"`.
    """
    s = unicodedata.normalize("NFKD", name)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = s.lower().replace("_", " ").replace("-", " ")
    s = re.sub(r"[^a-z0-9 ]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def resolve_columns(
    header: Sequence[str],
    profile: SourceProfile | None = None,
) -> tuple[dict[str, str], dict[str, str]]:
    """Mapea columnas canónicas -> nombre real en el CSV.

    El anclaje explícito del perfil gana sobre los alias. Los alias son una
    heurística cómoda para un CSV que se ve por primera vez; en cuanto la
    fuente es conocida y tiene varias columnas que compiten por la misma ranura
    canónica, dejar que gane la heurística es dejar que el orden de un
    diccionario decida el análisis.

    Devuelve `(obligatorias, opcionales)`. Lanza `MissingColumnError` listando los
    alias aceptados si falta alguna obligatoria.
    """
    normalized = {_normalize_key(h): h for h in header}
    pinned = dict(profile.columns) if profile else {}
    unknown = [c for c in pinned.values() if c not in header]
    if unknown:
        raise MissingColumnError(
            f"El perfil {profile.name!r} ancla columnas que el CSV no trae: "
            f"{', '.join(unknown)}.\n"
            f"Encabezados encontrados: {list(header)}"
        )

    required: dict[str, str] = {}
    missing: list[str] = []
    for canonical, aliases in COLUMN_ALIASES.items():
        if canonical in pinned:
            required[canonical] = pinned[canonical]
            continue
        for alias in aliases:
            if alias in normalized:
                required[canonical] = normalized[alias]
                break
        else:
            missing.append(canonical)

    if missing:
        detail = "\n".join(
            f"  {c}: {', '.join(COLUMN_ALIASES[c])}" for c in missing
        )
        raise MissingColumnError(
            f"Faltan columnas obligatorias: {', '.join(missing)}.\n"
            f"Encabezados encontrados: {list(header)}\n"
            f"Alias aceptados:\n{detail}"
        )

    optional: dict[str, str] = {}
    for canonical, aliases in OPTIONAL_ALIASES.items():
        if canonical in pinned:
            optional[canonical] = pinned[canonical]
            continue
        for alias in aliases:
            if alias in normalized:
                optional[canonical] = normalized[alias]
                break

    return required, optional


def parse_date(raw: str, utc_offset_h: float | None = None) -> datetime | None:
    """Parsea una fecha probando `DATE_FORMATS` en orden; `None` si ninguno aplica.

    Devuelve siempre un datetime **naíf en hora local**. Si el texto trae
    offset, se traslada a `utc_offset_h` (o se toma tal cual cuando no se
    indica) y se le quita el `tzinfo`: aguas abajo todo compara y ordena
    fechas, y mezclar naífs con conscientes revienta esas comparaciones.
    """
    raw = raw.strip()
    if not raw:
        return None
    dt: datetime | None = None
    for fmt in DATE_FORMATS:
        try:
            dt = datetime.strptime(raw, fmt)
            break
        except ValueError:
            continue
    if dt is None:
        try:
            dt = datetime.fromisoformat(raw)
        except ValueError:
            return None
    if dt.tzinfo is not None:
        tz = timezone(timedelta(hours=utc_offset_h)) if utc_offset_h is not None             else dt.tzinfo
        dt = dt.astimezone(tz).replace(tzinfo=None)
    return dt


# --------------------------------------------------------------------------- #
# Carga
# --------------------------------------------------------------------------- #


def load_crimes(
    path: str | Path,
    *,
    categories: Iterable[str] | None = None,
    month_from: str | None = None,
    month_to: str | None = None,
    dedupe: str = "auto",
    encoding: str = "utf-8-sig",
    profile: SourceProfile | str | None = None,
    cleaning: dict | None = None,
) -> tuple[list[CrimeRecord], ValidationReport]:
    """Carga, valida, deduplica, limpia y recorta un CSV de crímenes.

    Proyecta cualquier CSV de entrada al esquema de `CANONICAL_FIELDS`; las
    columnas sobrantes se registran en `report.columns_dropped` y se descartan.

    Args:
        categories: si se da, retiene solo las filas cuyo `tipo` esté en el
            conjunto (comparación sin distinguir mayúsculas ni espacios de
            sobra). `None` = todas.
        month_from: primer mes inclusive, `YYYY-MM`. `None` = sin límite inferior.
        month_to: último mes inclusive, `YYYY-MM`. `None` = sin límite superior.
        dedupe: `"auto"` usa `id` si el CSV lo trae y si no cae a `"content"`;
            `"id"` fuerza la clave de identificador; `"content"` usa la tupla
            (fecha, tipo, crimen, lat, lon); `"none"` no deduplica.

            Ojo con `"content"`: dos incidentes distintos pueden compartir esa
            tupla cuando la fuente geocodifica al centroide de la manzana, así
            que colapsa crímenes reales. Prefiere `id` siempre que exista.
        profile: perfil de fuente (ver `profiles.py`) o su nombre. Ancla las
            columnas, define el identificador compuesto y las exclusiones que
            la fuente exige. `None` = perfil genérico, todo por alias.
        cleaning: reglas de limpieza del `config.yaml` del dataset. Hoy solo
            `fallback_points` (`min_rows`, `min_addresses`), que es la única
            regla que necesita ver el archivo completo antes de decidir.

    Ordena las etapas validar -> excluir por fuente -> deduplicar -> filtrar ->
    limpiar, de modo que cada contador mide una cosa sola. Devuelve
    `(registros, reporte)`.
    """
    path = Path(path)
    started = time.perf_counter()
    prof = profile if isinstance(profile, SourceProfile) else get_profile(profile)
    # Las exclusiones declaradas en el config ganan sobre las del perfil. El
    # perfil trae el valor por defecto porque es donde se documenta *por qué*
    # esa marca invalida la fila; el config es donde se cambia de opinión sin
    # tocar código, que es lo que exige el analisis de sensibilidad (§10).
    if cleaning and cleaning.get("drop_where"):
        prof = replace(prof, drop_where={
            col: frozenset(str(v).strip().upper() for v in vals)
            for col, vals in cleaning["drop_where"].items()
        })
    if cleaning and cleaning.get("keep_where"):
        prof = replace(prof, keep_where={
            col: frozenset(str(v).strip().upper() for v in vals)
            for col, vals in cleaning["keep_where"].items()
        })
    report = ValidationReport(source=str(path), source_profile=prof.name)
    records: list[CrimeRecord] = []

    cat_set = {c.strip().upper() for c in categories} if categories is not None else None
    if cat_set is not None:
        report.filters_applied["categorias"] = sorted(cat_set)
    if month_from or month_to:
        report.filters_applied["ventana"] = f"{month_from or '*'} .. {month_to or '*'}"
    if prof.drop_where:
        report.source_rules["drop_where"] = {
            k: sorted(v) for k, v in prof.drop_where.items()
        }
    if prof.keep_where:
        report.source_rules["keep_where"] = {
            k: sorted(v) for k, v in prof.keep_where.items()
        }

    fb_cfg = (cleaning or {}).get("fallback_points") or None
    # El detector necesita la dirección escrita: sin ella no hay forma de
    # distinguir un centroide de relleno de una esquina caliente, y aplicar
    # solo el umbral de volumen se llevaría por delante justo lo que se busca.
    if fb_cfg and not prof.address_column:
        raise ValueError(
            f"cleaning.fallback_points exige que el perfil {prof.name!r} declare "
            "address_column; sin dirección el detector no puede decidir."
        )
    if fb_cfg:
        report.source_rules["fallback_points"] = dict(fb_cfg)

    # Coordenada cruda y dirección de cada registro retenido, alineadas por
    # posición con `records`. Solo se materializan si hay detector que las use.
    raw_pts: list[tuple[str, str]] = []
    raw_addrs: list[str] = []

    with path.open(newline="", encoding=encoding) as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames is None:
            raise MissingColumnError(f"{path} no tiene encabezado.")

        required, optional = resolve_columns(reader.fieldnames, prof)
        report.column_mapping = {**required, **optional}
        report.columns_in = list(reader.fieldnames)
        used = set(report.column_mapping.values())
        report.columns_dropped = [c for c in reader.fieldnames if c not in used]
        logger.info(
            "Proyección: %d columnas de entrada -> %d canónicas (%d descartadas)",
            len(report.columns_in),
            len(report.column_mapping),
            len(report.columns_dropped),
        )

        c_lat = required["lat"]
        c_lon = required["lon"]
        c_fecha = required["fecha"]
        c_crimen = required["crimen"]
        c_tipo = required["tipo"]
        c_lugar = optional.get("lugar")
        c_id = optional.get("id")
        c_addr = prof.address_column
        tz = prof.local_utc_offset_h

        if dedupe == "auto":
            dedupe = "id" if (c_id or prof.uid_columns) else "content"
        if dedupe == "id" and not (c_id or prof.uid_columns):
            raise MissingColumnError(
                "dedupe='id' pero el CSV no trae columna de identificador. "
                f"Alias aceptados: {', '.join(OPTIONAL_ALIASES['id'])}"
            )
        # El perfil manda: si define un uid compuesto, la clave que se reporta
        # son sus columnas y no la etiqueta genérica, porque es lo que hay que
        # poder auditar cuando el conteo de duplicados sorprenda.
        report.dedupe_key = "|".join(prof.uid_columns) if prof.uid_columns else dedupe
        seen: set[object] = set()

        for row in reader:
            report.total_rows += 1
            reason = _validate_row(row, c_lat, c_lon, c_fecha, c_crimen, c_tipo, tz)
            if reason is not None:
                report.discarded_rows += 1
                report.discards_by_reason[reason] += 1
                continue

            report.valid_rows += 1

            # Exclusión por decisión de la fuente. Va antes de deduplicar: una
            # fila que la fuente desautoriza no debe ocupar la ranura del uid y
            # hacer pasar por duplicada a la buena que venga detrás.
            if prof.rejects(row) is not None:
                report.source_excluded_rows += 1
                report.source_by_reason["marcada_por_la_fuente"] += 1
                continue

            lat = float(row[c_lat])
            lon = float(row[c_lon])
            fecha = parse_date(row[c_fecha], tz)
            assert fecha is not None  # garantizado por _validate_row
            crimen = row[c_crimen].strip()
            tipo = row[c_tipo].strip()
            lugar = row[c_lugar].strip() if c_lugar and row.get(c_lugar) else None
            base_uid = row[c_id].strip() if c_id and row.get(c_id) else None
            uid = prof.uid(row, base_uid)
            mes = f"{fecha.year:04d}-{fecha.month:02d}"

            if dedupe != "none":
                key = uid if dedupe == "id" else (
                    row[c_fecha], tipo, crimen, raw_lat_lon(row, c_lat, c_lon)
                )
                if key in seen:
                    report.duplicate_rows += 1
                    continue
                seen.add(key)

            if cat_set is not None and tipo.upper() not in cat_set:
                report.filtered_rows += 1
                report.filters_by_reason["fuera_de_categoria"] += 1
                continue
            if (month_from and mes < month_from) or (month_to and mes > month_to):
                report.filtered_rows += 1
                report.filters_by_reason["fuera_de_ventana"] += 1
                continue

            records.append(
                CrimeRecord(
                    lat=lat, lon=lon, fecha=fecha, crimen=crimen,
                    tipo=tipo, lugar=lugar, uid=uid,
                )
            )
            if fb_cfg:
                raw_pts.append(raw_lat_lon(row, c_lat, c_lon))
                raw_addrs.append((row.get(c_addr) or "") if c_addr else "")

    # Segunda pasada: centroides de relleno.
    #
    # Se cuenta sobre lo ya retenido y no sobre el CSV entero, porque lo que
    # importa es la concentración *del material de análisis*. Contar sobre el
    # archivo completo mezclaría categorías que no se van a usar y movería el
    # umbral por razones ajenas al estudio.
    if fb_cfg and records:
        bad = find_fallback_points(
            raw_pts, raw_addrs,
            min_rows=int(fb_cfg["min_rows"]),
            min_addresses=int(fb_cfg["min_addresses"]),
        )
        if bad:
            weight = Counter(p for p in raw_pts if p in bad)
            report.fallback_points = [
                {"point": f"{la},{lo}", "rows": n,
                 "min_addresses": int(fb_cfg["min_addresses"])}
                for (la, lo), n in weight.most_common()
            ]
            dropped = sum(weight.values())
            keep = [i for i, pt in enumerate(raw_pts) if pt not in bad]
            report.source_excluded_rows += dropped
            report.source_by_reason["centroide_de_relleno"] += dropped
            records = [records[i] for i in keep]
            logger.info(
                "Centroides de relleno: %d puntos, %s hechos descartados",
                len(bad), f"{dropped:,}",
            )

    # Las estadísticas se calculan sobre lo que realmente se devuelve, después
    # de todas las etapas. Acumularlas dentro del bucle las dejaría contando
    # filas que la limpieza posterior se lleva.
    report.kept_rows = len(records)
    lat_min = lon_min = float("inf")
    lat_max = lon_max = float("-inf")
    for r in records:
        report.months[r.mes] = report.months.get(r.mes, 0) + 1
        report.categories[r.tipo] = report.categories.get(r.tipo, 0) + 1
        lat_min, lat_max = min(lat_min, r.lat), max(lat_max, r.lat)
        lon_min, lon_max = min(lon_min, r.lon), max(lon_max, r.lon)
    if records:
        report.bbox = (lat_min, lon_min, lat_max, lon_max)
    report.elapsed_s = time.perf_counter() - started

    logger.info(
        "Retenidos %s de %s crímenes de %s en %.2f s "
        "(%s descartados por calidad, %s excluidos por la fuente, "
        "%s duplicados, %s filtrados por alcance)",
        f"{report.kept_rows:,}",
        f"{report.total_rows:,}",
        path.name,
        report.elapsed_s,
        f"{report.discarded_rows:,}",
        f"{report.source_excluded_rows:,}",
        f"{report.duplicate_rows:,}",
        f"{report.filtered_rows:,}",
    )
    return records, report


def raw_lat_lon(row: dict[str, str | None], c_lat: str, c_lon: str) -> tuple[str, str]:
    """Coordenada tal como viene en el CSV, para la clave de deduplicación."""
    return ((row.get(c_lat) or "").strip(), (row.get(c_lon) or "").strip())


def _validate_row(
    row: dict[str, str | None],
    c_lat: str,
    c_lon: str,
    c_fecha: str,
    c_crimen: str,
    c_tipo: str,
    utc_offset_h: float | None = None,
) -> str | None:
    """Devuelve la razón de descarte, o `None` si la fila es válida."""
    raw_lat = (row.get(c_lat) or "").strip()
    try:
        lat = float(raw_lat)
    except ValueError:
        return "lat_invalida"
    if not (-90.0 <= lat <= 90.0):
        return "lat_invalida"

    raw_lon = (row.get(c_lon) or "").strip()
    try:
        lon = float(raw_lon)
    except ValueError:
        return "lon_invalida"
    if not (-180.0 <= lon <= 180.0):
        return "lon_invalida"

    if parse_date(row.get(c_fecha) or "", utc_offset_h) is None:
        return "fecha_invalida"
    if not (row.get(c_crimen) or "").strip():
        return "crimen_vacio"
    if not (row.get(c_tipo) or "").strip():
        return "tipo_vacio"
    return None


def iter_months(records: Sequence[CrimeRecord]) -> Iterator[str]:
    """Meses distintos presentes, en orden cronológico."""
    yield from sorted({r.mes for r in records})


def fingerprint(records: Sequence[CrimeRecord]) -> str:
    """Huella de *esta* secuencia de registros, en *este* orden.

    Dos artefactos que se indexan por posición (el `crimes.bin` del navegador y
    el vector de nodos del snapping) solo son combinables si salieron de la
    misma carga. `load_crimes` es determinista, así que basta con firmar el
    número de filas y unas cuantas coordenadas repartidas: si alguien cambia el
    CSV, la ventana o las categorías de uno de los dos, la huella deja de
    coincidir y el consumidor puede negarse a mezclarlos en vez de dibujar un
    mapa silenciosamente equivocado.
    """
    n = len(records)
    probes = sorted({0, n // 4, n // 2, 3 * n // 4, n - 1} & set(range(n)))
    payload = f"{n}|" + "|".join(
        f"{records[i].lat:.6f},{records[i].lon:.6f},{records[i].fecha.isoformat()}"
        for i in probes
    )
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16]
