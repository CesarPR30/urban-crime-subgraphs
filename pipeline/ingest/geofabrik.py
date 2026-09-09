"""POIs desde instantáneas anuales de Geofabrik, en vez de Overpass.

## Por qué no Overpass

Overpass es un servicio interactivo con racionamiento agresivo, y el pipeline lo
usaba como fuente única: durante tres días la descarga de Lima fue imposible
porque la IP quedó limitada a nivel de red, y el dataset se quedó sin perfil
funcional. Peor que la caída fue el modo de fallo silencioso: un espejo regional
respondía HTTP 200 con **cero elementos** para consultas fuera de su región.

Geofabrik publica extractos por país como fichero estático. La descarga no tiene
cuota, la URL es fija y citable, y el resultado es idéntico en cada ejecución —
que es lo que una tesis necesita y lo que Overpass no puede dar.

## El eje temporal

Geofabrik guarda además una instantánea **por cada 1 de enero desde 2014**:

    https://download.geofabrik.de/south-america/peru-180101.osm.pbf

Eso permite caracterizar cada hotspot con los POIs de **su propio año** en vez
de describir uno de 2018 con el mapa de 2026. Se usa la instantánea del *inicio*
del año, no la del final, para no meter información del futuro en la
caracterización de un mes de ese año.

## La trampa que hay que declarar

El histórico de OSM registra **cuándo alguien mapeó algo**, no cuándo abrió. En
Lima esto no es teórico: `education` se multiplica por 3.4 entre 2018 y 2025, y
al repartir por año de edición el 75 % de la categoría cae en 2018-2019, con
`amenity=school` (6 015) y `amenity=kindergarten` (5 873) casi 1:1. Es el volcado
de un padrón escolar, no colegios nuevos.

Por eso el perfil temporal **no debe leerse en recuentos absolutos**. Ver
`features/pois.py`: el cociente de localización compara cada zona contra la
ciudad *del mismo año*, así que una importación de alcance nacional mueve
numerador y denominador a la vez y se cancela.

Lo que sí se comprobó que **no** ocurre es el sesgo que haría inservible el eje:
el crecimiento no favorece a los distritos ricos. Miraflores x1.4 frente a San
Juan de Lurigancho x2.1 entre 2018 y 2025.
"""

from __future__ import annotations

import logging
import time
import urllib.request
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

BASE_URL = "https://download.geofabrik.de"

#: Geofabrik exige un User-Agent identificable; el genérico de urllib recibe 403.
USER_AGENT = "TesisSubgrafosUrbanos/1.0 (+contacto en el repositorio)"

#: Margen alrededor del bbox de la red, en grados (~2 km). Un POI justo fuera de
#: la red puede caer dentro de la envolvente bufferizada de un subgrafo del
#: borde; recortar exactamente al bbox lo perdería.
BBOX_MARGIN_DEG = 0.02


def snapshot_url(region: str, year: int) -> str:
    """URL de la instantánea del 1 de enero de `year`.

    `region` es la ruta de Geofabrik sin extensión, p.ej. `south-america/peru`,
    y se usa tal cual: el nombre del fichero repite el último segmento.
    """
    return f"{BASE_URL}/{region}-{year % 100:02d}0101.osm.pbf"


def download_snapshot(region: str, year: int, dest_dir: Path,
                      *, force: bool = False) -> Path:
    """Descarga (o reutiliza) la instantánea de un año."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    dst = dest_dir / f"{region.rsplit('/', 1)[-1]}-{year % 100:02d}0101.osm.pbf"
    if dst.exists() and not force:
        logger.info("  %d: ya está en disco (%.0f MB)", year,
                    dst.stat().st_size / 1e6)
        return dst

    url = snapshot_url(region, year)
    t = time.perf_counter()
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    tmp = dst.with_suffix(".part")
    with urllib.request.urlopen(req, timeout=120) as r, open(tmp, "wb") as fh:
        while chunk := r.read(1 << 20):
            fh.write(chunk)
    tmp.replace(dst)
    mb, secs = dst.stat().st_size / 1e6, time.perf_counter() - t
    logger.info("  %d: %.0f MB en %.0f s (%.1f MB/s)", year, mb, secs, mb / max(secs, 0.1))
    return dst


@dataclass
class SnapshotReport:
    """Qué salió de una instantánea. Los tres destinos, separados."""

    year: int
    kept: int = 0
    excluded: int = 0
    unmapped: int = 0
    by_category: dict[str, int] = field(default_factory=dict)
    top_unmapped: list[tuple[str, int]] = field(default_factory=list)
    elapsed_s: float = 0.0

    @property
    def seen(self) -> int:
        return self.kept + self.excluded + self.unmapped

    @property
    def unmapped_rate(self) -> float:
        return self.unmapped / self.seen if self.seen else 0.0

    def to_dict(self) -> dict:
        return {
            "year": self.year, "kept": self.kept, "excluded": self.excluded,
            "unmapped": self.unmapped,
            "unmapped_rate": round(self.unmapped_rate, 4),
            "by_category": self.by_category,
            "top_unmapped": self.top_unmapped[:15],
            "elapsed_s": round(self.elapsed_s, 1),
        }


def extract_pois(pbf: Path, taxonomy, bbox: tuple[float, float, float, float],
                 year: int) -> tuple[list[dict], SnapshotReport]:
    """POIs clasificados de un `.osm.pbf`, recortados a `bbox`.

    `bbox` es `(lat_min, lon_min, lat_max, lon_max)`.

    Se leen nodos **y vías**: un colegio, un mercado o un parque son polígonos,
    no puntos, y quedarse solo con los nodos perdería justo las categorías de
    mayor extensión. Eso obliga a mantener el índice de localizaciones de los
    nodos (`with_locations`), que es lo que hace que la pasada tarde minutos y
    no segundos.
    """
    import osmium

    from .pois import OSM_KEYS

    lat_min, lon_min, lat_max, lon_max = bbox
    rep = SnapshotReport(year=year)
    unmapped: Counter = Counter()
    rows: list[dict] = []
    t0 = time.perf_counter()

    fp = (osmium.FileProcessor(str(pbf), osmium.osm.NODE | osmium.osm.WAY)
          .with_locations()
          .with_filter(osmium.filter.KeyFilter(*OSM_KEYS)))

    for obj in fp:
        if obj.is_node():
            lat, lon = obj.location.lat, obj.location.lon
        else:
            pts = [(n.location.lat, n.location.lon)
                   for n in obj.nodes if n.location.valid()]
            if not pts:
                continue
            # Centroide en grados. A esta latitud y con zonas de decenas de
            # metros la diferencia con el centroide proyectado es de
            # centímetros, y proyectar millones de vías costaría minutos.
            lat = sum(p[0] for p in pts) / len(pts)
            lon = sum(p[1] for p in pts) / len(pts)
        if not (lat_min <= lat <= lat_max and lon_min <= lon <= lon_max):
            continue

        tags = dict(obj.tags)
        cat = taxonomy.classify(tags)
        if cat == taxonomy.EXCLUDED:
            rep.excluded += 1
            continue
        key = next((k for k in OSM_KEYS if k in tags), "")
        if cat is None:
            rep.unmapped += 1
            if key:
                unmapped[f"{key}={tags[key]}"] += 1
            continue

        rows.append({
            "lat": float(lat), "lon": float(lon), "category": cat,
            "name": str(tags.get("name") or ""),
            "osm_key": key, "osm_value": str(tags.get(key, "")),
            "year": int(year),
        })

    rep.kept = len(rows)
    rep.by_category = {c: 0 for c in taxonomy.categories}
    for r in rows:
        rep.by_category[r["category"]] += 1
    rep.top_unmapped = unmapped.most_common(25)
    rep.elapsed_s = time.perf_counter() - t0
    return rows, rep


def bbox_with_margin(lat_min, lon_min, lat_max, lon_max,
                     margin: float = BBOX_MARGIN_DEG):
    return (lat_min - margin, lon_min - margin, lat_max + margin, lon_max + margin)


def load_yearly(region: str, years: list[int], taxonomy,
                bbox: tuple[float, float, float, float], snapshots_dir: Path,
                *, force: bool = False):
    """DataFrame de POIs con columna `year`, uno por instantánea.

    Devuelve `(DataFrame, [SnapshotReport])`.
    """
    import pandas as pd

    rows: list[dict] = []
    reports: list[SnapshotReport] = []
    logger.info("Instantáneas de %s: %s", region,
                ", ".join(str(y) for y in years))
    for y in years:
        pbf = download_snapshot(region, y, snapshots_dir, force=force)
        chunk, rep = extract_pois(pbf, taxonomy, bbox, y)
        rows.extend(chunk)
        reports.append(rep)
        logger.info("  %d: %s POIs · %s excluidos · %s sin mapear (%.1f %%) · %.0f s",
                    y, f"{rep.kept:,}", f"{rep.excluded:,}", f"{rep.unmapped:,}",
                    100 * rep.unmapped_rate, rep.elapsed_s)
        if rep.unmapped_rate > 0.10:
            logger.warning(
                "  %d: el %.0f %% no encaja en la taxonomía. Más frecuentes: %s",
                y, 100 * rep.unmapped_rate,
                ", ".join(t for t, _ in rep.top_unmapped[:5]))

    df = pd.DataFrame(rows, columns=["lat", "lon", "category", "name",
                                     "osm_key", "osm_value", "year"])
    return df, reports
