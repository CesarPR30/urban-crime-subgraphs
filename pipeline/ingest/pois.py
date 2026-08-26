"""Descarga, clasificación y caché de puntos de interés (PROJECT_SPEC §3.2).

Cada POI de OSM se reduce a **un punto** —su centroide si la geometría no es
puntual— y se mapea a una de las ocho categorías funcionales mediante el YAML
versionado `poi_taxonomy.yaml`. El mapeo no está en el código a propósito: es lo
que hay que reescribir para llevar el pipeline a otra ciudad, y tiene que poder
auditarse sin leer Python.

La descarga es lenta y no determinista en el tiempo, igual que la de la red
vial, así que se cachea en parquet y no se repite salvo `--force`.

## Qué se descarta y por qué se cuenta

Un POI cuyas etiquetas no encajen en ninguna categoría se descarta. La fracción
de descartes **es la métrica de calidad de la taxonomía**: si sube mucho al
cambiar de ciudad, el YAML se ha quedado corto y hay que extenderlo. Por eso el
reporte la publica en vez de tragársela en silencio.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)

#: Llaves OSM que se consultan. Fijas: son las que nombra §3.2.
OSM_KEYS: tuple[str, ...] = ("amenity", "shop", "leisure")


@dataclass
class Taxonomy:
    """El YAML de §3.2, ya invertido para consultar en O(1)."""

    categories: list[str]
    labels: dict[str, str]
    #: `(llave OSM, valor)` -> categoría funcional.
    lookup: dict[tuple[str, str], str]
    #: `(llave OSM, valor)` descartados a propósito. Ver el YAML.
    excluded: set[tuple[str, str]]
    shop_default: str | None

    #: `classify` devuelve esto para lo que se descarta a propósito, para poder
    #: distinguirlo de lo que no está mapeado (que devuelve `None`).
    EXCLUDED = "__excluded__"

    @classmethod
    def load(cls, path: Path) -> "Taxonomy":
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        cats = list(raw["categories"])
        lookup: dict[tuple[str, str], str] = {}
        for key, by_cat in raw["tags"].items():
            for cat, values in by_cat.items():
                if cat not in cats:
                    raise ValueError(
                        f"{path}: la categoría {cat!r} de {key} no está en `categories`"
                    )
                for v in values:
                    lookup[(key, str(v))] = cat

        excluded = {(key, str(v))
                    for key, values in (raw.get("excluded") or {}).items()
                    for v in values}
        clash = excluded & set(lookup)
        if clash:
            raise ValueError(f"{path}: {sorted(clash)} están clasificados y excluidos")

        default = raw.get("shop_default")
        if default is not None and default not in cats:
            raise ValueError(f"{path}: shop_default {default!r} no está en `categories`")
        return cls(categories=cats, labels=dict(raw.get("labels", {})),
                   lookup=lookup, excluded=excluded, shop_default=default)

    def classify(self, tags: dict[str, object]) -> str | None:
        """Categoría de un POI, `EXCLUDED` si se descarta a propósito, o `None`.

        El orden de `OSM_KEYS` decide los empates: un local con `amenity=cafe` y
        `shop=bakery` es alimentación por las dos vías, pero uno con
        `amenity=pub` y `shop=alcohol` se resuelve por `amenity`, que describe la
        función principal.

        Un elemento excluido por una llave puede seguir clasificándose por otra:
        un centro comercial con `amenity=parking` y `shop=mall` es comercio, no
        un aparcamiento. Por eso la exclusión no corta el bucle.
        """
        excluded = False
        for key in OSM_KEYS:
            value = tags.get(key)
            if value is None or value != value:        # None o NaN de pandas
                continue
            pair = (key, str(value))
            if pair in self.excluded:
                excluded = True
                continue
            cat = self.lookup.get(pair)
            if cat:
                return cat
            if key == "shop" and self.shop_default:
                return self.shop_default
        return self.EXCLUDED if excluded else None


@dataclass
class PoiReport:
    """Resumen auditable de una descarga de POIs."""

    place: str
    source: str = "download"
    downloaded: int = 0
    kept: int = 0
    #: Infraestructura y mobiliario urbano: descartes deliberados (ver el YAML).
    excluded: int = 0
    #: Etiquetas que la taxonomía no conoce. **Esta** mide su cobertura.
    unknown: int = 0
    dropped_no_geometry: int = 0
    by_category: dict[str, int] = field(default_factory=dict)
    #: Las etiquetas desconocidas más frecuentes, para saber qué añadir.
    top_unknown: list[tuple[str, int]] = field(default_factory=list)
    elapsed_s: float = 0.0

    @property
    def unknown_rate(self) -> float:
        """Fracción de lo que la taxonomía *no supo* clasificar.

        El denominador excluye lo descartado a propósito: si no, el indicador
        mediría cuántos aparcamientos tiene la ciudad en vez de si el YAML cubre
        sus usos de suelo.
        """
        base = self.kept + self.unknown
        return self.unknown / base if base else 0.0

    def to_dict(self) -> dict:
        return {
            "place": self.place,
            "source": self.source,
            "downloaded": self.downloaded,
            "kept": self.kept,
            "excluded": self.excluded,
            "unknown": self.unknown,
            "unknown_rate": round(self.unknown_rate, 4),
            "dropped_no_geometry": self.dropped_no_geometry,
            "by_category": self.by_category,
            "top_unknown": [{"tag": t, "n": n} for t, n in self.top_unknown],
            "elapsed_s": round(self.elapsed_s, 2),
        }

    def summary(self) -> str:
        lines = [
            f"POIs de {self.place}  ({self.source})",
            "─" * 62,
            f"  descargados             {self.downloaded:>8,}",
            f"  clasificados            {self.kept:>8,}",
            f"  excluidos a propósito   {self.excluded:>8,}   infraestructura y mobiliario",
            f"  sin mapear              {self.unknown:>8,}   ({self.unknown_rate * 100:.1f} % de los funcionales)",
            f"  sin geometría           {self.dropped_no_geometry:>8,}",
            "",
        ]
        total = self.kept or 1
        for cat, n in sorted(self.by_category.items(), key=lambda kv: -kv[1]):
            lines.append(f"    {cat:<12} {n:>7,}   {n / total * 100:5.1f} %")
        if self.top_unknown:
            lines += ["", "  etiquetas sin mapear más frecuentes:"]
            for tag, n in self.top_unknown[:8]:
                lines.append(f"    {n:>6,}  {tag}")
        return "\n".join(lines)


#: Overpass corta la conexión antes de servir `amenity` + `shop` + `leisure` de
#: una ciudad entera en una sola petición. 10 min por llave, una llave por
#: petición.
REQUEST_TIMEOUT_S = 600


def download(cfg, taxonomy: Taxonomy):
    """Descarga los POIs del lugar configurado y los clasifica.

    Devuelve `(DataFrame, PoiReport)` con columnas `lat`, `lon`, `category`,
    `name`, `osm_key`, `osm_value`.

    Se consulta **una llave OSM por petición** y no las tres juntas: la consulta
    combinada sobre Chicago agota el tiempo de Overpass. Además, así una llave
    que falle no tira la descarga entera.
    """
    import osmnx as ox
    import pandas as pd

    ox.settings.requests_timeout = REQUEST_TIMEOUT_S

    t0 = time.perf_counter()
    logger.info("Descargando POIs de %r desde OSM. Esto tarda varios minutos…",
                cfg.network.place)

    parts = []
    for key in OSM_KEYS:
        t1 = time.perf_counter()
        try:
            part = ox.features_from_place(cfg.network.place, tags={key: True})
        except Exception as exc:
            logger.warning("La llave %r falló (%s); se continúa sin ella.", key, exc)
            continue
        logger.info("  %-8s %6d elementos en %.0f s", key, len(part),
                    time.perf_counter() - t1)
        parts.append(part)

    if not parts:
        raise RuntimeError("Ninguna llave OSM se pudo descargar")

    import geopandas as gpd
    gdf = gpd.GeoDataFrame(pd.concat(parts))
    # Un mismo elemento puede traer `amenity` y `shop` a la vez y aparecer en
    # dos de las respuestas. Es un POI, no dos.
    gdf = gdf[~gdf.index.duplicated(keep="first")]
    logger.info("Descargados %d elementos únicos en %.1f s",
                len(gdf), time.perf_counter() - t0)

    report = PoiReport(place=cfg.network.place, downloaded=len(gdf))

    # Todo a un punto: el centroide en un plano métrico, no en grados, para que
    # el centroide de un polígono alargado caiga donde debe.
    geom = gdf.geometry
    ok = geom.notna() & ~geom.is_empty
    report.dropped_no_geometry = int((~ok).sum())
    gdf = gdf[ok]

    pts = gdf.geometry.to_crs(gdf.estimate_utm_crs()).centroid.to_crs("EPSG:4326")

    from collections import Counter

    rows = []
    unknown = Counter()
    for (_, tags), pt in zip(gdf.iterrows(), pts):
        cat = taxonomy.classify(tags)
        if cat == taxonomy.EXCLUDED:
            report.excluded += 1
            continue
        if cat is None:
            report.unknown += 1
            for k in OSM_KEYS:
                v = tags.get(k)
                if v is not None and v == v:
                    unknown[f"{k}={v}"] += 1
                    break
            continue
        key = next((k for k in OSM_KEYS
                    if tags.get(k) is not None and tags.get(k) == tags.get(k)), "")
        rows.append({
            "lat": float(pt.y),
            "lon": float(pt.x),
            "category": cat,
            "name": str(tags.get("name") or ""),
            "osm_key": key,
            "osm_value": str(tags.get(key, "")),
        })

    df = pd.DataFrame(rows, columns=["lat", "lon", "category", "name",
                                     "osm_key", "osm_value"])
    report.kept = len(df)
    report.by_category = {c: int((df["category"] == c).sum())
                          for c in taxonomy.categories}
    report.top_unknown = unknown.most_common(25)
    report.elapsed_s = time.perf_counter() - t0

    if report.unknown_rate > 0.10:
        logger.warning(
            "El %.0f %% de los POIs funcionales no encaja en la taxonomía. "
            "Revisa `poi_taxonomy.yaml`: se queda corta para esta ciudad. "
            "Las más frecuentes: %s",
            report.unknown_rate * 100,
            ", ".join(t for t, _ in report.top_unknown[:5]),
        )
    return df, report


def load_pois(cfg, *, force: bool = False):
    """POIs clasificados, usando el caché parquet si existe."""
    import pandas as pd

    taxonomy = Taxonomy.load(cfg.path(cfg.pois.taxonomy))
    cache = cfg.path(cfg.pois.cache)

    if cache.exists() and not force:
        logger.info("Cargando POIs desde caché %s", cache)
        df = pd.read_parquet(cache)
        report = PoiReport(
            place=cfg.network.place, source="cache",
            downloaded=len(df), kept=len(df),
            by_category={c: int((df["category"] == c).sum())
                         for c in taxonomy.categories},
        )
        return df, report, taxonomy

    df, report = download(cfg, taxonomy)
    cache.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cache, index=False)
    logger.info("POIs cacheados en %s", cache)
    return df, report, taxonomy
