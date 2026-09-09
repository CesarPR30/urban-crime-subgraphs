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
from collections import Counter
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
    #: Año -> nº de POIs, solo con fuente temporal. Vacío = una sola instantánea.
    years: dict[int, int] = field(default_factory=dict)
    #: Un `SnapshotReport.to_dict()` por año, para auditar cada instantánea.
    snapshots: list[dict] = field(default_factory=list)
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
            "years": {str(y): n for y, n in self.years.items()},
            "snapshots": self.snapshots,
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
        if self.years:
            lines += ["", "  Por instantánea anual"]
            for y, n in sorted(self.years.items()):
                lines.append(f"    {y}         {n:>7,}")
            lines.append(
                "  Los recuentos por año mezclan cambio urbano con actividad de")
            lines.append(
                "  mapeo; el contraste se hace por cociente de localización.")
        if self.top_unknown:
            lines += ["", "  etiquetas sin mapear más frecuentes:"]
            for tag, n in self.top_unknown[:8]:
                lines.append(f"    {n:>6,}  {tag}")
        return "\n".join(lines)


#: Overpass corta la conexión antes de servir `amenity` + `shop` + `leisure` de
#: una ciudad entera en una sola petición. Una llave por petición.
#:
#: Cinco minutos y no diez: con `KEY_BUDGET_S` acotando el total, un timeout
#: largo gasta todo el presupuesto en un solo intento colgado en vez de
#: repartirlo entre espejos. Overpass, cuando va a responder, responde dentro
#: de este margen; lo que pasa de aquí es una conexión que ya no va a devolver
#: nada.
REQUEST_TIMEOUT_S = 300

#: Identificación en las peticiones a Overpass.
#:
#: **No es cortesía: es el motivo por el que las descargas fallaban.** Con el
#: User-Agent que trae OSMnx por defecto, `overpass.kumi.systems` devuelve
#: `429` con el texto «Please include a meaningful User-Agent string with your
#: requests to avoid rate-limiting», y `overpass-api.de` responde `406`. OSMnx
#: interpreta el 429 como «servidor ocupado», espera y reintenta dentro de la
#: misma llamada, así que el síntoma no era un error sino una descarga que se
#: quedaba media hora sin avanzar y sin decir por qué.
#:
#: Las instancias públicas piden un agente que identifique al cliente y una
#: forma de contacto. Ver https://dev.overpass-api.de/overpass-doc/ y la
#: política de uso de cada espejo.
USER_AGENT = (
    "urban-crime-subgraphs/1.0 (research; +https://github.com/CesarPR30) "
    "osmnx"
)

#: Instancias de Overpass a las que preguntar, en orden de preferencia.
#:
#: **Solo instancias con cobertura mundial.** `overpass.osm.ch` estuvo aquí y
#: hubo que sacarlo: sirve una base regional de Suiza y responde `200` con cero
#: elementos a cualquier consulta sobre Lima. Un espejo que falla se reintenta
#: en otro; un espejo que contesta «no hay nada» se cree, y el analisis sale
#: adelante sin POIs y sin avisar. Antes de anadir un espejo hay que
#: comprobar que devuelve elementos para la ciudad del dataset, no solo que
#: responde.
#:
#: Todas las que quedan sirven la misma base mundial, asi que rotar entre ellas
#: no cambia el resultado, solo la probabilidad de obtenerlo.
OVERPASS_MIRRORS: tuple[str, ...] = (
    "https://overpass-api.de/api",
    "https://overpass.kumi.systems/api",
    "https://overpass.private.coffee/api",
)

#: Intentos por llave, repartidos entre los espejos. Una llave que agote todos
#: se salta con aviso: el reporte de POIs dirá qué falta.
MAX_ATTEMPTS = 4

#: Espera entre intentos. Crece para dar tiempo a que la cuota se reponga; no
#: es exponencial pura porque la cuota de Overpass se recupera en decenas de
#: segundos, no en minutos.
BACKOFF_S: tuple[float, ...] = (5.0, 20.0, 45.0, 90.0, 120.0)

#: Lado de la rejilla del plan B: la ciudad se parte en `TILE_GRID` ×
#: `TILE_GRID` celdas y cada una se pide por separado.
#:
#: Overpass no acota el trabajo por área sino por elementos devueltos, así que
#: la consulta que falla sobre un polígono grande **sí** entra troceada. Sobre
#: la Provincia de Lima, `amenity` sobre el polígono completo devuelve
#: `InsufficientResponseError` tras seis minutos; en dieciseisavos entra sin
#: despeinarse. 4×4 es el compromiso: bastantes celdas para que ninguna sea
#: pesada, pocas para no gastar la cuota en el ida y vuelta.
TILE_GRID = 4

#: Presupuesto de reloj por llave. **Este es el límite que manda**, no el número
#: de intentos.
#:
#: Sin él, `MAX_ATTEMPTS` × `requests_timeout` × 3 llaves da un peor caso de
#: horas, y el paso `features` —que descarga POIs al final, después de calcular
#: los embeddings— se queda colgado indefinidamente detrás de una descarga
#: opcional. Un cuarto de hora por llave es de sobra cuando Overpass responde;
#: cuando no responde, ninguna cantidad de espera lo arregla.
#:
#: **No es un límite duro y no puede serlo desde aquí.** Se comprueba *entre*
#: intentos, y OSMnx, al recibir un `429` o un `504`, duerme y reintenta dentro
#: de la misma llamada: un intento que caiga en esa espera se pasa del
#: presupuesto sin que haya forma de interrumpirlo sin sacar la descarga a otro
#: proceso. Para cuando eso importe —una corrida que no puede esperar— está la
#: bandera `--no-pois` de `features` y `export`.
KEY_BUDGET_S = 900.0


#: Cuánto se espera a que una IP acepte la conexión antes de probar la
#: siguiente. Corto a propósito: aquí solo se comprueba que el puerto abre.
PROBE_TIMEOUT_S = 6.0

#: hostname -> IP verificada. Se resuelve una vez por proceso.
_PINNED_IPS: dict[str, str] = {}


def _reachable_ip(hostname: str, port: int = 443) -> str | None:
    """Primera dirección del registro DNS que acepta una conexión TCP.

    OSMnx fija **una sola** IP por host (`_http._config_dns`) parcheando
    `socket.getaddrinfo`, para que la comprobación de cuota y la consulta caigan
    en la misma máquina y no se viole el reparto de turnos del servidor. La
    elige con `socket.gethostbyname`, que devuelve la primera del registro sin
    mirar si responde.

    `overpass-api.de` publica dos direcciones y desde algunas redes solo una
    acepta conexiones. Cuando el resolutor devuelve primero la muerta, OSMnx la
    fija y **todas** las peticiones de la corrida mueren en un ConnectTimeout
    idéntico, sin failover posible: el parche de `getaddrinfo` es precisamente
    lo que impide a urllib3 pasar a la segunda.

    Esta función hace lo que falta —probar antes de fijar— y devuelve la IP
    buena. `None` si ninguna responde.
    """
    import socket

    try:
        infos = socket.getaddrinfo(hostname, port, socket.AF_INET, socket.SOCK_STREAM)
    except OSError:
        return None
    seen: list[str] = []
    for info in infos:
        ip = info[4][0]
        if ip in seen:
            continue
        seen.append(ip)
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(PROBE_TIMEOUT_S)
        try:
            sock.connect((ip, port))
            return ip
        except OSError:
            continue
        finally:
            sock.close()
    return None


def _install_dns_pin(ox) -> None:
    """Hace que el pinado de OSMnx caiga en una IP que sí responde.

    Envuelve `socket.gethostbyname` para los hosts de `OVERPASS_MIRRORS`: OSMnx
    lo llama dentro de `_config_dns` y se queda con lo que devuelva. Sondear
    aquí es lo que convierte «la primera del registro» en «la primera que
    contesta».

    Se instala una sola vez y no toca el resto de resoluciones del proceso.
    """
    import socket

    if getattr(socket.gethostbyname, "_crimepipe_pinned", False):
        return

    hosts = {m.split("//", 1)[-1].split("/", 1)[0] for m in OVERPASS_MIRRORS}
    original = socket.gethostbyname

    def gethostbyname(host: str) -> str:
        if host in hosts:
            if host not in _PINNED_IPS:
                ip = _reachable_ip(host)
                if ip:
                    _PINNED_IPS[host] = ip
                    logger.debug("DNS: %s fijado a %s (verificado)", host, ip)
                else:
                    logger.warning("DNS: ninguna IP de %s acepta conexión", host)
            if host in _PINNED_IPS:
                return _PINNED_IPS[host]
        return original(host)

    gethostbyname._crimepipe_pinned = True
    socket.gethostbyname = gethostbyname


def _fetch_key(ox, place: str, key: str):
    """Descarga una llave OSM rotando espejos y reintentando.

    Devuelve el GeoDataFrame, o `None` si ningún intento prospera. Rota el
    espejo en *cada* intento, no solo al fallar el último: si la instancia
    actual está racionando, insistir en ella es tiempo perdido.
    """
    deadline = time.perf_counter() + KEY_BUDGET_S
    last: Exception | None = None
    for attempt in range(MAX_ATTEMPTS):
        if time.perf_counter() >= deadline:
            logger.warning("  %-8s agotó su presupuesto de %.0f s; se abandona.",
                           key, KEY_BUDGET_S)
            break
        mirror = OVERPASS_MIRRORS[attempt % len(OVERPASS_MIRRORS)]
        ox.settings.overpass_url = mirror
        t1 = time.perf_counter()
        try:
            part = ox.features_from_place(place, tags={key: True})
            logger.info("  %-8s %6d elementos en %.0f s  (%s)", key, len(part),
                        time.perf_counter() - t1, mirror)
            return part
        except Exception as exc:      # noqa: BLE001 — cualquier fallo es reintentable
            last = exc
            wait = BACKOFF_S[min(attempt, len(BACKOFF_S) - 1)]
            logger.warning(
                "  %-8s intento %d/%d falló en %s tras %.0f s (%s); "
                "reintento en %.0f s",
                key, attempt + 1, MAX_ATTEMPTS, mirror,
                time.perf_counter() - t1, type(exc).__name__, wait,
            )
            if attempt < MAX_ATTEMPTS - 1 and time.perf_counter() + wait < deadline:
                time.sleep(wait)
    logger.warning("La llave %r no se pudo descargar (%s); se continúa sin ella.",
                   key, last)
    return None


def _fetch_key_tiled(ox, place: str, key: str):
    """Plan B: la misma llave, pedida por celdas de una rejilla.

    Se usa cuando la consulta sobre el polígono completo no entra. La ciudad se
    divide en `TILE_GRID`² celdas, se intersecta cada una con el polígono real
    del lugar —para no arrastrar POIs de fuera de la jurisdicción, que es justo
    lo que el recorte del dataset excluye— y se pide una por una.

    Una celda vacía o fallida no aborta el resto: se registra y se sigue. Un
    hueco en el mapa de POIs degrada el perfil funcional de los subgrafos de esa
    zona, pero perder las quince celdas restantes por culpa de una es peor.

    Devuelve el GeoDataFrame concatenado, o `None` si ninguna celda respondió.
    """
    import pandas as pd
    from shapely.geometry import box

    try:
        poly = ox.geocode_to_gdf(place).geometry.iloc[0]
    except Exception as exc:      # noqa: BLE001
        logger.warning("  %-8s no se pudo geocodificar %r para trocear (%s)",
                       key, place, exc)
        return None

    west, south, east, north = poly.bounds
    dx = (east - west) / TILE_GRID
    dy = (north - south) / TILE_GRID

    parts, ok, empty, failed = [], 0, 0, 0
    t0 = time.perf_counter()
    for i in range(TILE_GRID):
        for j in range(TILE_GRID):
            cell = box(west + i * dx, south + j * dy,
                       west + (i + 1) * dx, south + (j + 1) * dy)
            piece = cell.intersection(poly)
            if piece.is_empty:
                continue
            # Rotar el espejo por celda reparte la cuota en vez de agotar la de
            # una sola instancia en las primeras celdas.
            ox.settings.overpass_url = OVERPASS_MIRRORS[
                (i * TILE_GRID + j) % len(OVERPASS_MIRRORS)]
            try:
                part = ox.features_from_polygon(piece, tags={key: True})
            except Exception as exc:      # noqa: BLE001
                # OSMnx lanza si la celda no tiene ni un elemento; eso no es un
                # fallo de red y no debe contarse como tal.
                if type(exc).__name__ == "InsufficientResponseError":
                    empty += 1
                else:
                    failed += 1
                    logger.debug("    celda (%d,%d) de %s falló: %s", i, j, key, exc)
                continue
            ok += 1
            parts.append(part)

    if not parts:
        return None
    logger.info("  %-8s %6d elementos en %.0f s  (troceado %dx%d: %d celdas con "
                "datos, %d vacías, %d fallidas)",
                key, sum(len(p) for p in parts), time.perf_counter() - t0,
                TILE_GRID, TILE_GRID, ok, empty, failed)
    return pd.concat(parts)


def download(cfg, taxonomy: Taxonomy):
    """Descarga los POIs del lugar configurado y los clasifica.

    Devuelve `(DataFrame, PoiReport)` con columnas `lat`, `lon`, `category`,
    `name`, `osm_key`, `osm_value`.

    Se consulta **una llave OSM por petición** y no las tres juntas: la consulta
    combinada sobre Chicago agota el tiempo de Overpass. Además, así una llave
    que falle no tira la descarga entera. Cada llave se reintenta rotando
    espejos (ver `_fetch_key`).
    """
    import osmnx as ox
    import pandas as pd

    ox.settings.requests_timeout = REQUEST_TIMEOUT_S
    ox.settings.http_user_agent = USER_AGENT
    # OSMnx resuelve los nombres por DNS-over-HTTPS y **se queda con la primera
    # direccion de la respuesta**. `overpass-api.de` publica dos (65.109.112.52
    # y 162.55.144.139) y desde algunas redes solo la segunda acepta
    # conexiones: el resolutor elige siempre la primera y toda descarga muere
    # en un ConnectTimeout de 102 s, reproducible y sin explicacion visible.
    #
    # El resolutor del sistema prueba **todas** las direcciones del registro,
    # que es justo el comportamiento que hace falta. El DoH de OSMnx existe
    # para saltarse bloqueos por DNS; aqui no hay ninguno que saltarse, y el
    # precio de usarlo es perder el failover entre IPs.
    ox.settings.doh_url_template = None
    _install_dns_pin(ox)

    t0 = time.perf_counter()
    logger.info("Descargando POIs de %r desde OSM. Esto tarda varios minutos…",
                cfg.network.place)

    parts = []
    failed: list[str] = []
    for key in OSM_KEYS:
        part = _fetch_key(ox, cfg.network.place, key)
        if part is None:
            logger.info("  %-8s no entra de una pieza; se pide troceado.", key)
            part = _fetch_key_tiled(ox, cfg.network.place, key)
        if part is None:
            failed.append(key)
        else:
            parts.append(part)

    if not parts:
        raise RuntimeError(
            f"Ninguna llave OSM se pudo descargar tras {MAX_ATTEMPTS} intentos "
            f"por llave sobre {len(OVERPASS_MIRRORS)} espejos. Overpass suele "
            "estar racionando; vuelve a intentarlo en unos minutos."
        )
    if failed:
        # Sale por warning y no en silencio: un POI que falta no es un POI que
        # no existe, y la fracción de descartes del reporte dejaría de medir la
        # calidad de la taxonomía si además absorbiese llaves ausentes.
        logger.warning(
            "POIs incompletos: falta la llave %s. El perfil funcional de los "
            "subgrafos se calcula sin ella. Reejecuta `crimepipe pois --force` "
            "cuando Overpass responda.", ", ".join(failed),
        )

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


def _network_bbox(cfg):
    """Recuadro de la red vial, con margen. `(lat_min, lon_min, lat_max, lon_max)`.

    Sale de la red y no del bbox de los crímenes porque los POIs se asocian a
    subgrafos, que viven sobre la red: recortar por donde hay crimen dejaría sin
    POIs a las zonas de red que aún no han sido hotspot.
    """
    from .geofabrik import bbox_with_margin
    from .network import load_network

    G, _ = load_network(cfg, force=False)
    ys = [d["y"] for _, d in G.nodes(data=True)]
    xs = [d["x"] for _, d in G.nodes(data=True)]
    return bbox_with_margin(min(ys), min(xs), max(ys), max(xs))


def _download_geofabrik(cfg, taxonomy, *, force: bool):
    """Ruta Geofabrik: una instantánea por año, o solo la más reciente."""
    from .geofabrik import load_yearly

    years = list(cfg.pois.years) or [_latest_year()]
    df, reports = load_yearly(
        cfg.pois.region, years, taxonomy, _network_bbox(cfg),
        cfg.path(cfg.pois.snapshots_dir), force=force,
    )
    report = PoiReport(
        place=f"{cfg.pois.region} (Geofabrik, {years[0]}..{years[-1]})",
        source="geofabrik", downloaded=len(df), kept=len(df),
        excluded=sum(r.excluded for r in reports),
        unknown=sum(r.unmapped for r in reports),
        by_category={c: int((df["category"] == c).sum())
                     for c in taxonomy.categories},
    )
    report.top_unknown = Counter(
        dict(t for r in reports for t in r.top_unmapped)).most_common(25)
    report.snapshots = [r.to_dict() for r in reports]
    return df, report


def _latest_year() -> int:
    """Año de la instantánea más reciente que Geofabrik puede tener publicada.

    La del 1 de enero del año en curso siempre existe; la del siguiente, no.
    """
    from datetime import date
    return date.today().year


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
        if "year" in df.columns:
            report.years = {int(y): int(n) for y, n
                            in df["year"].value_counts().sort_index().items()}
        return df, report, taxonomy

    if cfg.pois.source == "geofabrik":
        df, report = _download_geofabrik(cfg, taxonomy, force=force)
        if "year" in df.columns:
            report.years = {int(y): int(n) for y, n
                            in df["year"].value_counts().sort_index().items()}
    else:
        df, report = download(cfg, taxonomy)
    cache.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cache, index=False)
    logger.info("POIs cacheados en %s", cache)
    return df, report, taxonomy
