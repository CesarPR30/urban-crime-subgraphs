"""Artefactos estáticos para el dashboard (PROJECT_SPEC §8).

El pipeline produce artefactos; el dashboard **solo lee**. Aquí se traduce
`hotspots.json` —que habla en ids de nodo OSM— a geometría que un mapa puede
pintar sin saber nada del grafo.

Cada hotspot sale como tres *features* de un mismo `FeatureCollection`,
separables por la propiedad `kind`:

``area``
    Polígono de la envolvente convexa de sus nodos, bufferizada. Es la misma
    construcción que §5.6 usa para asociar POIs a un subgrafo, así que lo que se
    ve en el mapa es literalmente la zona de la que se extraerán los POIs.
``lines``
    Los segmentos viales del subgrafo, con la geometría real de OSM cuando la
    arista simplificada la trae (si no, recta entre intersecciones).
``seed``
    El nodo semilla, el de más crímenes de la región.

Además se emite el artefacto binario del snapping (`snapping_binary`), que es lo
que permite al dashboard dibujar el crimen donde el pipeline lo pone —sobre un
nodo de la red— y no donde lo pone el CSV.
"""

from __future__ import annotations

import logging
import sys
from array import array
from pathlib import Path

import networkx as nx
import numpy as np
from pyproj import Transformer
from shapely.geometry import LineString, MultiPoint, mapping
from shapely.ops import transform as shp_transform

logger = logging.getLogger(__name__)

PRECISION = 5          # ≈ 1.1 m; suficiente para dibujar


def _round(geom_coords):
    if isinstance(geom_coords[0], (int, float)):
        return [round(float(c), PRECISION) for c in geom_coords[:2]]
    return [_round(c) for c in geom_coords]


def _edge_geometry(G: nx.MultiDiGraph, u: int, v: int) -> list[list[float]]:
    """Coordenadas de la arista `u—v`, con la geometría real si OSM la trae."""
    for a, b in ((u, v), (v, u)):
        if G.has_edge(a, b):
            data = min(G[a][b].values(), key=lambda d: d.get("length", 0.0))
            geom = data.get("geometry")
            if geom is not None:
                coords = list(geom.coords)
                if a != u:
                    coords.reverse()
                return _round(coords)
            break
    return _round([(G.nodes[u]["x"], G.nodes[u]["y"]),
                   (G.nodes[v]["x"], G.nodes[v]["y"])])


def _hull(coords: list[tuple[float, float]], to_m, to_deg, buffer_m: float):
    """Envolvente convexa bufferizada, en grados.

    El buffer se aplica **en el plano proyectado en metros**: bufferizar en
    grados deformaría la forma, porque un grado de longitud no mide lo mismo
    que uno de latitud.
    """
    hull_m = shp_transform(to_m, MultiPoint(coords).convex_hull)
    return shp_transform(to_deg, hull_m.buffer(buffer_m))


def hotspots_geojson(art: dict, G: nx.MultiDiGraph, cfg) -> dict:
    """Traduce el artefacto de hotspots a GeoJSON dibujable."""
    months = sorted({h["month"] for h in art["hotspots"]})
    mindex = {m: i for i, m in enumerate(months)}
    categories = sorted({c for h in art["hotspots"] for c in h["by_category"]})

    # Proyección local en metros para el buffer de la envolvente.
    lats = [G.nodes[n]["y"] for n in list(G.nodes)[:1000]]
    lons = [G.nodes[n]["x"] for n in list(G.nodes)[:1000]]
    zone = int((sum(lons) / len(lons) + 180) // 6) + 1
    epsg = 32600 + zone if sum(lats) / len(lats) >= 0 else 32700 + zone
    to_m = Transformer.from_crs("EPSG:4326", f"EPSG:{epsg}", always_xy=True).transform
    to_deg = Transformer.from_crs(f"EPSG:{epsg}", "EPSG:4326", always_xy=True).transform

    # El MISMO buffer que usa §5.6 para asociar POIs. Si el dibujo y el criterio
    # de asociación divergieran, el usuario vería un área y el perfil hablaría
    # de otra: el mapa dejaría de ser una explicación de los datos.
    buffer_m = float(cfg.pois.buffer_m)

    features: list[dict] = []
    for h in art["hotspots"]:
        coords = [(G.nodes[n]["x"], G.nodes[n]["y"]) for n in h["nodes"]]
        props = {
            "id": h["id"],
            "m": mindex[h["month"]],
            "month": h["month"],
            "rank": int(h["id"].rsplit("h", 1)[-1]),
            "crimes": h["crimes"],
            "n_nodes": h["n_nodes"],
            "density": h["density"],
            "peak_f": h["peak_f"],
            "persistence": h["persistence"],
            "cats": [h["by_category"].get(c, 0) for c in categories],
        }

        hull = _hull(coords, to_m, to_deg, buffer_m)
        features.append({
            "type": "Feature",
            "geometry": {"type": "Polygon",
                         "coordinates": _round(mapping(hull)["coordinates"])},
            "properties": {**props, "kind": "area"},
        })

        lines = [_edge_geometry(G, u, v) for u, v in h["edges"]]
        if not lines:                       # hotspot de un solo nodo
            x, y = coords[0]
            lines = [_round([(x, y), (x, y)])]
        features.append({
            "type": "Feature",
            "geometry": {"type": "MultiLineString", "coordinates": lines},
            "properties": {**props, "kind": "lines"},
        })

        seed = G.nodes[h["seed_node"]]
        features.append({
            "type": "Feature",
            "geometry": {"type": "Point",
                         "coordinates": _round((seed["x"], seed["y"]))},
            "properties": {**props, "kind": "seed"},
        })

    return {
        "type": "FeatureCollection",
        "features": features,
        "meta": {
            "months": months,
            "categories": categories,
            "summary": art["summary"],
            "monthly": art["monthly"],
            "hull_buffer_m": buffer_m,
        },
    }


# Nota (§8): la red vial completa NO se exporta. 29 832 nodos y 49 289 aristas
# son varios MB de GeoJSON que no aportan nada —el mapa base ya dibuja las
# calles—. Lo que hace falta es el tramo concreto que forma cada subgrafo, y eso
# va en sus features `lines`.


# --------------------------------------------------------------------------- #
# Snapping: de coordenada cruda a nodo de la red (§4.0)
# --------------------------------------------------------------------------- #

QUANT_MAX = 65535
#: Marca de «este crimen no llegó a la red». `snap_crimes` devuelve -1.
NO_NODE = 0xFFFFFFFF


def snapping_binary(
    node_ids: list[int],
    node_lat: list[float],
    node_lon: list[float],
    crime_node: np.ndarray,        # ids OSM por crimen; -1 si no snappeó
    *,
    fingerprint: str,
    n_crimes: int,
    stats: dict | None = None,
) -> tuple[bytes, dict]:
    """Empaqueta el resultado del snapping para el dashboard.

    El mapa de calor «por puntos» dibuja la coordenada que trae el CSV; el mapa
    «por nodos» tiene que dibujar dónde acabó cada crimen **después** de
    proyectarlo sobre la red vial. Esa segunda vista es la que deja ver el
    artefacto real del snapping: los nodos no reciben carga uniforme, y las
    intersecciones grandes acumulan los crímenes de todo su entorno.

    En vez de precalcular conteos por nodo —que obligaría a fijar de antemano la
    agregación mes×tipo y pesaría más que los propios crímenes— se exporta la
    **asignación crimen→nodo**. El navegador ya tiene mes y tipo de cada crimen
    en `crimes.bin`, así que puede agregar por nodo *con los mismos filtros que
    ya aplica a los puntos*, y las dos vistas no pueden desincronizarse.

    Disposición del búfer, con `n` crímenes y `N` nodos:

        offset        tipo      campo
        0             Uint32    node[n]     índice en la tabla de nodos, o NO_NODE
        4n            Uint16    lat[N]      cuantizada sobre el bbox de la red
        4n + 2N       Uint16    lon[N]

    El `Uint32` va primero para que su offset quede alineado a 4 bytes; `4n` es
    múltiplo de 4, así que los `Uint16` que siguen también quedan alineados.

    Devuelve `(buffer, meta)`.
    """
    if len(node_ids) != len(node_lat) or len(node_ids) != len(node_lon):
        raise ValueError("las tres tablas de nodos deben tener la misma longitud")
    if len(node_ids) > NO_NODE:
        raise ValueError(f"{len(node_ids)} nodos no caben en Uint32")

    crime_node = np.asarray(crime_node)
    if crime_node.size != n_crimes:
        raise ValueError(
            f"el vector de snapping tiene {crime_node.size} entradas y la carga "
            f"de crímenes {n_crimes}: no son la misma corrida"
        )

    lat0, lat1 = min(node_lat), max(node_lat)
    lon0, lon1 = min(node_lon), max(node_lon)
    dlat = (lat1 - lat0) or 1.0
    dlon = (lon1 - lon0) or 1.0

    # ids OSM → índice denso en la tabla de nodos.
    dense = {osm: i for i, osm in enumerate(node_ids)}
    q_node = array("I", bytes(4 * n_crimes))
    unsnapped = 0
    for i, osm in enumerate(crime_node.tolist()):
        j = dense.get(int(osm), -1) if osm >= 0 else -1
        if j < 0:
            q_node[i] = NO_NODE
            unsnapped += 1
        else:
            q_node[i] = j

    q_lat = array("H", bytes(2 * len(node_ids)))
    q_lon = array("H", bytes(2 * len(node_ids)))
    for i, (la, lo) in enumerate(zip(node_lat, node_lon)):
        q_lat[i] = round((la - lat0) / dlat * QUANT_MAX)
        q_lon[i] = round((lo - lon0) / dlon * QUANT_MAX)

    if sys.byteorder != "little":
        for a in (q_node, q_lat, q_lon):
            a.byteswap()

    buf = b"".join(a.tobytes() for a in (q_node, q_lat, q_lon))

    # Cuántos nodos reciben al menos un crimen. Es la cifra que interesa mirar:
    # la fracción de la red que el fenómeno toca siquiera una vez.
    touched = len({int(v) for v in q_node if v != NO_NODE})

    meta = {
        "n": n_crimes,
        "nodes": len(node_ids),
        "bbox": [lat0, lon0, lat1, lon1],
        "quant_max": QUANT_MAX,
        "no_node": int(NO_NODE),
        "fingerprint": fingerprint,
        "snapped": n_crimes - unsnapped,
        "unsnapped": unsnapped,
        "nodes_touched": touched,
        "stats": stats or {},
        "layout": [
            {"field": "node", "type": "Uint32", "offset": 0},
            {"field": "lat", "type": "Uint16", "offset": 4 * n_crimes},
            {"field": "lon", "type": "Uint16", "offset": 4 * n_crimes + 2 * len(node_ids)},
        ],
    }
    return buf, meta


# --------------------------------------------------------------------------- #
# Puntos de interés (§3.2)
# --------------------------------------------------------------------------- #


def pois_binary(lat, lon, category, categories: list[str],
                labels: dict[str, str] | None = None) -> tuple[bytes, dict]:
    """Empaqueta los POIs clasificados para el mapa.

    Mismo criterio que `crimes.bin`: 5 bytes por POI en vez de ~90 de JSON.

        offset      tipo      campo
        0           Uint16    lat[n]   cuantizada sobre el bbox de los POIs
        2n          Uint16    lon[n]
        4n          Uint8     cat[n]   índice en `categories`

    Los nombres NO van: son 27 000 cadenas de texto, más peso que todo lo demás
    junto, y el mapa muestra categorías, no rótulos.
    """
    lat = np.asarray(lat, dtype=np.float64)
    lon = np.asarray(lon, dtype=np.float64)
    if len(categories) > 256:
        raise ValueError(f"{len(categories)} categorías no caben en Uint8")

    index = {c: i for i, c in enumerate(categories)}
    codes = np.array([index.get(str(c), -1) for c in category], dtype=np.int64)
    keep = codes >= 0
    lat, lon, codes = lat[keep], lon[keep], codes[keep]

    n = lat.size
    lat0, lat1 = (float(lat.min()), float(lat.max())) if n else (0.0, 0.0)
    lon0, lon1 = (float(lon.min()), float(lon.max())) if n else (0.0, 0.0)
    dlat = (lat1 - lat0) or 1.0
    dlon = (lon1 - lon0) or 1.0

    q_lat = array("H", ((lat - lat0) / dlat * QUANT_MAX).round().astype(np.uint16).tobytes())
    q_lon = array("H", ((lon - lon0) / dlon * QUANT_MAX).round().astype(np.uint16).tobytes())
    q_cat = array("B", codes.astype(np.uint8).tobytes())

    if sys.byteorder != "little":
        for a in (q_lat, q_lon):
            a.byteswap()

    buf = b"".join(a.tobytes() for a in (q_lat, q_lon, q_cat))
    meta = {
        "n": int(n),
        "bbox": [lat0, lon0, lat1, lon1],
        "quant_max": QUANT_MAX,
        "categories": list(categories),
        "labels": dict(labels or {}),
        "counts": [int((codes == i).sum()) for i in range(len(categories))],
        "layout": [
            {"field": "lat", "type": "Uint16", "offset": 0},
            {"field": "lon", "type": "Uint16", "offset": 2 * int(n)},
            {"field": "cat", "type": "Uint8", "offset": 4 * int(n)},
        ],
    }
    return buf, meta
