"""Proyección de crímenes sobre el grafo vial (PROJECT_SPEC §4.0).

Dos pasos, **y el orden importa**:

1. Localizar la **arista** geométricamente más cercana al punto.
2. Asignar el crimen al **más cercano de los dos nodos extremos** de esa arista.

Ir directo al nodo más cercano asignaría crímenes a nodos de otra calle: un
punto a mitad de cuadra puede tener como nodo más próximo la esquina de la
calle paralela, aunque su arista sea inequívocamente la de su propia calle.
Es una fuente sistemática de error en la construcción del campo, no un detalle
(§11.2).

La búsqueda de la arista más cercana se hace con el índice espacial vectorizado
de OSMnx sobre el grafo **proyectado a UTM**, no con una búsqueda lineal: a
escala real O(n·m) es inviable (§11.3).
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

import networkx as nx
import numpy as np
import osmnx as ox
from pyproj import Transformer

logger = logging.getLogger(__name__)

CHUNK = 25_000


def haversine(
    lat1: np.ndarray, lon1: np.ndarray,
    lat2: np.ndarray, lon2: np.ndarray,
    radius_m: float = 6_371_000.0,
) -> np.ndarray:
    """Distancia geodésica sobre la esfera, en metros.

    .. math::
        a = \\sin^2(\\Delta\\varphi/2)
            + \\cos\\varphi_1 \\cos\\varphi_2 \\sin^2(\\Delta\\lambda/2)

        d = 2R \\cdot \\arctan\\!\\left(\\sqrt{a} \\,/\\, \\sqrt{1-a}\\right)

    con :math:`\\varphi` latitud y :math:`\\lambda` longitud en radianes y
    :math:`R` el radio terrestre.
    """
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dphi = p2 - p1
    dlam = np.radians(lon2) - np.radians(lon1)
    a = np.sin(dphi / 2.0) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dlam / 2.0) ** 2
    a = np.clip(a, 0.0, 1.0)
    return 2.0 * radius_m * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))


@dataclass
class SnapResult:
    """Salida del snapping: un nodo por crimen, más el resumen del paso."""

    node: np.ndarray          # id de nodo OSM por crimen; -1 si quedó fuera
    dist_edge_m: np.ndarray   # distancia del punto a la arista elegida
    dist_node_m: np.ndarray   # distancia del punto al nodo asignado
    assigned: int = 0
    dropped_far: int = 0
    elapsed_s: float = 0.0
    stats: dict = field(default_factory=dict)

    @property
    def mask(self) -> np.ndarray:
        """Máscara de crímenes efectivamente asignados a la red."""
        return self.node >= 0


def snap_crimes(
    G: nx.MultiDiGraph,
    lats: np.ndarray,
    lons: np.ndarray,
    cfg,
) -> SnapResult:
    """Asigna cada crimen a un nodo de la red por la vía arista → extremo.

    Args:
        G: red vial de OSMnx en WGS84 (sin proyectar).
        lats, lons: coordenadas de los crímenes.

    Returns:
        Un `SnapResult` con el nodo asignado por crimen.
    """
    t0 = time.perf_counter()
    lats = np.asarray(lats, dtype=np.float64)
    lons = np.asarray(lons, dtype=np.float64)
    n = lats.size

    # (1) Arista más cercana. Se busca sobre el grafo proyectado a UTM para que
    #     la distancia esté en metros y no en grados, que no son isótropos.
    logger.info("Proyectando la red a UTM para la búsqueda espacial…")
    Gp = ox.project_graph(G)
    crs = Gp.graph["crs"]
    tx = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    xs, ys = tx.transform(lons, lats)

    logger.info("Buscando arista más cercana para %s crímenes…", f"{n:,}")
    edges = np.empty((n, 2), dtype=np.int64)
    dist_edge = np.empty(n, dtype=np.float64)
    for start in range(0, n, CHUNK):
        end = min(n, start + CHUNK)
        ne, dd = ox.distance.nearest_edges(
            Gp, X=xs[start:end], Y=ys[start:end], return_dist=True
        )
        edges[start:end, 0] = [e[0] for e in ne]
        edges[start:end, 1] = [e[1] for e in ne]
        dist_edge[start:end] = dd
        logger.info("  %s / %s", f"{end:,}", f"{n:,}")

    # (2) El más cercano de los dos extremos, con haversine sobre WGS84.
    node_lat = np.array([G.nodes[u]["y"] for u in edges[:, 0]])
    node_lon = np.array([G.nodes[u]["x"] for u in edges[:, 0]])
    du = haversine(lats, lons, node_lat, node_lon, cfg.snapping.earth_radius_m)

    node_lat = np.array([G.nodes[v]["y"] for v in edges[:, 1]])
    node_lon = np.array([G.nodes[v]["x"] for v in edges[:, 1]])
    dv = haversine(lats, lons, node_lat, node_lon, cfg.snapping.earth_radius_m)

    pick_u = du <= dv
    node = np.where(pick_u, edges[:, 0], edges[:, 1])
    dist_node = np.where(pick_u, du, dv)

    # Crímenes demasiado lejos de cualquier arista: fuera de la red transitable
    # (interior de un parque, muelle, error de geocodificación).
    too_far = dist_edge > cfg.snapping.max_distance_m
    node = np.where(too_far, -1, node)

    res = SnapResult(
        node=node,
        dist_edge_m=dist_edge,
        dist_node_m=np.where(too_far, np.nan, dist_node),
        assigned=int((~too_far).sum()),
        dropped_far=int(too_far.sum()),
        elapsed_s=time.perf_counter() - t0,
    )
    ok = dist_edge[~too_far]
    # La distancia al NODO es otra cosa que la distancia a la arista, y mucho
    # mayor: el crimen cae sobre la calle (mediana < 1 m) pero la intersección
    # más cercana está a media manzana. Es el desplazamiento real que sufre el
    # dato al entrar al grafo, así que se mide y se publica en vez de dejarlo
    # implícito.
    okn = dist_node[~too_far]
    res.stats = {
        "crs": str(crs),
        "max_distance_m": cfg.snapping.max_distance_m,
        "dist_edge_median_m": float(np.median(ok)) if ok.size else None,
        "dist_edge_p95_m": float(np.percentile(ok, 95)) if ok.size else None,
        "dist_edge_max_m": float(ok.max()) if ok.size else None,
        "dist_node_median_m": float(np.median(okn)) if okn.size else None,
        "dist_node_p95_m": float(np.percentile(okn, 95)) if okn.size else None,
        "dist_node_max_m": float(okn.max()) if okn.size else None,
        "nodes_touched": int(np.unique(node[node >= 0]).size),
    }
    logger.info(
        "Snapping: %s asignados, %s descartados por distancia (>%.0f m) en %.1f s",
        f"{res.assigned:,}", f"{res.dropped_far:,}",
        cfg.snapping.max_distance_m, res.elapsed_s,
    )
    return res


def month_counts(
    nodes: np.ndarray, months: np.ndarray, n_months: int
) -> dict[int, np.ndarray]:
    """Agrega a `c_m(v)`: crímenes por nodo y mes (§4.0, salida).

    Returns:
        `{mes -> {node_id -> conteo}}` como arrays paralelos por mes, en un
        dict `mes -> (nodes, counts)`.
    """
    out: dict[int, np.ndarray] = {}
    valid = nodes >= 0
    for m in range(n_months):
        sel = valid & (months == m)
        uniq, cnt = np.unique(nodes[sel], return_counts=True)
        out[m] = (uniq, cnt)
    return out
