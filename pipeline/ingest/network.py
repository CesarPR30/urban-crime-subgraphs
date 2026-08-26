"""Descarga, simplificación y caché de la red vial (PROJECT_SPEC §3.3).

Red transitable de OSMnx con topología simplificada: cada arista es un segmento
real entre dos puntos de decisión (intersección o cul-de-sac).

La descarga es lenta y no determinista en el tiempo, así que el grafo se cachea
en GraphML y no se vuelve a bajar salvo `--force`. La fecha del snapshot queda
registrada en el manifest: si el conteo de nodos difiere de corridas anteriores
es porque OSM cambió, no porque haya un bug (§7.3).
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import networkx as nx
import osmnx as ox

logger = logging.getLogger(__name__)


def fold_highway(tag: object, mapping: dict[str, str], default: str) -> str:
    """Pliega la etiqueta `highway` de OSM a la jerarquía vial del config.

    OSM entrega a veces una lista (una arista simplificada que fusiona varios
    segmentos de clase distinta). En ese caso gana la clase **más alta** de la
    jerarquía presente, que es la que domina el carácter del tramo.
    """
    if tag is None:
        return default
    tags = tag if isinstance(tag, (list, tuple, set)) else [tag]
    order = list(dict.fromkeys(mapping.values()))
    best: str | None = None
    for t in tags:
        cls = mapping.get(str(t))
        if cls is None:
            continue
        if best is None or order.index(cls) < order.index(best):
            best = cls
    return best or default


def annotate(G: nx.MultiDiGraph, cfg) -> nx.MultiDiGraph:
    """Añade a cada arista su clase de jerarquía y garantiza `length` en metros."""
    mapping = cfg.network.osm_to_class
    default = cfg.network.hierarchy_default
    missing_len = 0
    for _, _, data in G.edges(data=True):
        data["road_class"] = fold_highway(data.get("highway"), mapping, default)
        if "length" not in data:
            missing_len += 1
            data["length"] = 0.0
    if missing_len:
        logger.warning("%d aristas sin `length`; puestas a 0", missing_len)
    return G


def to_undirected_simple(G: nx.MultiDiGraph) -> nx.Graph:
    """Colapsa el MultiDiGraph de OSMnx a un grafo simple no dirigido.

    La difusión del campo de densidad (§4.1) mide proximidad *a lo largo del
    tejido vial*, no accesibilidad para un vehículo: no debe respetar sentidos
    únicos. Entre dos nodos unidos por varias aristas paralelas sobrevive la
    más corta, que es la que define su distancia geodésica real.
    """
    H = nx.Graph()
    H.add_nodes_from((n, dict(d)) for n, d in G.nodes(data=True))
    for u, v, data in G.edges(data=True):
        if u == v:
            continue  # los bucles no aportan nada a la geodesia
        length = float(data.get("length", 0.0))
        prev = H.get_edge_data(u, v)
        if prev is None or length < prev["length"]:
            H.add_edge(u, v, length=length, road_class=data.get("road_class", "street"))
    return H


def download(cfg) -> nx.MultiDiGraph:
    """Descarga la red del lugar configurado. Lenta: solo se llama sin caché."""
    logger.info(
        "Descargando red %r (%s) de OSM. Esto tarda varios minutos…",
        cfg.network.place, cfg.network.network_type,
    )
    t0 = time.perf_counter()
    G = ox.graph_from_place(
        cfg.network.place,
        network_type=cfg.network.network_type,
        simplify=cfg.network.simplify,
    )
    logger.info(
        "Descargados %d nodos y %d aristas en %.1f s",
        G.number_of_nodes(), G.number_of_edges(), time.perf_counter() - t0,
    )
    return G


def load_network(cfg, *, force: bool = False) -> tuple[nx.MultiDiGraph, dict]:
    """Devuelve la red vial anotada, usando el caché GraphML si existe.

    Returns:
        `(G, info)` con `info` describiendo el origen y el tamaño del grafo.
    """
    cache = cfg.path(cfg.network.cache)
    info: dict = {"place": cfg.network.place, "network_type": cfg.network.network_type}

    if cache.exists() and not force:
        logger.info("Cargando red desde caché %s", cache)
        t0 = time.perf_counter()
        G = ox.load_graphml(cache)
        info["source"] = "cache"
        info["cached_at"] = time.strftime(
            "%Y-%m-%dT%H:%M:%S", time.localtime(cache.stat().st_mtime)
        )
        logger.info("Red cargada en %.1f s", time.perf_counter() - t0)
    else:
        G = download(cfg)
        cache.parent.mkdir(parents=True, exist_ok=True)
        ox.save_graphml(G, cache)
        info["source"] = "download"
        info["downloaded_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        logger.info("Red cacheada en %s", cache)

    G = annotate(G, cfg)
    info.update(nodes=G.number_of_nodes(), edges=G.number_of_edges())
    return G, info
