"""Descriptor estructural de 12 dimensiones (PROJECT_SPEC §5.2).

Ocho dimensiones de conectividad y cuatro de geometría métrica. **Ninguna
derivada de crimen**: ver el docstring del paquete y §5.1.

El bloque métrico no es decorativo. La conectividad sola no distingue una cadena
recta de una en forma de L: las dos son el grafo camino `P_n`, con el mismo
histograma de grados y la misma densidad. Lo que las separa es dónde caen sus
nodos en el plano, y para eso hay que proyectar a metros: en grados, un tramo
este-oeste y uno norte-sur de la misma longitud real miden distinto.

> **Nota de §5.1:** el descriptor del código anterior tenía 14 dims, dos de ellas
> derivadas de crimen (fracción de crímenes en el nodo pico e intensidad log por
> nodo). Están fuera. El bloque de conectividad queda en 8 y el total en 12.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import networkx as nx
import numpy as np
from pyproj import Transformer

logger = logging.getLogger(__name__)

#: Nombres de las 12 columnas, en el orden en que salen de `descriptor`.
CONNECTIVITY_DIMS: tuple[str, ...] = (
    "log_n_nodes",
    "log_n_edges",
    "density",
    "mean_degree",
    "deg_frac_1",
    "deg_frac_2",
    "deg_frac_3",
    "deg_frac_4plus",
)
GEOMETRY_DIMS: tuple[str, ...] = (
    "log_mean_edge_len",
    "edge_len_cv",
    "log_radius_gyration",
    "elongation",
)
STRUCTURAL_DIMS: tuple[str, ...] = CONNECTIVITY_DIMS + GEOMETRY_DIMS


@dataclass(slots=True)
class Subgraph:
    """Un hotspot reducido a lo que la caracterización necesita.

    Deliberadamente **no** guarda crímenes: quien tenga un `Subgraph` en la mano
    no puede contaminar la similitud aunque quiera.
    """

    id: str
    month: str
    nodes: list[int]                 # ids OSM, en orden estable
    edges: list[tuple[int, int]]     # aristas inducidas, como índices OSM
    xy: np.ndarray                   # (n, 2) coordenadas proyectadas en metros
    edge_class: list[str]            # clase vial de cada arista
    #: CRS del plano de `xy`. Viaja con el subgrafo porque cualquier otra cosa
    #: que haya que llevar a ese plano —los POIs de §5.6— tiene que usar
    #: exactamente el mismo, no volver a elegir zona por su cuenta.
    epsg: int = 0

    @property
    def n(self) -> int:
        return len(self.nodes)

    def graph(self) -> nx.Graph:
        """Grafo simple con los nodos renumerados a 0..n−1.

        La renumeración es intencionada: el embedding no debe poder distinguir
        dos subgrafos isomorfos por el id OSM de sus nodos (§9, criterio de
        aceptación de la fase E).

        Lleva además, como atributos, la geometría (`xy` por nodo, en metros del
        plano proyectado) y la clase vial (`cls` por arista). No son datos de
        crimen —§5.1 sigue a salvo— y los métodos de embedding que sí miran la
        forma métrica (WWL, scattering) los usan; graph2vec y netlsd los ignoran.
        """
        index = {osm: i for i, osm in enumerate(self.nodes)}
        g = nx.Graph()
        g.add_nodes_from(range(self.n))
        for i in range(self.n):
            g.nodes[i]["xy"] = (float(self.xy[i, 0]), float(self.xy[i, 1]))
        classes = self.edge_class or [""] * len(self.edges)
        for (u, v), cls in zip(self.edges, classes):
            g.add_edge(index[u], index[v], cls=cls)
        return g

    def degrees(self) -> np.ndarray:
        deg = np.zeros(self.n, dtype=np.int64)
        index = {osm: i for i, osm in enumerate(self.nodes)}
        for u, v in self.edges:
            deg[index[u]] += 1
            deg[index[v]] += 1
        return deg

    def edge_lengths(self) -> np.ndarray:
        """Distancia euclídea entre extremos, en metros del plano proyectado."""
        if not self.edges:
            return np.zeros(0, dtype=np.float64)
        index = {osm: i for i, osm in enumerate(self.nodes)}
        a = np.array([index[u] for u, _ in self.edges])
        b = np.array([index[v] for _, v in self.edges])
        return np.linalg.norm(self.xy[a] - self.xy[b], axis=1)


def local_epsg(lats, lons) -> int:
    """Zona UTM en la que caen los datos. Es la misma elección que `export`."""
    lat_c = float(np.mean(lats))
    lon_c = float(np.mean(lons))
    zone = int((lon_c + 180) // 6) + 1
    return 32600 + zone if lat_c >= 0 else 32700 + zone


def transformer_for(epsg: int):
    """Función `(lon, lat) -> (x, y)` en metros para un EPSG dado."""
    return Transformer.from_crs("EPSG:4326", f"EPSG:{epsg}", always_xy=True).transform


def local_transformer(lats, lons):
    """Proyección métrica de la zona en la que caen los datos.

    Devuelve `(transform, epsg)`. A escala de ciudad la distorsión de UTM es
    irrelevante y el plano es métrico de verdad. El `epsg` se devuelve porque
    quien proyecte más adelante otra capa sobre estos mismos datos tiene que
    reusar esta zona, no elegir la suya.
    """
    epsg = local_epsg(lats, lons)
    return transformer_for(epsg), epsg


def build_subgraphs(art: dict, G: nx.MultiDiGraph) -> list[Subgraph]:
    """Traduce el artefacto de hotspots a objetos caracterizables.

    Se proyecta **una sola vez** para toda la ciudad: una proyección por hotspot
    daría planos distintos y las longitudes no serían comparables entre ellos.
    """
    hotspots = art["hotspots"]
    all_nodes = sorted({n for h in hotspots for n in h["nodes"]})
    lats = [float(G.nodes[n]["y"]) for n in all_nodes]
    lons = [float(G.nodes[n]["x"]) for n in all_nodes]
    to_m, epsg = local_transformer(lats, lons)
    xs, ys = to_m(np.asarray(lons), np.asarray(lats))
    coord = {n: (float(x), float(y)) for n, x, y in zip(all_nodes, xs, ys)}

    out: list[Subgraph] = []
    for h in hotspots:
        nodes = list(h["nodes"])
        edges = [(int(u), int(v)) for u, v in h["edges"]]
        out.append(Subgraph(
            id=h["id"],
            month=h["month"],
            nodes=nodes,
            edges=edges,
            xy=np.array([coord[n] for n in nodes], dtype=np.float64),
            edge_class=[_edge_class(G, u, v) for u, v in edges],
            epsg=epsg,
        ))
    return out


def _edge_class(G: nx.MultiDiGraph, u: int, v: int) -> str:
    """Clase vial de la arista `u—v`, mirando en los dos sentidos."""
    for a, b in ((u, v), (v, u)):
        if G.has_edge(a, b):
            data = min(G[a][b].values(), key=lambda d: d.get("length", 0.0))
            return str(data.get("road_class", "street"))
    return "street"


def descriptor(sg: Subgraph) -> np.ndarray:
    """Las 12 dimensiones de un subgrafo, sin normalizar."""
    n, m = sg.n, len(sg.edges)
    deg = sg.degrees()

    # `log1p` y no `log`: un hotspot de un solo nodo tiene 0 aristas, y log(0)
    # es −inf. log1p es monótona, vale 0 en 0 y para |V| ≫ 1 es log salvo un
    # término despreciable, así que conserva el sentido de la dimensión.
    conn = [
        np.log1p(n),
        np.log1p(m),
        (2.0 * m / (n * (n - 1))) if n > 1 else 0.0,
        (2.0 * m / n) if n else 0.0,
        float(np.mean(deg == 1)) if n else 0.0,
        float(np.mean(deg == 2)) if n else 0.0,
        float(np.mean(deg == 3)) if n else 0.0,
        float(np.mean(deg >= 4)) if n else 0.0,
    ]

    lengths = sg.edge_lengths()
    mean_len = float(lengths.mean()) if lengths.size else 0.0
    cv = float(lengths.std() / mean_len) if lengths.size and mean_len > 0 else 0.0

    centroid = sg.xy.mean(axis=0) if n else np.zeros(2)
    d = sg.xy - centroid
    rg = float(np.sqrt((d ** 2).sum(axis=1).mean())) if n else 0.0

    geom = [np.log1p(mean_len), cv, np.log1p(rg), _elongation(d)]
    return np.array(conn + geom, dtype=np.float64)


def _elongation(centered: np.ndarray) -> float:
    """`√(λ₂/λ₁)` sobre la covarianza de las coordenadas centradas.

    ≈0 es una huella lineal (una avenida); ≈1 es isótropa o acodada (un damero).
    Con menos de dos nodos la forma no está definida: se devuelve 0, que es el
    mismo valor que da una recta, el caso degenerado natural.
    """
    if centered.shape[0] < 2:
        return 0.0
    cov = np.cov(centered.T)
    if not np.all(np.isfinite(cov)):
        return 0.0
    lam = np.linalg.eigvalsh(cov)          # ascendente
    l1, l2 = float(lam[-1]), float(max(lam[0], 0.0))
    return float(np.sqrt(l2 / l1)) if l1 > 0 else 0.0


def structural_matrix(subgraphs: list[Subgraph]) -> np.ndarray:
    """Matriz `(n_hotspots, 12)` normalizada min–max por columna (§5.2).

    La normalización es **sobre todo el corpus**, no por hotspot: si cada fila se
    normalizase sola, todas quedarían iguales y el descriptor no distinguiría
    nada.
    """
    X = np.vstack([descriptor(sg) for sg in subgraphs])
    lo = X.min(axis=0)
    hi = X.max(axis=0)
    span = hi - lo
    # Una columna constante no aporta información y su normalización es 0/0.
    flat = span <= 0
    if flat.any():
        logger.debug("Dimensiones constantes en el corpus: %s",
                     [STRUCTURAL_DIMS[i] for i in np.nonzero(flat)[0]])
    span = np.where(flat, 1.0, span)
    out = (X - lo) / span
    out[:, flat] = 0.0
    return out
