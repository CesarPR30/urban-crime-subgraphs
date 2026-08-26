"""Campo de densidad escalar sobre la red vial (PROJECT_SPEC §4.1).

Para cada mes, un kernel gaussiano difundido **sobre el grafo**:

.. math::
    f(v) = \\sum_{s \\in V} c(s)\\,
           \\exp\\!\\left(-\\frac{d_G(s,v)^2}{2\\sigma^2}\\right)
    \\qquad \\text{para } d_G(s,v) \\le r

============  ==========================================================
Símbolo       Significado
============  ==========================================================
``c(s)``      crímenes asignados al nodo ``s`` ese mes
``d_G(s,v)``  distancia geodésica **a lo largo de las aristas**
              (Dijkstra acotado), **no euclidiana**
``σ``         ancho de banda, 120 m por defecto (≈ una cuadra corta)
``r``         truncamiento, 3σ (más allá la gaussiana aporta <1 %)
============  ==========================================================

``d_G`` es lo que hace que la difusión respete la red: no atraviesa manzanas ni
cruza el río.

## Nota de implementación

El kernel emitido por un nodo fuente **no depende del mes**: solo depende de la
topología. Lo que cambia entre meses es el peso ``c(s)``. Por eso los vecindarios
se calculan una sola vez para todos los nodos que alguna vez tienen un crimen, y
los 24 campos mensuales son acumulaciones ponderadas sobre esa estructura ya
construida. Sin esta caché habría que repetir ~24× el mismo Dijkstra.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from heapq import heappop, heappush

import networkx as nx
import numpy as np

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CSRGraph:
    """Grafo en formato CSR con nodos renumerados a ``0..N-1``.

    networkx es cómodo pero su Dijkstra genérico es demasiado lento para
    decenas de miles de fuentes. Sobre arrays contiguos el mismo algoritmo va
    un orden de magnitud más rápido.
    """

    indptr: np.ndarray      # (N+1,) offsets de cada nodo en `indices`
    indices: np.ndarray     # (2E,) vecinos
    weights: np.ndarray     # (2E,) longitud de la arista en metros
    ids: np.ndarray         # (N,) id OSM de cada índice interno
    index: dict[int, int]   # id OSM -> índice interno

    @property
    def n(self) -> int:
        return self.ids.size


def to_csr(G: nx.Graph) -> CSRGraph:
    """Convierte un grafo no dirigido con `length` en aristas a CSR."""
    ids = np.fromiter(G.nodes(), dtype=np.int64, count=G.number_of_nodes())
    index = {int(nid): i for i, nid in enumerate(ids)}

    deg = np.zeros(ids.size + 1, dtype=np.int64)
    for u, v in G.edges():
        deg[index[u] + 1] += 1
        deg[index[v] + 1] += 1
    indptr = np.cumsum(deg)

    fill = indptr[:-1].copy()
    m = indptr[-1]
    indices = np.empty(m, dtype=np.int64)
    weights = np.empty(m, dtype=np.float64)
    for u, v, data in G.edges(data=True):
        iu, iv = index[u], index[v]
        w = float(data.get("length", 0.0))
        indices[fill[iu]] = iv; weights[fill[iu]] = w; fill[iu] += 1
        indices[fill[iv]] = iu; weights[fill[iv]] = w; fill[iv] += 1

    return CSRGraph(indptr=indptr, indices=indices, weights=weights, ids=ids, index=index)


def bounded_dijkstra(
    g: CSRGraph, src: int, cutoff: float
) -> tuple[np.ndarray, np.ndarray]:
    """Distancias geodésicas desde `src` hasta `cutoff`, por Dijkstra acotado.

    Returns:
        `(nodos, distancias)` de todos los nodos alcanzables dentro del radio.
    """
    dist: dict[int, float] = {}
    heap: list[tuple[float, int]] = [(0.0, src)]
    indptr, indices, weights = g.indptr, g.indices, g.weights
    while heap:
        d, u = heappop(heap)
        if u in dist:
            continue
        dist[u] = d
        for k in range(indptr[u], indptr[u + 1]):
            v = int(indices[k])
            if v in dist:
                continue
            nd = d + weights[k]
            if nd <= cutoff:
                heappush(heap, (nd, v))
    nodes = np.fromiter(dist.keys(), dtype=np.int64, count=len(dist))
    dists = np.fromiter(dist.values(), dtype=np.float64, count=len(dist))
    return nodes, dists


class KernelCache:
    """Vecindarios gaussianos precalculados, uno por nodo fuente.

    Para cada fuente `s` guarda los nodos dentro de `r` y su peso
    ``exp(−d²/2σ²)``. Independiente del mes: se construye una vez.
    """

    def __init__(self, g: CSRGraph, sigma_m: float, radius_m: float) -> None:
        self.g = g
        self.sigma = sigma_m
        self.radius = radius_m
        self._cache: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        self.settled = 0

    def get(self, src: int) -> tuple[np.ndarray, np.ndarray]:
        hit = self._cache.get(src)
        if hit is not None:
            return hit
        nodes, dists = bounded_dijkstra(self.g, src, self.radius)
        w = np.exp(-(dists ** 2) / (2.0 * self.sigma ** 2))
        self._cache[src] = (nodes, w)
        self.settled += nodes.size
        return nodes, w

    def build(self, sources: np.ndarray) -> "KernelCache":
        t0 = time.perf_counter()
        for i, s in enumerate(sources, 1):
            self.get(int(s))
            if i % 2000 == 0:
                logger.info("  kernels %s / %s", f"{i:,}", f"{len(sources):,}")
        logger.info(
            "Kernels: %s fuentes, %s nodos alcanzados (%.1f por fuente) en %.1f s",
            f"{len(sources):,}", f"{self.settled:,}",
            self.settled / max(1, len(sources)), time.perf_counter() - t0,
        )
        return self


def density_field(
    kernels: KernelCache, src_idx: np.ndarray, src_count: np.ndarray, n: int
) -> np.ndarray:
    """Evalúa `f` en todos los nodos para un mes.

    Args:
        src_idx: índices internos de los nodos con `c(s) > 0`.
        src_count: `c(s)` para cada uno.
        n: número total de nodos.
    """
    f = np.zeros(n, dtype=np.float64)
    for s, c in zip(src_idx, src_count):
        nodes, w = kernels.get(int(s))
        f[nodes] += c * w
    return f
