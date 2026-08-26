"""Línea base: region growing voraz por BFS (PROJECT_SPEC §7.2).

Sin línea base no hay forma de saber si el extractor topológico es mejor que lo
obvio — que es exactamente uno de los errores que el proyecto no quiere repetir
(§11.7).

El método es deliberadamente simple: se toman los nodos más calientes en crudo y
se crece alrededor de cada uno por anchura hasta agotar un presupuesto de nodos.
No hay campo de densidad, ni join tree, ni persistencia; solo «rodea el pico».

## Por qué el presupuesto de nodos importa

Comparar cobertura sin fijar la huella no dice nada: BFS ganaría simplemente
creciendo más. Por eso, con `match_footprint`, cada región BFS recibe como
presupuesto el **tamaño medio de las regiones topológicas de ese mismo mes**. Así
las dos familias de subgrafos ocupan aproximadamente los mismos nodos y la
comparación aísla lo único que cambia: *dónde* se ponen esos nodos.
"""

from __future__ import annotations

import logging
from collections import deque
from heapq import heappop, heappush

import numpy as np

from .density import CSRGraph
from .hotspots import Hotspot, induced_edges

logger = logging.getLogger(__name__)


def grow_bfs(
    g: CSRGraph, seed: int, budget: int, taken: np.ndarray, counts: np.ndarray
) -> list[int]:
    """Anchura pura: crece en todas las direcciones por igual.

    `taken` marca los nodos ya asignados a otra región: las regiones no se
    solapan, igual que las del extractor topológico, para que la huella total
    sea comparable.

    Es la variante *ingenua*. Al no mirar el crimen de los vecinos, se traga
    tantos nodos vacíos como calientes y su densidad se hunde. Se conserva
    porque delimita el suelo de la comparación: cuánto se pierde por no ser
    voraz.
    """
    if taken[seed] or budget <= 0:
        return []
    region = [seed]
    taken[seed] = True
    q = deque([seed])
    indptr, indices = g.indptr, g.indices
    while q and len(region) < budget:
        v = q.popleft()
        for k in range(indptr[v], indptr[v + 1]):
            u = int(indices[k])
            if not taken[u]:
                taken[u] = True
                region.append(u)
                q.append(u)
                if len(region) >= budget:
                    break
    return region


def grow_greedy(
    g: CSRGraph, seed: int, budget: int, taken: np.ndarray, counts: np.ndarray
) -> list[int]:
    """Region growing **voraz**: en cada paso absorbe el vecino más caliente.

    Es el `region growing voraz por BFS` de §7.2. La diferencia con la anchura
    pura es la política de la frontera: en vez de una cola FIFO, un montículo
    ordenado por ``c(v)`` descendente. La región sigue creciendo solo por nodos
    adyacentes —así que sigue siendo conexa— pero elige *hacia dónde*.

    Esa elección es toda la línea base: rodear un pico con los nodos que más
    crimen aportan es exactamente lo que un analista haría a mano, y es contra
    eso contra lo que hay que ganar. El desempate es por índice, para que el
    resultado sea reproducible.
    """
    if taken[seed] or budget <= 0:
        return []
    region = [seed]
    taken[seed] = True
    indptr, indices = g.indptr, g.indices
    heap: list[tuple[float, int]] = []
    seen: set[int] = {seed}

    def push_neighbors(v: int) -> None:
        for k in range(indptr[v], indptr[v + 1]):
            u = int(indices[k])
            if u not in seen and not taken[u]:
                seen.add(u)
                heappush(heap, (-float(counts[u]), u))

    push_neighbors(seed)
    while heap and len(region) < budget:
        _, u = heappop(heap)
        if taken[u]:
            continue
        taken[u] = True
        region.append(u)
        push_neighbors(u)
    return region


GROWTH = {"greedy": grow_greedy, "bfs": grow_bfs}


def extract_month_bfs(
    month: str,
    g: CSRGraph,
    counts: np.ndarray,
    cat_counts: dict[str, np.ndarray],
    *,
    top_k: int,
    budget: int,
    growth: str = "greedy",
) -> list[Hotspot]:
    """Extrae `top_k` regiones desde los nodos más calientes del mes.

    Args:
        budget: nodos por región. Con `match_footprint` es la media de las
            regiones topológicas de ese mes.
        growth: `greedy` (§7.2) o `bfs` (anchura pura, variante ingenua).
    """
    if budget <= 0 or counts.sum() <= 0:
        return []
    grow = GROWTH[growth]

    # Nodos más calientes en crudo. Desempate por índice para reproducibilidad.
    order = np.lexsort((np.arange(g.n), -counts))
    taken = np.zeros(g.n, dtype=bool)

    regions: list[list[int]] = []
    for seed in order:
        if len(regions) >= top_k:
            break
        seed = int(seed)
        if counts[seed] <= 0 or taken[seed]:
            continue
        region = grow(g, seed, budget, taken, counts)
        if region:
            regions.append(region)

    hotspots: list[Hotspot] = []
    scored = sorted(
        ((int(counts[np.asarray(r)].sum()), r) for r in regions),
        key=lambda t: -t[0],
    )
    for rank, (score, region) in enumerate(scored, 1):
        arr = np.asarray(region, dtype=np.int64)
        local = counts[arr]
        seed_i = int(arr[int(np.lexsort((np.arange(arr.size), -local))[0])])
        hotspots.append(Hotspot(
            id=f"{month}_bfs{rank:02d}",
            month=month,
            nodes=[int(x) for x in g.ids[arr]],
            edges=induced_edges(g, [int(i) for i in arr]),
            crimes=score,
            by_category={
                cat: int(vec[arr].sum()) for cat, vec in cat_counts.items()
                if int(vec[arr].sum()) > 0
            },
            seed_node=int(g.ids[seed_i]),
            peak_node=int(g.ids[seed_i]),
            peak_f=float(counts[seed_i]),
            persistence=0.0,
        ))
    return hotspots
