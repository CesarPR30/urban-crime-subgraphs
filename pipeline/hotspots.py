"""Segmentación del campo de densidad en hotspots (PROJECT_SPEC §4.2 – §4.3).

## Barrido por join tree (§4.2)

Barrido descendente sobre ``f``, equivalente al join tree de los conjuntos de
super-nivel. Se procesan los nodos en orden decreciente de ``f``, con union-find:

* **sin vecino procesado** → máximo local, abre una componente
* **exactamente un vecino** → punto regular, extiende esa componente
* **dos o más** → silla de unión; sobrevive la rama de pico más alto, las otras
  mueren ahí

Por construcción cada región es un **subgrafo conexo**: una componente solo
crece por nodos adyacentes a ella.

## Simplificación por persistencia (§4.3)

.. math::
    \\pi(m) = f(m) - f(s_m) \\qquad
    \\tau = \\alpha \\cdot \\max_v f(v)

con :math:`s_m` la silla donde el máximo :math:`m` se fusionó con una rama más
alta. Todo máximo con :math:`\\pi(m) < \\tau` se absorbe en esa rama; se sigue la
cadena de fusión hasta llegar a un máximo persistente.

Los nodos por debajo de un piso :math:`f_{min}` quedan fuera del barrido, para
que las regiones no se derramen por la cola del kernel.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np

from .density import CSRGraph

logger = logging.getLogger(__name__)


@dataclass
class Hotspot:
    """Una región conexa del campo de densidad, ya simplificada."""

    id: str
    month: str
    nodes: list[int]                      # ids OSM
    edges: list[tuple[int, int]]          # ids OSM
    crimes: int
    by_category: dict[str, int] = field(default_factory=dict)
    seed_node: int = -1                   # nodo con más crímenes (§4.3)
    peak_node: int = -1                   # máximo del campo que define la región
    peak_f: float = 0.0
    persistence: float = 0.0

    @property
    def n_nodes(self) -> int:
        return len(self.nodes)

    @property
    def density(self) -> float:
        """Crímenes por nodo dentro de la región."""
        return self.crimes / len(self.nodes) if self.nodes else 0.0

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "month": self.month,
            "nodes": self.nodes,
            "edges": [list(e) for e in self.edges],
            "crimes": self.crimes,
            "by_category": self.by_category,
            "seed_node": self.seed_node,
            "peak_node": self.peak_node,
            "peak_f": round(self.peak_f, 6),
            "persistence": round(self.persistence, 6),
            "n_nodes": self.n_nodes,
            "density": round(self.density, 4),
        }


def segment(
    f: np.ndarray, g: CSRGraph, alpha: float, f_min_ratio: float
) -> tuple[dict[int, list[int]], dict[int, float], dict[int, float]]:
    """Ejecuta el barrido y la simplificación por persistencia.

    Cada nodo se asigna a la región de **uno de sus vecinos ya procesados**, la
    de pico más alto. Esa es la invariante que garantiza que las regiones sean
    conexas, y no es equivalente a asignarlo a la raíz de un union-find: la raíz
    fusiona todas las ramas que se han tocado, incluidas las persistentes, así
    que puede devolver un pico al que el nodo no está físicamente conectado.

    La absorción se decide en el momento de la fusión, que es cuando ya se
    conoce :math:`\\pi(m) = f(m) - f(s_m)`: la rama muere en esa silla, no
    después. Un máximo absorbido queda apuntando a su absorbente en `alias`, y
    la resolución final sigue la cadena.

    Returns:
        `(regiones, persistencia, pico)` donde `regiones` mapea el índice
        interno del máximo persistente a la lista de índices de sus nodos.
    """
    n = f.size
    fmax = float(f.max()) if n else 0.0
    if fmax <= 0.0:
        return {}, {}, {}

    tau = alpha * fmax
    f_min = f_min_ratio * fmax

    order = np.argsort(-f, kind="stable")
    order = order[f[order] >= f_min]

    region = np.full(n, -1, dtype=np.int64)   # pico al que se asigna cada nodo
    processed = np.zeros(n, dtype=bool)
    alias: dict[int, int] = {}                # pico absorbido -> pico absorbente
    persistence: dict[int, float] = {}
    n_maxima = 0

    def top(p: int) -> int:
        """Pico vigente de `p`, siguiendo la cadena de absorción (con compresión)."""
        root = p
        while root in alias:
            root = alias[root]
        while p in alias and alias[p] != root:
            alias[p], p = root, alias[p]
        return root

    indptr, indices = g.indptr, g.indices

    for v in order:
        v = int(v)
        touched = set()
        for k in range(indptr[v], indptr[v + 1]):
            u = int(indices[k])
            if processed[u]:
                touched.add(top(int(region[u])))

        if not touched:
            # Máximo local: abre una región. Mientras no se fusione, su
            # persistencia es su altura completa.
            region[v] = v
            persistence[v] = float(f[v])
            n_maxima += 1
        else:
            # Silla de unión: sobrevive la rama de pico más alto.
            survivor = max(touched, key=lambda p: (f[p], -p))
            region[v] = survivor
            for p in touched:
                if p == survivor:
                    continue
                pi = float(f[p]) - float(f[v])
                persistence[p] = pi
                if pi < tau:
                    alias[p] = survivor      # se absorbe en la rama superviviente
                # Si π ≥ τ el máximo es persistente: conserva su región.
        processed[v] = True

    regions: dict[int, list[int]] = defaultdict(list)
    for v in order:
        v = int(v)
        regions[top(int(region[v]))].append(v)

    survivors = {p: persistence[p] for p in regions}
    peaks = {p: float(f[p]) for p in regions}
    logger.debug(
        "Barrido: %d máximos -> %d regiones (τ=%.3f, f_min=%.3f)",
        n_maxima, len(regions), tau, f_min,
    )
    return dict(regions), survivors, peaks


def induced_edges(g: CSRGraph, idx: list[int]) -> list[tuple[int, int]]:
    """Aristas del subgrafo inducido por `idx`, en ids OSM."""
    inside = set(idx)
    out: list[tuple[int, int]] = []
    for v in idx:
        for k in range(g.indptr[v], g.indptr[v + 1]):
            u = int(g.indices[k])
            if u in inside and u > v:
                out.append((int(g.ids[v]), int(g.ids[u])))
    return out


def extract_month(
    month: str,
    f: np.ndarray,
    g: CSRGraph,
    counts: np.ndarray,
    cat_counts: dict[str, np.ndarray],
    *,
    alpha: float,
    f_min_ratio: float,
    top_k: int,
) -> list[Hotspot]:
    """Extrae los `top_k` hotspots de un mes.

    Args:
        f: campo de densidad de ese mes, por índice interno.
        counts: `c(v)` de ese mes, por índice interno.
        cat_counts: `c(v)` desglosado por categoría.

    Las regiones se puntúan por **crimen crudo capturado** (`Σ c(v)`), no por
    densidad: el objetivo es cubrir crimen real, y puntuar por `f` premiaría a
    las regiones que solo son grandes.
    """
    regions, persistence, peaks = segment(f, g, alpha, f_min_ratio)
    if not regions:
        return []

    scored = []
    for peak, idx in regions.items():
        score = int(counts[idx].sum())
        if score > 0:
            scored.append((score, peak, idx))
    scored.sort(key=lambda t: (-t[0], -peaks[t[1]]))

    hotspots: list[Hotspot] = []
    for rank, (score, peak, idx) in enumerate(scored[:top_k], 1):
        arr = np.asarray(idx, dtype=np.int64)
        local = counts[arr]
        seed = int(arr[int(np.lexsort((-f[arr], -local))[0])])
        hotspots.append(
            Hotspot(
                id=f"{month}_h{rank:02d}",
                month=month,
                nodes=[int(x) for x in g.ids[arr]],
                edges=induced_edges(g, [int(i) for i in arr]),
                crimes=score,
                by_category={
                    cat: int(vec[arr].sum()) for cat, vec in cat_counts.items()
                    if int(vec[arr].sum()) > 0
                },
                seed_node=int(g.ids[seed]),
                peak_node=int(g.ids[peak]),
                peak_f=peaks[peak],
                persistence=persistence[peak],
            )
        )
    return hotspots
