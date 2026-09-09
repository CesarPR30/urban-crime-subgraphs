"""Los tres estudios de caso de la nota de tesis.

1. **¿Persistencia implica estabilidad espacial?** El espacio
   `Frequency x Spatial Stability` y sus cuatro cubos. Vive en
   `trajectories.py`; aquí solo se resume.
2. **¿Los hotspots persistentes tienen una topología distinta?** Se parte el
   conjunto en persistentes y episódicos y se contrasta cada descriptor.
3. **Matching.** Pares con `Topology(A) ≈ Topology(B)` pero
   `Crimes(A) >> Crimes(B)`: misma forma de calle, muy distinto crimen. La
   pregunta es qué los separa, y el candidato son los POIs.

## Una advertencia sobre el contraste múltiple

El estudio 2 compara ~20 descriptores entre dos grupos. Con α = 0.05 y veinte
pruebas se espera **una significativa por puro azar**, así que se reporta el
valor p de Benjamini-Hochberg junto al crudo y se ordena por tamaño de efecto,
no por significancia. Un descriptor con p = 0.001 y δ = 0.04 no dice nada
interesante por mucho que pase el umbral: con miles de subgrafos, casi cualquier
diferencia es «significativa».
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)

#: Descriptores extra que la nota pide y que no están en las 12 dimensiones de
#: `structural.py`. Se calculan **solo** para el contraste: meterlos en el
#: descriptor cambiaría el embedding y la similitud ya publicada (§5.1).
EXTRA_DIMS: tuple[str, ...] = (
    "betweenness_mean",
    "closeness_mean",
    "intersection_density_km",
)


def cliffs_delta(a: np.ndarray, b: np.ndarray) -> float:
    """Delta de Cliff: `P(a > b) − P(a < b)`, en `[-1, 1]`.

    Se usa esta y no la d de Cohen porque casi ningún descriptor es normal
    —`log_n_nodes`, las fracciones de grado y la densidad están acotadas o muy
    sesgadas— y la d supone normalidad y varianzas parecidas. Cliff no supone
    nada sobre la forma, que es lo que hace falta aquí.

    Convenio: |δ| < 0.147 despreciable, < 0.33 pequeño, < 0.474 mediano.
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.size == 0 or b.size == 0:
        return 0.0
    # Por orden en vez de comparar todos contra todos: |a|x|b| sería 10^7 pares.
    order = np.argsort(b)
    bs = b[order]
    gt = np.searchsorted(bs, a, side="left").sum()
    ge = np.searchsorted(bs, a, side="right").sum()
    lt = a.size * b.size - ge
    return float((gt - lt) / (a.size * b.size))


def benjamini_hochberg(pvals: list[float]) -> list[float]:
    """Valores p ajustados por tasa de falso descubrimiento.

    Bonferroni sería demasiado conservador con veinte descriptores
    correlacionados entre sí —`log_n_nodes` y `log_n_edges` miden casi lo
    mismo—; BH controla la proporción esperada de falsos positivos entre los
    rechazos, que es la garantía que interesa cuando se exploran descriptores.
    """
    p = np.asarray(pvals, dtype=np.float64)
    n = p.size
    if n == 0:
        return []
    order = np.argsort(p)
    ranked = p[order] * n / (np.arange(n) + 1)
    # Monótono desde el final: un p ajustado no puede bajar al subir el crudo.
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(n, dtype=np.float64)
    out[order] = np.minimum(ranked, 1.0)
    return [float(v) for v in out]


def extra_descriptors(subgraphs, *, betweenness_samples: int = 50) -> np.ndarray:
    """`(n, 3)` con las tres métricas de la nota que faltaban.

    La intermediación se **aproxima** por muestreo de fuentes en los subgrafos
    grandes: Brandes exacto es O(nm) por subgrafo y con 4 160 subgrafos de hasta
    635 nodos son miles de millones de operaciones en Python. Con 50 fuentes el
    orden entre subgrafos se conserva, que es lo único que el contraste usa.
    """
    import networkx as nx

    out = np.zeros((len(subgraphs), len(EXTRA_DIMS)), dtype=np.float64)
    for i, sg in enumerate(subgraphs):
        G = sg.graph()
        n = G.number_of_nodes()
        if n < 2:
            continue
        k = None if n <= betweenness_samples else betweenness_samples
        bet = nx.betweenness_centrality(G, k=k, normalized=True, seed=0)
        out[i, 0] = float(np.mean(list(bet.values())))
        # La cercanía de networkx ya normaliza por componente, así que un
        # subgrafo desconectado no la infla artificialmente.
        clo = nx.closeness_centrality(G)
        out[i, 1] = float(np.mean(list(clo.values())))

        # Intersecciones por kilómetro de calle: nodos de grado >= 3 sobre la
        # longitud total. Es la densidad de la nota, en unidades de red y no de
        # área, porque la unidad de análisis es la calle.
        deg3 = sum(1 for _, d in G.degree() if d >= 3)
        length_km = sum(
            float(np.hypot(*(sg.xy[a] - sg.xy[b])))
            for a, b in G.edges()
        ) / 1000.0
        out[i, 2] = deg3 / length_km if length_km > 0 else 0.0
    return out


@dataclass
class Contrast:
    """Una fila del estudio 2: un descriptor, dos grupos."""

    name: str
    mean_persistent: float
    mean_episodic: float
    delta: float
    u: float | None
    p: float | None
    p_adj: float | None

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "mean_persistent": round(self.mean_persistent, 4),
            "mean_episodic": round(self.mean_episodic, 4),
            "delta": round(self.delta, 4),
            "u": self.u,
            "p": None if self.p is None else round(self.p, 6),
            "p_adj": None if self.p_adj is None else round(self.p_adj, 6),
        }


def contrast_groups(X: np.ndarray, names, mask_a: np.ndarray,
                    mask_b: np.ndarray) -> list[Contrast]:
    """Contrasta cada columna de `X` entre los dos grupos.

    `mask_a` son los persistentes y `mask_b` los episódicos. Se ordena por
    tamaño de efecto y no por valor p: con miles de subgrafos casi cualquier
    diferencia sale significativa, y lo que decide si es interesante es cuánto
    se separan las distribuciones.
    """
    try:
        from scipy.stats import mannwhitneyu
    except ImportError:                       # pragma: no cover
        mannwhitneyu = None
        logger.warning("scipy no disponible: el contraste va sin valores p")

    rows: list[Contrast] = []
    for j, name in enumerate(names):
        a, b = X[mask_a, j], X[mask_b, j]
        u = pv = None
        if mannwhitneyu is not None and a.size and b.size:
            try:
                res = mannwhitneyu(a, b, alternative="two-sided")
                u, pv = float(res.statistic), float(res.pvalue)
            except ValueError:
                pass
        rows.append(Contrast(
            name=name,
            mean_persistent=float(a.mean()) if a.size else 0.0,
            mean_episodic=float(b.mean()) if b.size else 0.0,
            delta=cliffs_delta(a, b), u=u, p=pv, p_adj=None,
        ))

    known = [i for i, r in enumerate(rows) if r.p is not None]
    if known:
        adj = benjamini_hochberg([rows[i].p for i in known])
        for i, v in zip(known, adj):
            rows[i].p_adj = v
    rows.sort(key=lambda r: -abs(r.delta))
    return rows


def matched_pairs(X: np.ndarray, crimes: np.ndarray, ids: list[str],
                  months: list[str], node_sets: list[set] | None = None,
                  *, n_pairs: int = 25, topo_quantile: float = 0.02,
                  min_ratio: float = 3.0, max_overlap: float = 0.05,
                  max_per_member: int = 2) -> list[dict]:
    """Estudio 3: pares con topología casi igual y crimen muy distinto.

    Se buscan pares `(A, B)` cuya distancia en el descriptor estructural
    normalizado esté en el `topo_quantile` inferior —es decir, prácticamente la
    misma forma de calle— pero con `crimes(A) >= min_ratio * crimes(B)`.

    **¿Los dos del par tienen que ser de meses distintos?** De cualquiera: lo
    que se controla es la forma, no el tiempo. Se excluyen en cambio los pares
    que **solapan en el espacio** por encima de `max_overlap`: dos recortes de
    la misma esquina no son un contraste, son el mismo sitio medido dos veces, y
    sin ese filtro la lista se llena de un hotspot grande contra sus propias
    versiones de otros meses.

    También se limita a `max_per_member` las veces que un mismo subgrafo puede
    aparecer. Sin el tope, el subgrafo más extremo de la ciudad copa la lista
    entera y los 25 pares son 25 vistas del mismo caso.
    """
    n = X.shape[0]
    if n < 2:
        return []

    # Muestreo: 4 160^2 son 17 millones de pares y no hacen falta todos para
    # sacar 25 ejemplos. Se toma un subconjunto al azar y se busca en él.
    rng = np.random.default_rng(0)
    cap = 1500
    idx = rng.choice(n, size=min(n, cap), replace=False)
    Xs, cs = X[idx], crimes[idx]

    d = np.linalg.norm(Xs[:, None, :] - Xs[None, :, :], axis=2)
    iu = np.triu_indices(len(idx), k=1)
    dist = d[iu]
    if dist.size == 0:
        return []
    thr = float(np.quantile(dist, topo_quantile))

    cand: list[dict] = []
    for a, b in zip(*iu):
        if d[a, b] > thr:
            continue
        ca, cb = float(cs[a]), float(cs[b])
        hi, lo = (a, b) if ca >= cb else (b, a)
        chi, clo = max(ca, cb), min(ca, cb)
        if clo <= 0 or chi < min_ratio * clo:
            continue
        ov = 0.0
        if node_sets is not None:
            A, B = node_sets[idx[hi]], node_sets[idx[lo]]
            if A and B:
                inter = len(A & B)
                ov = inter / (len(A) + len(B) - inter)
            if ov > max_overlap:
                continue
        cand.append({
            "high": ids[idx[hi]], "low": ids[idx[lo]],
            "month_high": months[idx[hi]], "month_low": months[idx[lo]],
            "crimes_high": int(chi), "crimes_low": int(clo),
            "ratio": round(chi / clo, 2),
            "topo_distance": round(float(d[a, b]), 4),
            "overlap": round(ov, 4),
        })

    cand.sort(key=lambda r: (-r["ratio"], r["topo_distance"]))
    seen: dict[str, int] = {}
    out: list[dict] = []
    for r in cand:
        if any(seen.get(r[k], 0) >= max_per_member for k in ("high", "low")):
            continue
        for k in ("high", "low"):
            seen[r[k]] = seen.get(r[k], 0) + 1
        out.append(r)
        if len(out) >= n_pairs:
            break
    return out
