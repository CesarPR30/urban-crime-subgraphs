"""Fusión, similitud coseno, top-K y proyección 2D (PROJECT_SPEC §5.5).

El procedimiento es el de la spec, en su orden:

1. Estandarizar cada bloque por separado (estructural, embedding, jerarquía).
2. Ponderar cada bloque con los pesos del config.
3. Concatenar.
4. Similitud coseno `sim(hᵢ,hⱼ) = (xᵢ·xⱼ) / (‖xᵢ‖‖xⱼ‖)`.
5. Precalcular los top-K más similares de cada hotspot.
6. Proyectar a 2D con UMAP bajo métrica coseno; agrupar con HDBSCAN.

El paso 1 no es cosmético. El bloque estructural son 12 dims en [0,1], el
embedding son 128 dims con otra escala y la jerarquía son 6 fracciones. Sin
estandarizar, el bloque con más dimensiones y más varianza decide la similitud
él solo, y los pesos del config no significan nada.

## Similitud y mismo lugar

El corpus son los 480 hotspots de la ventana: 20 por mes, 24 meses. El mismo
cruce aparece mes tras mes, y sus copias son casi idénticas en forma, así que
tienden a copar los primeros puestos. Eso es correcto —lo son— pero inútil para
la pregunta del proyecto, que compara zonas distintas. Por eso se calculan **dos
listas**: la del spec y otra que excluye los solapes espaciales, medidos por
Jaccard de conjuntos de nodos.
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass, field

import numpy as np

logger = logging.getLogger(__name__)

#: Por debajo de este Jaccard dos huellas se consideran sitios distintos.
#: 0 es el criterio estricto: ni un nodo en común.
SAME_PLACE_JACCARD = 0.0


@dataclass
class Blocks:
    """Los tres bloques que entran en la similitud. Ninguno ve crimen (§5.1)."""

    structural: np.ndarray
    embedding: np.ndarray
    hierarchy: np.ndarray

    def names(self) -> list[str]:
        return ["structural", "embedding", "hierarchy"]

    def shapes(self) -> dict[str, tuple[int, int]]:
        return {n: getattr(self, n).shape for n in self.names()}


@dataclass
class SimilarityResult:
    ids: list[str]
    fused: np.ndarray                     # (n, d) vector fusionado y ponderado
    sim: np.ndarray                       # (n, n) matriz coseno
    top: dict[str, list[dict]]            # top-K según §5.5
    top_distinct: dict[str, list[dict]]   # top-K excluyendo el mismo lugar
    coords: np.ndarray                    # (n, 2) proyección
    labels: np.ndarray                    # (n,) etiqueta de clúster, −1 = ruido
    meta: dict = field(default_factory=dict)


def standardize(X: np.ndarray) -> np.ndarray:
    """z-score por columna. Las columnas constantes salen a cero, no a nan."""
    mu = X.mean(axis=0)
    sd = X.std(axis=0)
    flat = sd <= 1e-12
    sd = np.where(flat, 1.0, sd)
    out = (X - mu) / sd
    out[:, flat] = 0.0
    return out


def fuse(blocks: Blocks, weights) -> np.ndarray:
    """Estandariza, pondera y concatena los tres bloques (§5.5, pasos 1-3).

    El peso se reparte entre las dimensiones del bloque (`w/√d`) en vez de
    aplicarse a cada una. Sin eso, un bloque de 128 dims con peso 1.0 aporta
    ~128 unidades de norma² y uno de 12 aporta ~12: el "peso" real sería el
    número de columnas, no el del config.
    """
    parts = []
    for name in blocks.names():
        X = standardize(getattr(blocks, name))
        w = float(getattr(weights, name))
        if X.shape[1] == 0:
            continue
        parts.append(X * (w / np.sqrt(X.shape[1])))
    return np.hstack(parts)


def cosine_matrix(X: np.ndarray) -> np.ndarray:
    """Matriz coseno completa, con la diagonal a 1 exacta."""
    norm = np.linalg.norm(X, axis=1, keepdims=True)
    norm[norm == 0] = 1.0
    U = X / norm
    S = U @ U.T
    np.fill_diagonal(S, 1.0)
    return np.clip(S, -1.0, 1.0)


def jaccard_matrix(node_sets: list[set[int]]) -> np.ndarray:
    """Solape espacial entre huellas. `1` = mismos nodos, `0` = disjuntas."""
    n = len(node_sets)
    J = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        a = node_sets[i]
        for j in range(i + 1, n):
            b = node_sets[j]
            inter = len(a & b)
            if not inter:
                continue
            J[i, j] = J[j, i] = inter / (len(a) + len(b) - inter)
    np.fill_diagonal(J, 1.0)
    return J


def top_k(sim: np.ndarray, ids: list[str], months: list[str], k: int,
          jac: np.ndarray, *, exclude_same_place: bool) -> dict[str, list[dict]]:
    """Los `k` más similares de cada hotspot, ya ordenados."""
    n = len(ids)
    out: dict[str, list[dict]] = {}
    for i in range(n):
        s = sim[i].copy()
        s[i] = -np.inf                       # uno no es su propio similar
        if exclude_same_place:
            s[jac[i] > SAME_PLACE_JACCARD] = -np.inf
        # `argpartition` y luego ordenar solo los k: O(n) en vez de O(n log n)
        # por fila. Con 480 da igual, con 10 000 hotspots no.
        kk = min(k, int(np.isfinite(s).sum()))
        if kk <= 0:
            out[ids[i]] = []
            continue
        cand = np.argpartition(-s, kk - 1)[:kk]
        cand = cand[np.argsort(-s[cand])]
        out[ids[i]] = [
            {"id": ids[j], "month": months[j],
             "score": round(float(sim[i, j]), 5),
             "jaccard": round(float(jac[i, j]), 4)}
            for j in cand
        ]
    return out


def project(X: np.ndarray, cfg) -> tuple[np.ndarray, str]:
    """UMAP bajo métrica coseno (§5.5, paso 6), con PCA de reserva.

    La spec exige que las coordenadas del dashboard se generen siempre, así que
    si UMAP no está disponible se cae a las dos primeras componentes
    principales. Es una proyección peor —lineal, sin preservar vecindades— pero
    existe y es determinista.
    """
    from .features.embedding import _pca

    u = cfg.similarity.umap
    n = X.shape[0]
    if n < 5:
        logger.warning("Corpus de %d hotspots: UMAP no tiene vecindario que "
                       "preservar, se proyecta con PCA.", n)
        return _pca(X, 2), "pca"

    try:
        import umap
    except ImportError:
        logger.warning("umap-learn no está instalado: la proyección 2D usa PCA.")
        return _pca(X, 2), "pca"

    try:
        with warnings.catch_warnings():
            # UMAP avisa de que fijar `random_state` desactiva el paralelismo.
            # Es exactamente lo que queremos: §5.3 pide reproducibilidad.
            warnings.simplefilter("ignore")
            reducer = umap.UMAP(
                n_neighbors=min(u.n_neighbors, n - 1),
                min_dist=u.min_dist,
                metric=u.metric,
                n_components=2,
                random_state=cfg.project.random_state,
            )
            Y = np.asarray(reducer.fit_transform(X), dtype=np.float64)
    except Exception as exc:      # §5.3: las coordenadas se generan siempre
        logger.warning("UMAP falló (%s): la proyección 2D usa PCA.", exc)
        return _pca(X, 2), "pca"
    return Y, "umap"


def cluster(X: np.ndarray, cfg) -> tuple[np.ndarray, str]:
    """HDBSCAN sobre el vector fusionado (§5.5, paso 6).

    Se agrupa sobre el espacio fusionado, no sobre las coordenadas 2D: UMAP
    deforma densidades y agrupar sobre su salida encontraría clústeres que son
    artefactos de la proyección.
    """
    try:
        from sklearn.cluster import HDBSCAN
    except ImportError:
        logger.warning("scikit-learn no está instalado: sin clustering.")
        return np.full(X.shape[0], -1, dtype=np.int64), "none"

    n = X.shape[0]
    # HDBSCAN lanza si `min_cluster_size` supera el número de muestras. Con un
    # corpus pequeño —pocos meses, K bajo, o los tests— eso tumbaría el paso
    # entero por un parámetro que solo tiene sentido a partir de cierto tamaño.
    size = min(cfg.similarity.hdbscan.min_cluster_size, n)
    if n < 3 or size < 2:
        logger.warning("Corpus de %d hotspots: demasiado pequeño para agrupar.", n)
        return np.full(n, -1, dtype=np.int64), "none"

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        labels = HDBSCAN(min_cluster_size=size, metric="euclidean").fit_predict(X)
    return np.asarray(labels, dtype=np.int64), "hdbscan"


def run(subgraphs, blocks: Blocks, cfg) -> SimilarityResult:
    """El pipeline completo de §5.5 sobre un corpus ya caracterizado."""
    ids = [sg.id for sg in subgraphs]
    months = [sg.month for sg in subgraphs]
    node_sets = [set(sg.nodes) for sg in subgraphs]

    fused = fuse(blocks, cfg.similarity.weights)
    sim = cosine_matrix(fused)
    jac = jaccard_matrix(node_sets)
    k = cfg.similarity.top_k_similar

    coords, proj_method = project(fused, cfg)
    labels, clust_method = cluster(fused, cfg)

    n_clusters = int(len({int(l) for l in labels} - {-1}))
    logger.info("Similitud: %d hotspots, vector fusionado de %d dims, "
                "%d clústeres (%d ruido)",
                len(ids), fused.shape[1], n_clusters, int((labels == -1).sum()))

    return SimilarityResult(
        ids=ids,
        fused=fused,
        sim=sim,
        top=top_k(sim, ids, months, k, jac, exclude_same_place=False),
        top_distinct=top_k(sim, ids, months, k, jac, exclude_same_place=True),
        coords=coords,
        labels=labels,
        meta={
            "n": len(ids),
            "blocks": {n: list(s) for n, s in blocks.shapes().items()},
            "fused_dims": int(fused.shape[1]),
            "weights": cfg.similarity.weights.model_dump(),
            "top_k": k,
            "projection": proj_method,
            "clustering": clust_method,
            "n_clusters": n_clusters,
            "n_noise": int((labels == -1).sum()),
            "same_place_jaccard": SAME_PLACE_JACCARD,
            "random_state": cfg.project.random_state,
        },
    )
