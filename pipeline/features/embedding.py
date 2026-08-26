"""Embedding aprendido de subgrafos (PROJECT_SPEC §5.3).

Por defecto **graph2vec**: reetiquetado Weisfeiler–Lehman seguido de Doc2Vec en
su variante PV-DBOW.

    ℓ⁽ᵗ⁺¹⁾(v) = hash( ℓ⁽ᵗ⁾(v) ‖ sort{ ℓ⁽ᵗ⁾(u) : u ∈ N(v) } )

con `ℓ⁽⁰⁾(v) = deg(v)`. El `sort` es lo que da invariancia a la numeración de los
nodos: dos subgrafos isomorfos producen exactamente el mismo multiconjunto de
etiquetas, se numeren como se numeren.

El multiconjunto de etiquetas de todas las iteraciones es el "documento" del
hotspot, y el ajuste es **conjunto sobre todo el corpus a la vez**. Entrenar
hotspot por hotspot no daría un espacio común: los motivos estructurales
recurrentes solo reciben vectores cercanos si compiten en el mismo modelo.

## Por qué PV-DBOW está implementado aquí

`gensim` no publica wheel para Python 3.14 —pip resuelve a la 0.10.1, de 2014,
incompatible con numpy 2— así que el Doc2Vec va escrito en numpy. Son ~60 líneas
de descenso por gradiente con muestreo negativo, el mismo algoritmo que usa
gensim, y a cambio el pipeline deja de depender de una pieza que hoy no existe
para esta versión del intérprete.

Consecuencia buscada: el hash de las etiquetas WL usa `hashlib`, no el `hash()`
de Python, que va salteado por proceso y rompería la reproducibilidad entre
corridas (§5.3 exige resultados reproducibles con el mismo seed).
"""

from __future__ import annotations

import hashlib
import logging
from typing import Protocol

import networkx as nx
import numpy as np

logger = logging.getLogger(__name__)


class Embedder(Protocol):
    """Interfaz única de §5.3: todo método se selecciona por config."""

    name: str

    def fit_transform(self, graphs: list[nx.Graph]) -> np.ndarray:
        ...


# --------------------------------------------------------------------------- #
# Weisfeiler–Lehman
# --------------------------------------------------------------------------- #


def _hash(s: str) -> str:
    return hashlib.blake2s(s.encode("utf-8"), digest_size=8).hexdigest()


def wl_documents(graphs: list[nx.Graph], iterations: int) -> list[list[str]]:
    """Documento WL de cada grafo: sus etiquetas de todas las iteraciones."""
    docs: list[list[str]] = []
    for g in graphs:
        labels = {v: f"d{g.degree(v)}" for v in g.nodes}
        words = list(labels.values())
        for _ in range(iterations):
            nxt = {}
            for v in g.nodes:
                # `sorted` sobre el multiconjunto de vecinos: el orden canónico
                # es justo lo que borra la numeración de los nodos.
                sig = labels[v] + "|" + ",".join(sorted(labels[u] for u in g[v]))
                nxt[v] = _hash(sig)
            labels = nxt
            words.extend(labels.values())
        docs.append(words)
    return docs


# --------------------------------------------------------------------------- #
# Doc2Vec PV-DBOW
# --------------------------------------------------------------------------- #


def _sigmoid(x: np.ndarray) -> np.ndarray:
    # `clip` antes de exp: sin él, un score de −800 desborda a inf en float64 y
    # el gradiente sale nan. El recorte no cambia nada útil, σ(±40) ya es 0 o 1.
    return 1.0 / (1.0 + np.exp(-np.clip(x, -40.0, 40.0)))


class Doc2VecDBOW:
    """PV-DBOW con muestreo negativo, entrenado sobre todo el corpus.

    El vector del documento predice sus propias palabras; no hay vectores de
    contexto de palabra a palabra. Es la variante que usa graph2vec porque las
    "palabras" (etiquetas WL) no tienen orden: un documento es un multiconjunto,
    no una secuencia, y cualquier modelo con ventana estaría inventando
    adyacencias que no existen.
    """

    def __init__(self, dimensions: int = 128, epochs: int = 50, negative: int = 5,
                 alpha: float = 0.025, min_alpha: float = 1e-4,
                 random_state: int = 42) -> None:
        self.dimensions = dimensions
        self.epochs = epochs
        self.negative = negative
        self.alpha = alpha
        self.min_alpha = min_alpha
        self.random_state = random_state

    def fit(self, documents: list[list[str]]) -> np.ndarray:
        rng = np.random.RandomState(self.random_state)
        vocab: dict[str, int] = {}
        freq: list[int] = []
        docs_idx: list[np.ndarray] = []
        for doc in documents:
            ids = []
            for w in doc:
                i = vocab.get(w)
                if i is None:
                    i = vocab[w] = len(freq)
                    freq.append(0)
                freq[i] += 1
                ids.append(i)
            docs_idx.append(np.asarray(ids, dtype=np.int64))

        n_docs, n_vocab, k = len(documents), len(vocab), self.dimensions
        logger.info("Corpus WL: %d documentos, %d etiquetas distintas, "
                    "%d palabras en total", n_docs, n_vocab, sum(freq))

        # Init como word2vec: los vectores de documento aleatorios y pequeños,
        # los de salida a cero (así las primeras actualizaciones no arrastran
        # ruido).
        D = (rng.rand(n_docs, k).astype(np.float64) - 0.5) / k
        W = np.zeros((n_vocab, k), dtype=np.float64)

        # Distribución de ruido ∝ freq^0.75: la de word2vec. Aplana la cola de
        # las etiquetas rarísimas (casi todas las de la última iteración WL son
        # únicas) sin llegar a la uniforme.
        p = np.asarray(freq, dtype=np.float64) ** 0.75
        p /= p.sum()

        order = np.arange(n_docs)
        for ep in range(self.epochs):
            lr = self.alpha - (self.alpha - self.min_alpha) * ep / max(1, self.epochs - 1)
            rng.shuffle(order)
            for i in order:
                wi = docs_idx[i]
                if wi.size == 0:
                    continue
                ni = rng.choice(n_vocab, size=(wi.size, self.negative), p=p)
                d = D[i]

                pos = W[wi]                              # (nw, k)
                gp = (1.0 - _sigmoid(pos @ d)) * lr      # (nw,)
                neg = W[ni]                              # (nw, ng, k)
                gn = -_sigmoid(neg @ d) * lr             # (nw, ng)

                grad_d = gp @ pos + np.einsum("ij,ijk->k", gn, neg)
                np.add.at(W, wi, gp[:, None] * d)
                np.add.at(W, ni.ravel(), gn.reshape(-1, 1) * d)
                D[i] = d + grad_d

        return D


# --------------------------------------------------------------------------- #
# Métodos
# --------------------------------------------------------------------------- #


class Graph2Vec:
    """WL + PV-DBOW. El método por defecto de §5.3."""

    name = "graph2vec"

    def __init__(self, dimensions: int, wl_iterations: int, epochs: int,
                 random_state: int) -> None:
        self.dimensions = dimensions
        self.wl_iterations = wl_iterations
        self.epochs = epochs
        self.random_state = random_state

    def fit_transform(self, graphs: list[nx.Graph]) -> np.ndarray:
        docs = wl_documents(graphs, self.wl_iterations)
        model = Doc2VecDBOW(dimensions=self.dimensions, epochs=self.epochs,
                            random_state=self.random_state)
        return model.fit(docs)


class NetLSD:
    """Firma heat-kernel del laplaciano. El fallback de §5.3.

    `h(t) = Σᵢ exp(−t·λᵢ)` sobre una rejilla logarítmica de escalas: a `t`
    pequeño la firma ve la estructura local, a `t` grande la global. No aprende
    nada, así que no depende de nada y nunca falla —que es precisamente su
    trabajo aquí—.
    """

    name = "netlsd"

    def __init__(self, dimensions: int, timescales: int = 250,
                 t_min: float = -2.0, t_max: float = 2.0) -> None:
        self.dimensions = dimensions
        self.ts = np.logspace(t_min, t_max, timescales)

    def fit_transform(self, graphs: list[nx.Graph]) -> np.ndarray:
        sigs = np.vstack([self._signature(g) for g in graphs])
        return _pca(sigs, self.dimensions)

    def _signature(self, g: nx.Graph) -> np.ndarray:
        n = g.number_of_nodes()
        if n == 0:
            return np.zeros(self.ts.size)
        lam = np.linalg.eigvalsh(
            nx.normalized_laplacian_matrix(g).toarray().astype(np.float64)
        )
        # Normalizar por n hace la firma comparable entre subgrafos de tamaño
        # distinto, que es lo que pide la similitud coseno de §5.5.
        return np.exp(-np.outer(self.ts, lam)).sum(axis=1) / n


def _pca(X: np.ndarray, k: int) -> np.ndarray:
    """PCA por SVD. Determinista salvo el signo, que se fija por convención."""
    Xc = X - X.mean(axis=0)
    k = min(k, min(Xc.shape))
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    # El signo de cada componente es arbitrario en SVD: se ancla al del elemento
    # de mayor módulo para que dos corridas den literalmente el mismo array.
    for j in range(Vt.shape[0]):
        if Vt[j, np.argmax(np.abs(Vt[j]))] < 0:
            Vt[j] *= -1
            U[:, j] *= -1
    return U[:, :k] * S[:k]


#: Métodos que requieren dependencias que este entorno no tiene. §5.3 manda
#: caer a netlsd con un warning claro, nunca fallar.
_NEEDS_OPTIONAL = {
    "gl2vec": "karateclub",
    "feather": "karateclub",
    "gcn": "torch + torch-geometric",
}


def build_embedder(cfg) -> Embedder:
    """Instancia el método de `config.yaml`, con el fallback de §5.3."""
    e = cfg.features.embedding
    seed = cfg.project.random_state

    if e.method == "netlsd":
        return NetLSD(e.dimensions)
    if e.method == "graph2vec":
        return Graph2Vec(e.dimensions, e.wl_iterations, e.epochs, seed)

    dep = _NEEDS_OPTIONAL.get(e.method)
    logger.warning(
        "El método de embedding %r necesita %s, que no está instalado. "
        "Se usa el fallback netlsd + PCA (§5.3). Las coordenadas del dashboard "
        "se generan igual, pero no son las de %r.",
        e.method, dep, e.method,
    )
    return NetLSD(e.dimensions)
