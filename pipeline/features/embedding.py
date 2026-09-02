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


# --------------------------------------------------------------------------- #
# Wasserstein Weisfeiler–Lehman   (Togninalli et al., NeurIPS 2019)
# --------------------------------------------------------------------------- #


def _wl_node_cloud(g: nx.Graph, hier_order: list[str], iterations: int) -> np.ndarray:
    """Nube de vectores continuos por nodo, refinada con `iterations` pasadas WL.

    graph2vec compara *histogramas* de etiquetas WL: dos motivos casi iguales
    caen en cubos distintos y su parecido se pierde en la agregación. WWL guarda
    esta nube y compara dos grafos por el coste de transporte óptimo entre sus
    nubes (Togninalli et al., 2019, §3), que sí ve diferencias finas y admite
    atributos continuos —aquí, la geometría métrica y la clase vial—.
    """
    nodes = list(g.nodes)
    idx = {v: i for i, v in enumerate(nodes)}
    n = len(nodes)
    deg = np.array([g.degree(v) for v in nodes], dtype=np.float64)
    clus = np.array([nx.clustering(g, v) for v in nodes], dtype=np.float64)

    xy = np.array([g.nodes[v].get("xy", (0.0, 0.0)) for v in nodes], dtype=np.float64)
    xy = xy - xy.mean(axis=0) if n else xy
    rg = float(np.sqrt((xy ** 2).sum(axis=1).mean())) or 1.0
    dist_c = np.linalg.norm(xy, axis=1) / rg

    hi = {c: k for k, c in enumerate(hier_order)}
    road = np.zeros((n, len(hier_order)), dtype=np.float64)
    for v in nodes:
        inc = [g[v][u].get("cls", "") for u in g[v]]
        for c in inc:
            if c in hi:
                road[idx[v], hi[c]] += 1.0
        if inc:
            s = road[idx[v]].sum()
            if s:
                road[idx[v]] /= s

    base = np.column_stack([np.log1p(deg), clus, dist_c, road])
    if n:
        A = nx.to_numpy_array(g, nodelist=nodes)
        P = A / np.clip(A.sum(axis=1, keepdims=True), 1.0, None)
        feats, cur = [base], base
        for _ in range(iterations):
            cur = 0.5 * cur + 0.5 * (P @ cur)
            feats.append(cur)
        return np.hstack(feats)
    return base


def _sinkhorn_w1(C: np.ndarray, eps: float, iters: int) -> float:
    """Wasserstein-1 entrópico entre dos nubes con masa uniforme."""
    n, m = C.shape
    K = np.exp(-C / eps)
    u = np.ones(n)
    inv_a, b = 1.0 / n, np.full(m, 1.0 / m)
    for _ in range(iters):
        v = b / (K.T @ u + 1e-300)
        u = inv_a / (K @ v + 1e-300)
    return float(((u[:, None] * K * v[None, :]) * C).sum())


class WassersteinWL:
    """WWL como coordenadas: kernel de transporte óptimo + PCA de kernel.

    El kernel `k(G,G') = exp(−γ · W₁(nube(G), nube(G')))` se calcula sobre todo
    el corpus y se re-expresa como un embedding `(n, d)` cuyo coseno reproduce
    la geometría del kernel, para que encaje en la fusión de §5.5 sin tocarla.
    """

    name = "wwl"

    def __init__(self, dimensions: int, wl_iterations: int, hierarchy_order: list[str],
                 sinkhorn_eps_ratio: float = 0.5, sinkhorn_iters: int = 25) -> None:
        self.dimensions = dimensions
        self.wl_iterations = wl_iterations
        self.hierarchy_order = list(hierarchy_order)
        self.eps_ratio = sinkhorn_eps_ratio
        self.iters = sinkhorn_iters

    def fit_transform(self, graphs: list[nx.Graph]) -> np.ndarray:
        clouds = [_wl_node_cloud(g, self.hierarchy_order, self.wl_iterations)
                  for g in graphs]
        scale = np.vstack([c for c in clouds if len(c)]).std(axis=0)
        scale[scale <= 1e-12] = 1.0
        clouds = [c / scale for c in clouds]

        n = len(graphs)
        D = np.zeros((n, n), dtype=np.float64)
        # ε de Sinkhorn a partir de la distancia típica entre nodos del corpus
        med = np.median([np.linalg.norm(c - c.mean(0), axis=1).mean()
                         for c in clouds if len(c) > 1] or [1.0])
        eps = max(self.eps_ratio * float(med), 1e-3)
        step = max(1, n // 10)
        for i in range(n):
            if i % step == 0:
                logger.info("WWL: fila %d/%d", i, n)
            ci = clouds[i]
            for j in range(i + 1, n):
                cj = clouds[j]
                if not len(ci) or not len(cj):
                    d = 0.0
                else:
                    Cij = np.linalg.norm(ci[:, None, :] - cj[None, :, :], axis=2)
                    d = _sinkhorn_w1(Cij, eps, self.iters)
                D[i, j] = D[j, i] = d

        pos = D[D > 0]
        gamma = 1.0 / float(np.median(pos)) if pos.size else 1.0
        K = np.exp(-gamma * D)
        # PCA de kernel: K_c = J K J, autovectores * sqrt(autovalores)
        J = np.eye(n) - np.full((n, n), 1.0 / n)
        Kc = J @ K @ J
        w, V = np.linalg.eigh((Kc + Kc.T) / 2.0)
        order = np.argsort(w)[::-1]
        k = min(self.dimensions, int((w > 1e-9).sum()), n)
        idx = order[:max(k, 1)]
        return V[:, idx] * np.sqrt(np.clip(w[idx], 0.0, None))


# --------------------------------------------------------------------------- #
# Geometric Scattering Transform   (Gao, Wolf, Hirn, ICML 2019)
# --------------------------------------------------------------------------- #


class GeometricScattering:
    """Cascada de wavelets de difusión + módulo + momentos estadísticos.

    `P = ½(I + A D⁻¹)` es el random walk perezoso; los wavelets
    `Ψ_j = P^(2^(j−1)) − P^(2^j)` leen la escala `2^j`. Se aplican a varias
    señales de nodo (grado, clustering, geometría), se toma `|·|` y se resume el
    grafo con momentos: sale un vector de longitud fija e invariante a la
    numeración (Gao et al., 2019). No entrena nada: cae bien en el régimen de
    pocos cientos de subgrafos.
    """

    name = "scattering"

    def __init__(self, dimensions: int, scales: int = 4,
                 moments: tuple[int, ...] = (1, 2, 3, 4)) -> None:
        self.dimensions = dimensions
        self.J = scales
        self.moments = moments

    def _vector(self, g: nx.Graph) -> np.ndarray:
        nodes = list(g.nodes)
        n = len(nodes)
        if n == 0:
            return np.zeros(1)
        A = nx.to_numpy_array(g, nodelist=nodes)
        d = np.clip(A.sum(axis=1), 1.0, None)
        P = 0.5 * (np.eye(n) + A / d[None, :])
        Ppow = [np.eye(n), P]
        for _ in range(self.J):
            Ppow.append(Ppow[-1] @ Ppow[-1])
        psis = [Ppow[k] - Ppow[k + 1] for k in range(1, self.J + 1)]
        low = Ppow[self.J + 1]

        deg = A.sum(axis=1)
        clus = np.array([nx.clustering(g, v) for v in nodes], dtype=np.float64)
        xy = np.array([g.nodes[v].get("xy", (0.0, 0.0)) for v in nodes], dtype=np.float64)
        xy = xy - xy.mean(axis=0)
        rg = float(np.sqrt((xy ** 2).sum(axis=1).mean())) or 1.0
        signals = [deg / (deg.max() or 1.0), clus,
                   np.linalg.norm(xy, axis=1) / rg, xy[:, 0] / rg, xy[:, 1] / rg]

        mom = lambda x: [float(np.sum(np.abs(x) ** q)) for q in self.moments]
        out: list[float] = []
        for x in signals:
            out += mom(low @ x)
            w1 = [psi @ x for psi in psis]
            for a in range(self.J):
                out += mom(np.abs(w1[a]))
                for b in range(a + 1, self.J):
                    out += mom(np.abs(psis[b] @ np.abs(w1[a])))
        return np.asarray(out, dtype=np.float64)

    def fit_transform(self, graphs: list[nx.Graph]) -> np.ndarray:
        X = np.log1p(np.vstack([self._vector(g) for g in graphs]))
        return _pca(X, self.dimensions)


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


def build_embedder(cfg, method: str | None = None) -> Embedder:
    """Instancia un método de embedding, con el fallback de §5.3.

    `method` sobreescribe `config.yaml`; sirve para el modo comparativo, que
    calcula varios embeddings en la misma corrida (`features.embedding.compare`).
    """
    e = cfg.features.embedding
    seed = cfg.project.random_state
    name = method or e.method

    if name == "netlsd":
        return NetLSD(e.dimensions)
    if name == "graph2vec":
        return Graph2Vec(e.dimensions, e.wl_iterations, e.epochs, seed)
    if name == "wwl":
        return WassersteinWL(e.dimensions, e.wl_iterations, cfg.network.hierarchy_order)
    if name == "scattering":
        return GeometricScattering(e.dimensions)

    dep = _NEEDS_OPTIONAL.get(name)
    logger.warning(
        "El método de embedding %r necesita %s, que no está instalado. "
        "Se usa el fallback netlsd + PCA (§5.3). Las coordenadas del dashboard "
        "se generan igual, pero no son las de %r.",
        name, dep, name,
    )
    return NetLSD(e.dimensions)


def build_embedders(cfg) -> list[Embedder]:
    """Los métodos a comparar en esta corrida: el principal primero, sin repetir.

    Vacío `features.embedding.compare` ⇒ solo el principal, comportamiento de
    siempre.
    """
    names = [cfg.features.embedding.method]
    for n in getattr(cfg.features.embedding, "compare", []) or []:
        if n not in names:
            names.append(n)
    return [build_embedder(cfg, n) for n in names]
