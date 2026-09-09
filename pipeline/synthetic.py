"""Validación con datos sintéticos: ¿acierta el extractor dónde está el hotspot?

Réplica del experimento de **Shiode, S. & Shiode, N. (2020)**, *A network-based
scan statistic for detecting the exact location and extent of hotspots along
urban streets* (Computers, Environment and Urban Systems), §3 y §5.

## Por qué hace falta

Todas las métricas del pipeline hasta ahora —cobertura, densidad, *lift*,
ganancia sobre la línea base— son **relativas**: dicen que el extractor
topológico captura más crimen que hacer crecer regiones a lo bruto. Ninguna
dice si acierta *dónde está el hotspot*, porque sobre datos reales no se sabe
dónde está: no hay verdad conocida contra la que comparar.

Con datos sintéticos sí la hay. Se inyectan concentraciones en calles elegidas
a mano, se le pasa el resultado al extractor sin decirle nada, y se mide cuánto
de lo que encuentra es de verdad y cuánto de lo que hay encuentra.

## El proceso de Poisson por clústeres (Shiode §3)

Tres pasos, igual que el paper:

1. **Puntos padre.** `n_parents` aristas de la red elegidas al azar. Son las
   calles donde va a haber concentración.
2. **Puntos hijo.** `n_offspring` incidentes repartidos entre esas aristas. Son
   el hotspot.
3. **Fondo.** `n_background` incidentes uniformes sobre toda la red, que es el
   ruido sobre el que hay que distinguirlos.

**La adaptación honesta.** El paper trabaja sobre segmentos continuos y coloca
los hijos *a lo largo* de la arista; aquí el campo vive sobre nodos, así que un
incidente en una arista se asigna a uno de sus dos extremos. En consecuencia la
verdad de campo —`Strue`— es el conjunto de **nodos extremo** de las aristas
sembradas, y los puntos de referencia con los que se miden PPV y sensibilidad
son los nodos de la red, no una rejilla de 30 m como en la Figura 1(f) del
paper. Es la traducción directa del experimento a este modelo de datos, pero no
es idéntica: los números **no** son comparables con la Tabla 1 del paper, solo
entre los métodos que se comparan aquí.

## Las métricas (Shiode §5, ec. 3 y 4)

.. math::
    PPV = \\frac{\\#\\{r_i \\in S^* \\cap S_{true}\\}}{\\#\\{r_j \\in S^*\\}}
    \\qquad
    Sens = \\frac{\\#\\{r_i \\in S^* \\cap S_{true}\\}}{\\#\\{r_j \\in S_{true}\\}}

`PPV` mide **sobredisparo**: de lo que el método marca, cuánto es hotspot de
verdad. `Sens` —el paper la llama *specificity*, que es un nombre desafortunado
porque no es la especificidad del análisis de decisión— mide **subdisparo**: de
lo que es hotspot, cuánto encuentra. Se reportan las dos porque cada una sola
se maximiza trivialmente: marcar toda la ciudad da sensibilidad 1, y marcar un
solo nodo acertado da PPV 1. Se añade `F1`, su media armónica, que es lo que el
propio Shiode sugiere como cierre en la discusión.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class Truth:
    """Una realización sintética: los conteos y dónde está la verdad."""

    counts: np.ndarray
    """`c(v)` por índice interno de nodo."""

    true_nodes: np.ndarray
    """`Strue`: nodos de las aristas sembradas."""

    parent_edges: list[tuple[int, int]]
    n_offspring: int
    n_background: int

    @property
    def n_crimes(self) -> int:
        return int(self.counts.sum())


@dataclass
class Score:
    """PPV, sensibilidad y F1 de un método sobre una realización."""

    ppv: float
    sensitivity: float
    n_detected: int
    n_true: int
    n_hit: int

    @property
    def f1(self) -> float:
        d = self.ppv + self.sensitivity
        return 0.0 if d == 0 else 2 * self.ppv * self.sensitivity / d

    def to_dict(self) -> dict:
        return {
            "ppv": round(self.ppv, 4),
            "sensitivity": round(self.sensitivity, 4),
            "f1": round(self.f1, 4),
            "detected": self.n_detected,
            "true": self.n_true,
            "hit": self.n_hit,
        }


@dataclass
class Benchmark:
    """Resultado agregado sobre todas las realizaciones."""

    methods: dict[str, list[Score]] = field(default_factory=dict)
    params: dict = field(default_factory=dict)

    def stats(self, method: str) -> dict:
        """Media, desviación y coeficiente de variación, como la Tabla 1.

        El coeficiente de variación es la columna que más dice del paper: mide
        si el método se comporta igual en todas las realizaciones o depende de
        la suerte del sembrado. Shiode lo usa justo para eso —NetScan tiene
        CV 0.03 en PPV frente a 0.29 de la versión circular—.
        """
        rows = self.methods[method]
        out = {}
        for key in ("ppv", "sensitivity", "f1"):
            v = np.array([getattr(r, key) for r in rows], dtype=np.float64)
            mean = float(v.mean())
            sd = float(v.std(ddof=1)) if v.size > 1 else 0.0
            out[key] = {
                "mean": round(mean, 4),
                "sd": round(sd, 4),
                "cv": round(sd / mean, 4) if mean else 0.0,
            }
        out["detected_mean"] = round(
            float(np.mean([r.n_detected for r in rows])), 1)
        return out

    def to_dict(self) -> dict:
        return {
            "params": self.params,
            "realisations": len(next(iter(self.methods.values()), [])),
            "methods": {
                k: {"summary": self.stats(k), "per_realisation": [s.to_dict() for s in v]}
                for k, v in self.methods.items()
            },
            "tests": self.pairwise_tests(),
        }

    def pairwise_tests(self) -> dict:
        """U de Mann-Whitney entre cada par de métodos (Shiode, Tabla 2).

        No paramétrico y a dos colas, sobre muestras pequeñas —diez
        realizaciones en el paper—, que es donde una t de Student no estaría
        justificada. Si scipy no está, se omite en vez de improvisar un test.
        """
        try:
            from scipy.stats import mannwhitneyu
        except ImportError:      # pragma: no cover
            logger.warning("scipy no disponible: se omite la U de Mann-Whitney")
            return {}

        names = list(self.methods)
        out: dict = {}
        for i, a in enumerate(names):
            for b in names[i + 1:]:
                row = {}
                for key in ("ppv", "sensitivity", "f1"):
                    xa = [getattr(r, key) for r in self.methods[a]]
                    xb = [getattr(r, key) for r in self.methods[b]]
                    try:
                        u, pv = mannwhitneyu(xa, xb, alternative="two-sided")
                        row[key] = {"U": float(u), "p": round(float(pv), 5)}
                    except ValueError:
                        # Ocurre cuando las dos muestras son constantes e
                        # iguales: no hay nada que contrastar y el test no está
                        # definido. Decirlo es más útil que devolver p=1.
                        row[key] = {"U": None, "p": None}
                out[f"{a} vs {b}"] = row
        return out


# --------------------------------------------------------------------------- #
# Recorte de la red
# --------------------------------------------------------------------------- #


def subnetwork(g, n_nodes: int, rng: np.random.Generator):
    """Subred conexa de `n_nodes` nodos, por anchura desde una semilla al azar.

    **Por qué recortar.** Shiode & Shiode no corren su experimento sobre Buffalo
    entera sino sobre un recorte de 900 m x 750 m con 394 puntos de referencia,
    y siembran 14 segmentos: casi el 4 % de la red. Repetir sus 300 puntos sobre
    una ciudad completa cambia el experimento por completo —30 nodos sembrados
    entre 29 832 son el 0.1 %, con un fondo tan diluido que cualquier método
    acierta— y el resultado deja de medir nada.

    Recortar a un vecindario del tamaño del paper devuelve la relación
    señal/ruido a la que el experimento estaba pensado.

    Devuelve un `CSRGraph` nuevo con los nodos renumerados; los `ids` siguen
    siendo los de OSM, así que el resultado es utilizable por todo el pipeline.
    """
    from .density import CSRGraph

    seed = int(rng.integers(0, g.n))
    seen = {seed}
    frontier = [seed]
    order = [seed]
    while frontier and len(order) < n_nodes:
        nxt = []
        for v in frontier:
            for k in range(g.indptr[v], g.indptr[v + 1]):
                u = int(g.indices[k])
                if u not in seen:
                    seen.add(u)
                    order.append(u)
                    nxt.append(u)
                    if len(order) >= n_nodes:
                        break
            if len(order) >= n_nodes:
                break
        frontier = nxt

    keep = np.sort(np.array(order, dtype=np.int64))
    remap = {int(v): i for i, v in enumerate(keep)}

    indptr = np.zeros(keep.size + 1, dtype=np.int64)
    indices: list[int] = []
    weights: list[float] = []
    for i, v in enumerate(keep):
        v = int(v)
        for k in range(g.indptr[v], g.indptr[v + 1]):
            u = int(g.indices[k])
            j = remap.get(u)
            if j is not None:
                indices.append(j)
                weights.append(float(g.weights[k]))
        indptr[i + 1] = len(indices)

    return CSRGraph(
        indptr=indptr,
        indices=np.array(indices, dtype=np.int64),
        weights=np.array(weights, dtype=np.float64),
        ids=g.ids[keep],
        index={int(nid): i for i, nid in enumerate(g.ids[keep])},
    )


# --------------------------------------------------------------------------- #
# Generación
# --------------------------------------------------------------------------- #


def poisson_cluster(
    g,
    *,
    n_parents: int,
    n_offspring: int,
    n_background: int,
    rng: np.random.Generator,
) -> Truth:
    """Una realización del proceso de Poisson por clústeres sobre la red.

    Los hijos se reparten entre las aristas padre **con densidad constante**,
    como en el paper («a total of 200 offspring points are randomly placed
    across these segments with a consistent density»), y cada uno cae en uno de
    los dos extremos de su arista con igual probabilidad.

    Las aristas padre se eligen sin reemplazo: en el paper dos padres cayeron
    en el mismo segmento y quedaron 14 segmentos en vez de 15, lo que hace el
    recuento de la verdad ambiguo. Aquí se evita de entrada.
    """
    # Lista de aristas u<v, en índices internos.
    edges = [
        (v, int(g.indices[k]))
        for v in range(g.n)
        for k in range(g.indptr[v], g.indptr[v + 1])
        if int(g.indices[k]) > v
    ]
    if n_parents > len(edges):
        raise ValueError(
            f"{n_parents} aristas padre pedidas y la red solo tiene {len(edges)}"
        )
    pick = rng.choice(len(edges), size=n_parents, replace=False)
    parents = [edges[int(i)] for i in pick]

    counts = np.zeros(g.n, dtype=np.float64)

    # Paso 2: hijos, repartidos uniformemente entre las aristas sembradas.
    which = rng.integers(0, n_parents, size=n_offspring)
    ends = rng.integers(0, 2, size=n_offspring)
    for w, e in zip(which, ends):
        counts[parents[int(w)][int(e)]] += 1.0

    # Paso 3: fondo uniforme sobre toda la red.
    if n_background:
        bg = rng.integers(0, g.n, size=n_background)
        np.add.at(counts, bg, 1.0)

    true_nodes = np.unique(np.array([n for e in parents for n in e], dtype=np.int64))
    return Truth(
        counts=counts, true_nodes=true_nodes, parent_edges=parents,
        n_offspring=n_offspring, n_background=n_background,
    )


# --------------------------------------------------------------------------- #
# Medición
# --------------------------------------------------------------------------- #


def score(detected: np.ndarray, truth: Truth) -> Score:
    """PPV y sensibilidad de un conjunto de nodos detectados (ec. 3 y 4)."""
    det = np.unique(np.asarray(detected, dtype=np.int64))
    hit = int(np.intersect1d(det, truth.true_nodes, assume_unique=True).size)
    n_det, n_true = int(det.size), int(truth.true_nodes.size)
    return Score(
        ppv=hit / n_det if n_det else 0.0,
        sensitivity=hit / n_true if n_true else 0.0,
        n_detected=n_det, n_true=n_true, n_hit=hit,
    )
