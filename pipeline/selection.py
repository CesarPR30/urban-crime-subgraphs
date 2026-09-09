"""Cuántos hotspots se queda cada mes (§4.3, extensión).

El extractor produce **todas** las regiones persistentes del campo de densidad
—unas cuantas decenas o centenares por mes— y hay que decidir cuáles se
retienen. Hasta aquí la respuesta era `top_k = 20`, fija.

Un K fijo tiene un problema concreto: fuerza el mismo número de hotspots en un
mes tranquilo y en uno con un brote. Si en enero hay ocho concentraciones
reales, el top-20 rellena con doce regiones que no son nada; si en julio hay
veinticinco, se pierden cinco. El número de hotspots deja de ser un resultado y
pasa a ser un parámetro.

Este módulo ofrece tres criterios, seleccionables desde `config.yaml`:

``fixed``
    Los `top_k` de siempre. Es el valor por defecto para que nada cambie en
    silencio, y sigue siendo lo correcto cuando se quiere una huella comparable
    mes a mes por construcción.

``percentile``
    Retiene las regiones cuyo crimen capturado esté en el percentil `q` **de
    ese mes**. Adaptativo y sin coste: el número sale de la forma de la
    distribución del propio mes. Lo que no da es una interpretación
    estadística — que una región esté en el decil superior de su mes no dice
    que sea improbable bajo azar.

``montecarlo``
    Significancia por distribución nula, siguiendo a Kulldorff (1997) y su
    adaptación a red vial de Shiode & Shiode (2020). Se simulan `replicates`
    realizaciones del mes bajo la hipótesis nula, se extrae de cada una el
    estadístico **máximo**, y se conserva toda región observada cuyo
    estadístico supere ese máximo con probabilidad menor que `alpha_sig`.

    Comparar contra el máximo por réplica —y no contra la distribución de
    todas las regiones nulas— es lo que hace que **no haga falta corregir por
    test múltiple**: el estadístico ya es el del extremo, así que el error de
    tipo I queda controlado a nivel de familia (Kulldorff & Nagarwalla, 1995).
    Es exactamente el argumento por el que Shiode & Shiode prefieren Scan
    Statistic a GAM o a Besag–Newell.

## La hipótesis nula

Shiode & Shiode (2020, §2) asumen *proceso de Poisson homogéneo y continuo
sobre la red*: los incidentes se reparten al azar a lo largo de las calles con
intensidad constante. Aquí el campo vive sobre nodos, no sobre segmentos
continuos, así que el equivalente es repartir los crímenes del mes uniformemente
entre los nodos de la red (`uniform`).

Se ofrece además un nulo alternativo (`permutation`) que baraja los conteos
observados **entre los nodos que ya tenían crimen**. Los dos responden a
preguntas distintas y conviene no confundirlas:

* `uniform` pregunta *¿está el crimen más concentrado de lo que estaría si
  cayera al azar sobre la ciudad?* Es el nulo del paper. Detecta mucho, porque
  el crimen urbano nunca es uniforme: hay calles sin portales.
* `permutation` pregunta *¿está más concentrado de lo que estaría si la misma
  cantidad de crimen se repartiese al azar entre los sitios donde de hecho pasa
  algo?* Condiciona sobre dónde hay oportunidad y es bastante más exigente.

El primero mide concentración contra la geografía; el segundo, contra la
oportunidad.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# El estadístico
# --------------------------------------------------------------------------- #


def poisson_llr(
    n_in: np.ndarray | float,
    size_in: np.ndarray | float,
    n_total: float,
    size_total: float,
) -> np.ndarray:
    """Log-razón de verosimilitud de Kulldorff para una región.

    .. math::
        \Lambda = n_Z \log\frac{n_Z}{\lambda_Z}
                 + (n_G - n_Z)\log\frac{n_G - n_Z}{n_G - \lambda_Z}

    con :math:`\lambda_Z = n_G\,|Z|/|N|` los casos esperados bajo el nulo, y
    cero cuando la región no está enriquecida (:math:`n_Z \le \lambda_Z`).

    **Por qué no basta con contar crímenes.** Puntuar una región por el crimen
    que captura es lo correcto para *ordenarlas* —el objetivo declarado es
    cubrir crimen real (§4.2)— pero es inservible como estadístico de contraste,
    y el nulo lo deja a la vista: repartiendo el crimen uniformemente por la
    red, el campo se queda sin estructura, el extractor devuelve unas pocas
    regiones enormes y cada una captura cientos de crímenes por puro tamaño. El
    máximo nulo salía *mayor* que cualquier región observada, con lo que nada
    era significativo. El fallo no era del nulo sino del estadístico: comparaba
    concentración contra tamaño.

    :math:`\Lambda` normaliza por el tamaño esperado, que es exactamente lo que
    Shiode & Shiode (2020, ec. 1) hacen con la longitud de su ventana de
    búsqueda. Aquí el tamaño se mide en **nodos** y no en metros, porque el
    campo vive sobre nodos: bajo el nulo cada nodo es igual de probable, así que
    el número de nodos *es* la población en riesgo de la región.
    """
    n_in = np.asarray(n_in, dtype=np.float64)
    lam = n_total * np.asarray(size_in, dtype=np.float64) / max(size_total, 1.0)

    out = np.zeros(n_in.shape, dtype=np.float64)
    # Solo las regiones enriquecidas puntúan. Una región con *menos* crimen del
    # esperado es un hueco frío, no un hotspot, y el test es de una cola.
    ok = (n_in > lam) & (n_in > 0) & (lam > 0) & (n_in < n_total)
    if not np.any(ok):
        return out
    ni, li = n_in[ok], lam[ok]
    out[ok] = (ni * np.log(ni / li)
               + (n_total - ni) * np.log((n_total - ni) / (n_total - li)))
    return out

#: Criterios admitidos. `fixed` es el de la especificación original.
METHODS = ("fixed", "percentile", "montecarlo")

#: Nulos admitidos para `montecarlo`.
NULLS = ("uniform", "permutation")


@dataclass
class Selection:
    """Qué regiones sobreviven y por qué."""

    keep: list[int]
    """Posiciones dentro de la lista de candidatas, ya ordenada de mayor a
    menor puntuación."""

    method: str
    n_candidates: int
    threshold: float = 0.0
    """Puntuación mínima retenida. Con `montecarlo`, el cuantil del nulo del
    estadístico de verosimilitud."""

    p_values: list[float] = field(default_factory=list)
    """Un p-valor por región retenida, en el mismo orden que `keep`. Vacío
    salvo en `montecarlo`."""

    null_max: list[float] = field(default_factory=list)
    """Los estadísticos máximos por réplica. Se guardan para poder dibujar la
    distribución nula en el dashboard: un umbral sin su distribución detrás no
    se puede juzgar."""

    def summary(self) -> dict:
        out: dict = {
            "method": self.method,
            "candidates": self.n_candidates,
            "kept": len(self.keep),
            "threshold": round(float(self.threshold), 4),
        }
        if self.p_values:
            out["p_max"] = round(max(self.p_values), 4)
        if self.null_max:
            arr = np.asarray(self.null_max, dtype=np.float64)
            out["null"] = {
                "replicates": int(arr.size),
                "mean": round(float(arr.mean()), 2),
                "p95": round(float(np.quantile(arr, 0.95)), 2),
                "max": round(float(arr.max()), 2),
            }
        return out


# --------------------------------------------------------------------------- #
# Criterios
# --------------------------------------------------------------------------- #


def _clamp(keep: list[int], min_k: int, max_k: int | None, n: int) -> list[int]:
    """Aplica los topes de seguridad al conjunto retenido.

    `min_k` evita que un mes se quede sin nada que enseñar cuando el criterio
    es severo; `max_k` acota el coste aguas abajo —cada hotspot arrastra un
    embedding y una fila en la matriz de similitud— y evita que un mes
    degenerado inunde el artefacto. Ambos son topes, no objetivos: si el
    criterio devuelve algo entre los dos, mandan los datos.
    """
    k = len(keep)
    if k < min_k:
        keep = list(range(min(min_k, n)))
    elif max_k is not None and k > max_k:
        keep = keep[:max_k]
    return keep


def select_fixed(scores: np.ndarray, *, top_k: int) -> Selection:
    """Los `top_k` primeros. `scores` viene ordenado de mayor a menor."""
    keep = list(range(min(top_k, scores.size)))
    return Selection(
        keep=keep, method="fixed", n_candidates=int(scores.size),
        threshold=float(scores[keep[-1]]) if keep else 0.0,
    )


def select_percentile(
    scores: np.ndarray, *, q: float, min_k: int = 1, max_k: int | None = None
) -> Selection:
    """Regiones por encima del percentil `q` de su propio mes.

    `q` está en [0, 1). Con 0.90 sobrevive el decil superior de las candidatas
    del mes, sea eso 8 regiones o 25.
    """
    if scores.size == 0:
        return Selection(keep=[], method="percentile", n_candidates=0)
    thr = float(np.quantile(scores, q))
    keep = [i for i, s in enumerate(scores) if s >= thr]
    keep = _clamp(keep, min_k, max_k, int(scores.size))
    return Selection(
        keep=keep, method="percentile", n_candidates=int(scores.size),
        threshold=thr,
    )


def select_montecarlo(
    llr: np.ndarray,
    null_max: np.ndarray,
    *,
    alpha_sig: float = 0.05,
    min_k: int = 1,
    max_k: int | None = None,
) -> Selection:
    """Regiones significativas contra la distribución nula del máximo.

    El p-valor de una región con estadístico `T` es

    .. math:: p = \\frac{1 + \\#\\{T^* \\ge T\\}}{B + 1}

    que es la forma de Dwass: el `+1` cuenta la propia observación y evita el
    p-valor cero, que sería una afirmación que `B` réplicas no pueden sostener.
    Con `B = 99` el p-valor más pequeño alcanzable es 0.01.
    """
    b = int(null_max.size)
    if llr.size == 0 or b == 0:
        return Selection(keep=[], method="montecarlo", n_candidates=int(llr.size),
                         null_max=[float(x) for x in null_max])

    # `llr` viene en el orden de las candidatas (que están ordenadas por crimen
    # capturado, no por verosimilitud), así que el `searchsorted` se hace sobre
    # el nulo ordenado y devuelve el conteo para cada región de golpe.
    ordered = np.sort(null_max)
    ge = b - np.searchsorted(ordered, llr, side="left")
    p = (1.0 + ge) / (b + 1.0)

    keep = [i for i, pv in enumerate(p) if pv < alpha_sig]
    keep = _clamp(keep, min_k, max_k, int(llr.size))
    return Selection(
        keep=keep, method="montecarlo", n_candidates=int(llr.size),
        threshold=float(np.quantile(ordered, 1.0 - alpha_sig)),
        p_values=[float(p[i]) for i in keep],
        null_max=[float(x) for x in null_max],
    )


# --------------------------------------------------------------------------- #
# Distribución nula
# --------------------------------------------------------------------------- #


def null_max_statistic(
    n_crimes: int,
    pool: np.ndarray,
    observed_counts: np.ndarray,
    *,
    kernels,
    g,
    segment_fn,
    alpha: float,
    f_min_ratio: float,
    replicates: int,
    null: str,
    rng: np.random.Generator,
    progress=None,
) -> np.ndarray:
    """Estadístico máximo de cada réplica bajo la hipótesis nula.

    Args:
        n_crimes: total de crímenes del mes, que se conserva en cada réplica.
        pool: nodos candidatos a recibir crimen bajo `uniform`. Sus kernels
            tienen que estar ya construidos: calcularlos aquí, dentro del bucle
            de réplicas, dispararía una Dijkstra por nodo nuevo y por réplica.
        observed_counts: `c(v)` observado; solo lo usa `permutation`.
        segment_fn: `hotspots.segment`, inyectado para no importar en círculo.

    El estadístico es la **log-razón de verosimilitud de la mejor región**
    (`poisson_llr`), la misma que se calcula sobre el dato observado. Que el
    estadístico del nulo y el del dato salgan del mismo código no es un
    detalle: cualquier asimetría entre ambos se leería como significancia.
    """
    if null not in NULLS:
        raise ValueError(f"null {null!r} desconocido; usa uno de {NULLS}")
    if n_crimes <= 0 or replicates <= 0:
        return np.zeros(0, dtype=np.float64)

    out = np.zeros(replicates, dtype=np.float64)
    active = np.nonzero(observed_counts)[0]
    obs_vals = observed_counts[active]

    for b in range(replicates):
        if null == "uniform":
            # Poisson homogéneo sobre la red: cada crimen cae en un nodo al
            # azar, con reemplazo, y se agregan por nodo.
            hits = rng.integers(0, pool.size, size=n_crimes)
            cnt = np.bincount(hits, minlength=pool.size).astype(np.float64)
            nz = np.nonzero(cnt)[0]
            src, val = pool[nz], cnt[nz]
        else:
            # Se conserva el multiconjunto de conteos y el conjunto de sitios
            # activos; solo se rompe la correspondencia entre ambos.
            src = active
            val = rng.permutation(obs_vals)

        f = np.zeros(g.n, dtype=np.float64)
        for s, c in zip(src, val):
            nodes, w = kernels.get(int(s))
            f[nodes] += c * w

        full = np.zeros(g.n, dtype=np.float64)
        full[src] = val
        regions, _, _ = segment_fn(f, g, alpha, f_min_ratio)
        if regions:
            sizes = np.fromiter((len(idx) for idx in regions.values()),
                                dtype=np.float64, count=len(regions))
            hits = np.fromiter((float(full[idx].sum()) for idx in regions.values()),
                               dtype=np.float64, count=len(regions))
            out[b] = float(poisson_llr(hits, sizes, float(n_crimes), float(g.n)).max())
        if progress is not None and (b + 1) % 10 == 0:
            progress(b + 1, replicates)

    return out


def resolve(
    cfg_sel,
    scores: np.ndarray,
    llr: np.ndarray,
    null_max: np.ndarray | None,
) -> Selection:
    """Aplica el criterio configurado a las candidatas de un mes.

    `scores` es el crimen capturado —con el que se **ordenan** las regiones— y
    `llr` su verosimilitud —con la que se decide si son **significativas**. Son
    dos cosas distintas y hacen falta las dos: la región que más crimen captura
    no tiene por qué ser la más improbable bajo azar.
    """
    m = cfg_sel.method
    if m == "fixed":
        return select_fixed(scores, top_k=cfg_sel.top_k)
    if m == "percentile":
        return select_percentile(scores, q=cfg_sel.percentile,
                                 min_k=cfg_sel.min_k, max_k=cfg_sel.max_k)
    if m == "montecarlo":
        return select_montecarlo(
            llr, null_max if null_max is not None else np.zeros(0),
            alpha_sig=cfg_sel.alpha_sig, min_k=cfg_sel.min_k, max_k=cfg_sel.max_k,
        )
    raise ValueError(f"selection.method {m!r} desconocido; usa uno de {METHODS}")
