"""Identidad de un hotspot a través de los meses: trayectorias.

Hasta aquí cada mes se extraía por separado y `2018-03_h07` y `2018-04_h11` eran
dos objetos sin relación, aunque fueran la misma esquina. Este módulo los enlaza,
y de ese enlace salen las cuatro cantidades que pide la nota de tesis:

* **Frequency(H)** = `#meses donde aparece / #meses analizados`.
* **Intensity(H)** = crímenes del subgrafo en el mes (recuento o densidad).
* **Stability(H)** = `1/(k-1) · Σ IoU(H_t, H_{t+1})`, media del solape entre
  apariciones consecutivas.
* **Movement(H)** = `1/(k-1) · Σ d_G(seed_t, seed_{t+1})`, media de la distancia
  **geodésica sobre la red** entre semillas consecutivas.

## Cómo se decide que dos subgrafos son "el mismo"

Por solape de nodos, que es lo que indica la nota: *«si 2 grafos comparten más
de t, entonces se considera el mismo»*. Se usa el índice de Jaccard

    IoU(A, B) = |A ∩ B| / |A ∪ B|

y no el recuento crudo de nodos compartidos, porque el recuento premia a los
subgrafos grandes: dos regiones de 300 nodos que compartan 30 tienen tanto en
común como dos de 35 que compartan 30, y solo el segundo par es la misma esquina.

## Por qué no se toma la componente conexa

La tentación es construir el grafo «A se parece a B» sobre los 4 160 subgrafos y
quedarse con sus componentes conexas. No sirve: el enlace simple **encadena**. A
solapa con B, B con C, y C puede estar a dos kilómetros de A; sobre 90 meses eso
funde media ciudad en una sola trayectoria y la frecuencia resultante no
significa nada.

Se hace en cambio seguimiento en orden temporal, que es el planteamiento estándar
en seguimiento de objetos: cada mes, los subgrafos se emparejan con las
trayectorias **activas** comparando contra su última aparición, uno a uno y por
IoU descendente. Lo que no encuentra pareja abre trayectoria nueva.

## El hueco

Una trayectoria sobrevive `max_gap` meses sin aparecer. Sin eso, un hotspot que
falta un mes se parte en dos trayectorias de frecuencia baja, y la métrica que
la nota quiere —*«muy frecuente»* frente a *«resaltó por algo puntual»*— mediría
sobre todo el ruido mes a mes. Con eso, una ausencia corta es lo que
intuitivamente es: el mismo sitio, apagado un mes.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

logger = logging.getLogger(__name__)


def iou(a: set[int], b: set[int]) -> float:
    """Índice de Jaccard entre dos conjuntos de nodos."""
    if not a or not b:
        return 0.0
    inter = len(a & b)
    return inter / (len(a) + len(b) - inter)


@dataclass
class Trajectory:
    """Un mismo sitio a lo largo de la ventana."""

    id: str
    members: list[str] = field(default_factory=list)
    months: list[str] = field(default_factory=list)
    crimes: list[int] = field(default_factory=list)
    densities: list[float] = field(default_factory=list)
    n_nodes: list[int] = field(default_factory=list)
    seeds: list[int] = field(default_factory=list)
    #: IoU entre apariciones consecutivas; `k-1` valores.
    ious: list[float] = field(default_factory=list)
    #: Distancia geodésica entre semillas consecutivas, en metros.
    steps_m: list[float] = field(default_factory=list)
    #: Unión de todos los nodos por los que pasó.
    union: set[int] = field(default_factory=set)

    @property
    def k(self) -> int:
        return len(self.members)

    def frequency(self, n_months: int) -> float:
        """`#meses donde aparece / #meses analizados`."""
        return len(set(self.months)) / n_months if n_months else 0.0

    @property
    def intensity_mean(self) -> float:
        return float(np.mean(self.crimes)) if self.crimes else 0.0

    @property
    def intensity_max(self) -> int:
        return int(max(self.crimes)) if self.crimes else 0

    @property
    def density_mean(self) -> float:
        return float(np.mean(self.densities)) if self.densities else 0.0

    @property
    def stability(self) -> float | None:
        """Media de los IoU consecutivos. `None` con una sola aparición.

        Devolver `None` y no `0` importa: un subgrafo que aparece una vez no es
        inestable, es que la pregunta no aplica. Un 0 lo metería en el cubo de
        «muy móvil» y contaminaría el eje entero.
        """
        return float(np.mean(self.ious)) if self.ious else None

    @property
    def movement_m(self) -> float | None:
        """Media del desplazamiento geodésico de la semilla, en metros."""
        return float(np.mean(self.steps_m)) if self.steps_m else None

    def to_dict(self, n_months: int) -> dict:
        return {
            "id": self.id,
            "members": self.members,
            "months": self.months,
            "k": self.k,
            "frequency": round(self.frequency(n_months), 4),
            "intensity_mean": round(self.intensity_mean, 2),
            "intensity_max": self.intensity_max,
            "density_mean": round(self.density_mean, 4),
            "nodes_mean": round(float(np.mean(self.n_nodes)), 1) if self.n_nodes else 0,
            "nodes_union": len(self.union),
            "stability": (None if self.stability is None
                          else round(self.stability, 4)),
            "movement_m": (None if self.movement_m is None
                           else round(self.movement_m, 1)),
            "crimes": self.crimes,
        }


def _geodesic(g, a: int, b: int, cutoff: float) -> float:
    """Distancia sobre la red entre dos nodos OSM, o `cutoff` si no alcanza.

    Se acota a propósito. Un Dijkstra sin tope entre dos semillas que acabaron
    en extremos opuestos de Lima recorrería los 135 633 nodos, y por cada par;
    además, una vez que la semilla se ha ido más de `cutoff` la magnitud exacta
    da igual, ya no es el mismo sitio. Devolver el tope y no `inf` mantiene la
    media finita, y se declara como censura en el reporte.
    """
    from .density import bounded_dijkstra

    ia, ib = g.index.get(a), g.index.get(b)
    if ia is None or ib is None:
        return cutoff
    if ia == ib:
        return 0.0
    nodes, dist = bounded_dijkstra(g, ia, cutoff)
    hit = np.flatnonzero(nodes == ib)
    return float(dist[hit[0]]) if hit.size else cutoff


def link(hotspots, months: list[str], *, min_iou: float = 0.2,
         max_gap: int = 3, g=None, cutoff_m: float = 3000.0) -> list[Trajectory]:
    """Enlaza subgrafos entre meses y devuelve las trayectorias.

    `hotspots` son dicts con `id`, `month`, `nodes`, `crimes`, `density`,
    `n_nodes` y `seed_node` — el artefacto de `crimepipe hotspots`.

    Con `g` (un `CSRGraph`) se calcula además el desplazamiento geodésico de la
    semilla; sin él, `movement_m` queda a `None` y el resto funciona igual.
    """
    by_month: dict[str, list[dict]] = {m: [] for m in months}
    for h in hotspots:
        if h["month"] in by_month:
            by_month[h["month"]].append(h)

    month_ix = {m: i for i, m in enumerate(months)}
    trajs: list[Trajectory] = []
    #: trayectoria -> (índice del último mes visto, conjunto de nodos, semilla)
    last: dict[int, tuple[int, set[int], int]] = {}

    for m in months:
        i = month_ix[m]
        # Solo las que siguen vivas: pasado `max_gap` una trayectoria se cierra
        # y un subgrafo en el mismo sitio abrirá otra nueva.
        active = [t for t, (j, _, _) in last.items() if i - j <= max_gap]

        cand: list[tuple[float, int, int]] = []
        here = by_month[m]
        sets = [set(h["nodes"]) for h in here]
        for hi, s in enumerate(sets):
            for t in active:
                v = iou(s, last[t][1])
                if v >= min_iou:
                    cand.append((v, hi, t))
        cand.sort(key=lambda c: -c[0])

        taken_h: set[int] = set()
        taken_t: set[int] = set()
        matched: dict[int, tuple[int, float]] = {}
        for v, hi, t in cand:
            if hi in taken_h or t in taken_t:
                continue
            taken_h.add(hi)
            taken_t.add(t)
            matched[hi] = (t, v)

        for hi, h in enumerate(here):
            s = sets[hi]
            if hi in matched:
                t, v = matched[hi]
                tr = trajs[t]
                tr.ious.append(v)
                if g is not None:
                    tr.steps_m.append(
                        _geodesic(g, last[t][2], h["seed_node"], cutoff_m))
            else:
                t = len(trajs)
                trajs.append(Trajectory(id=f"T{t:05d}"))
                tr = trajs[t]
            tr.members.append(h["id"])
            tr.months.append(m)
            tr.crimes.append(int(h["crimes"]))
            tr.densities.append(float(h["density"]))
            tr.n_nodes.append(int(h["n_nodes"]))
            tr.seeds.append(int(h["seed_node"]))
            tr.union |= s
            last[t] = (i, s, int(h["seed_node"]))

    return trajs


def quadrants(trajs: list[Trajectory], n_months: int,
              *, freq_split: float | None = None,
              stab_split: float | None = None) -> dict:
    """Los cuatro cubos de la nota: frecuente/episódico x estable/móvil.

    Los cortes son las **medianas observadas** y no valores fijos: «frecuente»
    no significa lo mismo en una ventana de 24 meses que en una de 90, y fijar
    0.5 dejaría los cuatro cubos vacíos salvo uno. Se devuelven los cortes
    usados para que el reparto sea auditable.
    """
    usable = [t for t in trajs if t.stability is not None]
    if not usable:
        return {"freq_split": None, "stab_split": None, "counts": {}, "labels": {}}

    fr = np.array([t.frequency(n_months) for t in usable])
    st = np.array([t.stability for t in usable])
    fs = float(np.median(fr)) if freq_split is None else freq_split
    ss = float(np.median(st)) if stab_split is None else stab_split

    labels: dict[str, str] = {}
    for t in usable:
        f = t.frequency(n_months) >= fs
        s = t.stability >= ss
        labels[t.id] = ("frecuente_estable" if f and s else
                        "frecuente_movil" if f else
                        "episodico_estable" if s else "episodico_movil")
    counts: dict[str, int] = {}
    for v in labels.values():
        counts[v] = counts.get(v, 0) + 1
    return {"freq_split": round(fs, 4), "stab_split": round(ss, 4),
            "counts": counts, "labels": labels,
            "single_appearance": len(trajs) - len(usable)}
