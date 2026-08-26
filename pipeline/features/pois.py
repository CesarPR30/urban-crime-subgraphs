"""Perfil de POIs por subgrafo (PROJECT_SPEC §5.6).

Los POIs se asocian a un subgrafo por **contención en la envolvente convexa
bufferizada** de sus nodos. Es exactamente la misma construcción que el
dashboard dibuja como área del hotspot, así que lo que se ve en el mapa es
literalmente la zona de la que salen estos POIs — no dos criterios distintos que
por casualidad se parecen.

De cada zona salen:

* la distribución normalizada `p_k` sobre las ocho categorías,
* la densidad de POIs por nodo,
* la **entropía de Shannon** `H = −Σ p_k log p_k`, que mide diversidad
  funcional: alta = mezcla de usos, baja = zona monofuncional.

> **§5.1:** nada de esto entra en la similitud. `POIProfile` es una variable a
> contrastar en la fase 3, no un descriptor de forma. La comparación se hace
> *entre* subgrafos que ya se parecen estructuralmente, y la pregunta es cuánta
> de la variación de crimen restante se alinea con diferencias de composición.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass

import numpy as np
from shapely.geometry import MultiPoint, Point
from shapely.strtree import STRtree

from .structural import Subgraph, transformer_for

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class PoiProfile:
    """§5.6 para un subgrafo. Solo reporte y contraste, nunca similitud."""

    counts: np.ndarray        # (8,) conteo por categoría
    total: int
    per_node: float
    entropy: float
    #: Entropía normalizada a [0,1] dividiendo por log(k) — el máximo posible
    #: con k categorías. Sin esto, comparar entropías entre taxonomías de
    #: distinto tamaño no significa nada.
    entropy_norm: float

    def distribution(self) -> np.ndarray:
        return self.counts / self.total if self.total else np.zeros_like(self.counts)


def shannon(counts: np.ndarray) -> float:
    """`H = −Σ pₖ log pₖ` en nats. Las categorías ausentes no contribuyen.

    El convenio `0·log 0 = 0` es el estándar en teoría de la información y aquí
    además es necesario: casi ninguna zona tiene las ocho categorías.
    """
    total = counts.sum()
    if total <= 0:
        return 0.0
    p = counts[counts > 0] / total
    return float(-(p * np.log(p)).sum())


def hulls(subgraphs: list[Subgraph], buffer_m: float):
    """Envolvente convexa bufferizada de cada subgrafo, en metros proyectados.

    Se trabaja en el plano métrico y no en grados: bufferizar 50 m en grados
    deformaría la zona, porque un grado de longitud no mide lo mismo que uno de
    latitud. `Subgraph.xy` ya viene proyectado por `build_subgraphs`.
    """
    out = []
    for sg in subgraphs:
        pts = MultiPoint([tuple(p) for p in sg.xy])
        out.append(pts.convex_hull.buffer(buffer_m))
    return out


def assign(subgraphs: list[Subgraph], poi_lat, poi_lon, poi_cat,
           categories: list[str], buffer_m: float) -> list[PoiProfile]:
    """Asocia POIs a subgrafos y calcula el perfil de cada uno.

    Un POI puede caer en **varias** zonas y se cuenta en todas: las envolventes
    de meses distintos se solapan por construcción, y descontar el solape haría
    que el perfil de una zona dependiera de qué otras zonas existen.
    """
    if not subgraphs:
        return []

    # Mismo plano métrico que los subgrafos, no uno nuevo: los POIs se extienden
    # más allá de los nodos de la red y elegir zona por su cuenta daría dos
    # planos distintos, con las contenciones mal en los bordes.
    to_m = transformer_for(subgraphs[0].epsg)
    px, py = to_m(np.asarray(poi_lon, dtype=np.float64),
                  np.asarray(poi_lat, dtype=np.float64))

    cat_index = {c: i for i, c in enumerate(categories)}
    codes = np.array([cat_index.get(str(c), -1) for c in poi_cat], dtype=np.int64)

    points = [Point(x, y) for x, y in zip(np.asarray(px), np.asarray(py))]
    tree = STRtree(points)

    zones = hulls(subgraphs, buffer_m)
    profiles: list[PoiProfile] = []
    k = len(categories)
    for sg, zone in zip(subgraphs, zones):
        counts = np.zeros(k, dtype=np.int64)
        # El árbol filtra por bbox; `contains` decide de verdad.
        for i in tree.query(zone):
            if zone.contains(points[i]) and codes[i] >= 0:
                counts[codes[i]] += 1
        total = int(counts.sum())
        h = shannon(counts)
        profiles.append(PoiProfile(
            counts=counts,
            total=total,
            per_node=total / sg.n if sg.n else 0.0,
            entropy=h,
            entropy_norm=h / math.log(k) if k > 1 else 0.0,
        ))
    return profiles
