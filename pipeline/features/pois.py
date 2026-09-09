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
    #: Cociente de localización por categoría, contra la ciudad **del mismo
    #: año**. `None` cuando no hay eje temporal. Ver `location_quotient`.
    lq: np.ndarray | None = None
    #: Instantánea usada. `None` con fuente estática.
    year: int | None = None

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


def location_quotient(counts: np.ndarray, city: np.ndarray,
                      n_nodes: int, city_nodes: int) -> np.ndarray:
    """Concentración de cada categoría en la zona, con la **red** como base.

    `LQ_c(H) = [n_c(H) / N_c(t)] / [|H| / |G|]`

    **Por qué la red y no el total de POIs.** Con instantáneas anuales de OSM un
    recuento por año mezcla dos cosas: cuánto cambió la ciudad y cuánto cambió
    el mapa. En Lima `education` se multiplica por 3.4 entre 2018 y 2025 y el
    75 % de esas altas cae en 2018-2019 —el volcado de un padrón escolar—, así
    que leer ese crecimiento como urbanización sería un artefacto.

    El cociente de localización clásico usa el total de POIs como base,
    `(n_c/n) / (N_c/N)`, y **no** basta. Una importación que multiplica una sola
    categoría mueve `n` y `N` en proporciones distintas según la composición de
    la zona, así que el cociente se desplaza: medido sobre este caso, un x4 en
    una categoría lleva el LQ de 2.00 a 1.37. Amortigua la proporción cruda, que
    se va de 0.57 a 0.84, pero no la cancela.

    Con la red como base sí se cancela, y exactamente: la importación multiplica
    `n_c(H)` y `N_c(t)` por el mismo factor, su cociente no se mueve, y el número
    de nodos no depende de OSM. Lo que queda medido es «¿está esta categoría más
    concentrada aquí que en el resto de la Lima de su año?», que es la pregunta.

    `LQ = 1` es la densidad media de la ciudad, `> 1` sobrerrepresentación. Una
    categoría ausente en la ciudad da `0`: no hay base contra la que comparar y
    fingir un 1 sería inventarse el dato.
    """
    out = np.zeros(len(counts), dtype=np.float64)
    if n_nodes <= 0 or city_nodes <= 0:
        return out
    share_nodes = n_nodes / city_nodes
    ok = city > 0
    out[ok] = (counts[ok] / city[ok]) / share_nodes
    return out


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
           categories: list[str], buffer_m: float,
           *, poi_year=None, sub_year=None,
           city_nodes: int | None = None) -> list[PoiProfile]:
    """Asocia POIs a subgrafos y calcula el perfil de cada uno.

    Un POI puede caer en **varias** zonas y se cuenta en todas: las envolventes
    de meses distintos se solapan por construcción, y descontar el solape haría
    que el perfil de una zona dependiera de qué otras zonas existen.

    Con `poi_year` y `sub_year` la asociación pasa a ser **temporal**: cada
    subgrafo se caracteriza contra la instantánea de su propio año, en vez de
    describir un hotspot de 2018 con el mapa de 2026. El cociente de
    localización se calcula contra la ciudad de ese mismo año, y necesita
    `city_nodes` —los nodos de la red completa— como base; sin él no se calcula,
    porque la alternativa (usar el total de POIs) es justo la que **no**
    neutraliza las importaciones masivas de OSM.
    """
    if not subgraphs:
        return []

    if (poi_year is None) != (sub_year is None):
        raise ValueError("poi_year y sub_year van juntos o no van")

    # Mismo plano métrico que los subgrafos, no uno nuevo: los POIs se extienden
    # más allá de los nodos de la red y elegir zona por su cuenta daría dos
    # planos distintos, con las contenciones mal en los bordes.
    to_m = transformer_for(subgraphs[0].epsg)
    px, py = to_m(np.asarray(poi_lon, dtype=np.float64),
                  np.asarray(poi_lat, dtype=np.float64))

    cat_index = {c: i for i, c in enumerate(categories)}
    codes = np.array([cat_index.get(str(c), -1) for c in poi_cat], dtype=np.int64)

    points = [Point(x, y) for x, y in zip(np.asarray(px), np.asarray(py))]
    k = len(categories)
    zones = hulls(subgraphs, buffer_m)

    # Sin eje temporal hay un solo grupo, con todos los POIs. Con él, un árbol
    # por año: consultar el árbol global y filtrar después por año recorrería
    # ocho veces los mismos candidatos.
    if poi_year is None:
        groups = {None: np.arange(len(points), dtype=np.int64)}
        sub_of = [None] * len(subgraphs)
    else:
        py_arr = np.asarray(poi_year, dtype=np.int64)
        years = np.unique(py_arr)
        groups = {int(y): np.flatnonzero(py_arr == y) for y in years}
        # Un subgrafo de un año sin instantánea cae en la más cercana: es
        # preferible a dejarlo sin perfil, y se puede auditar por `.year`.
        sub_of = [int(years[np.argmin(np.abs(years - int(y)))]) for y in sub_year]

    trees = {y: STRtree([points[i] for i in idx]) for y, idx in groups.items()}
    city = {y: np.bincount(codes[idx][codes[idx] >= 0], minlength=k).astype(np.int64)
            for y, idx in groups.items()}

    profiles: list[PoiProfile] = []
    for sg, zone, y in zip(subgraphs, zones, sub_of):
        idx, tree = groups[y], trees[y]
        counts = np.zeros(k, dtype=np.int64)
        # El árbol filtra por bbox; `contains` decide de verdad.
        for j in tree.query(zone):
            i = int(idx[j])
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
            lq=(location_quotient(counts, city[y], sg.n, city_nodes)
                if city_nodes else None),
            year=y,
        ))
    return profiles
