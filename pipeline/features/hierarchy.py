"""Huella de jerarquía vial (PROJECT_SPEC §5.4).

Distribución normalizada de las aristas del subgrafo sobre las clases viales,
más la participación de la clase dominante.

Separa dos subgrafos con la misma forma pero distinta materia: un damero de
calles residenciales y un damero de arterias tienen idéntico descriptor
estructural —misma conectividad, misma geometría— y son cosas muy distintas.
"""

from __future__ import annotations

import numpy as np

from .structural import Subgraph


def dims(order: list[str]) -> tuple[str, ...]:
    """Nombres de las columnas para una jerarquía dada."""
    return tuple(f"frac_{c}" for c in order) + ("dominant_share",)


def fingerprint(sg: Subgraph, order: list[str]) -> np.ndarray:
    """Vector de `len(order) + 1` dims para un subgrafo.

    Un hotspot sin aristas (un solo nodo) no tiene jerarquía que describir: sale
    el vector cero, que es honesto —no "todo residencial"—.
    """
    counts = np.zeros(len(order), dtype=np.float64)
    if not sg.edge_class:
        return np.append(counts, 0.0)

    index = {c: i for i, c in enumerate(order)}
    for cls in sg.edge_class:
        i = index.get(cls)
        if i is not None:
            counts[i] += 1.0

    total = counts.sum()
    if total <= 0:                      # clases fuera de la jerarquía del config
        return np.append(counts, 0.0)
    frac = counts / total
    return np.append(frac, float(frac.max()))


def hierarchy_matrix(subgraphs: list[Subgraph], order: list[str]) -> np.ndarray:
    """Matriz `(n_hotspots, len(order) + 1)`.

    No se normaliza aquí: las fracciones ya viven en [0,1] y suman 1, y
    estandarizarlas es trabajo del paso de fusión (§5.5).
    """
    return np.vstack([fingerprint(sg, order) for sg in subgraphs])
