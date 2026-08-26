"""Caracterización de subgrafos (PROJECT_SPEC §5).

La restricción que gobierna este paquete está en §5.1 y no admite excepciones:

    Lo que entra en la similitud describe **forma**, nunca crimen.

`StructuralDescriptor`, `GraphEmbedding` y `RoadHierarchyFingerprint` se calculan
sin mirar un solo conteo de crímenes. El crimen vive en `CrimeProfile`, que solo
se usa para reportar y para el contraste final de la fase 3. Si las dos cosas se
mezclan, el resultado del proyecto se vuelve circular: se estaría "descubriendo"
que zonas con crimen parecido tienen crimen parecido.

`tests/test_features.py` falla si un campo derivado de crimen aparece en el
vector de similitud.
"""

from __future__ import annotations

from .structural import (
    CONNECTIVITY_DIMS,
    GEOMETRY_DIMS,
    STRUCTURAL_DIMS,
    Subgraph,
    build_subgraphs,
    structural_matrix,
)

__all__ = [
    "CONNECTIVITY_DIMS",
    "GEOMETRY_DIMS",
    "STRUCTURAL_DIMS",
    "Subgraph",
    "build_subgraphs",
    "structural_matrix",
]
