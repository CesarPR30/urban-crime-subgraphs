"""Perfiles de fuente: lo que cada proveedor tiene de particular (§3.1).

`crimes.py` sabe proyectar *cualquier* CSV al esquema canónico resolviendo
alias de columna. Eso basta mientras el proveedor solo difiera en cómo llama a
las cosas. No basta cuando difiere en **qué hay que hacer con las filas**.

Un perfil recoge exactamente eso y nada más:

``columns``
    Anclaje explícito de columna canónica → nombre real, para cuando los alias
    no bastan o resolverían mal. El CSV de la PNP trae `tipo_hecho`,
    `subtipo_hecho` *y* `modalidad_hecho`: los tres son «tipo» para el
    resolutor de alias, y el que gane decide el análisis entero. Se fija a mano.

``uid_columns``
    Identificador compuesto. Hay fuentes donde ninguna columna sola identifica
    el hecho.

``drop_where``
    Exclusión por valor de una columna del CSV *de origen* (no del esquema
    canónico): filas que la fuente marca como no fiables. Cuenta aparte de los
    descartes por calidad y de los filtros de alcance, porque no es ni una cosa
    ni la otra: es una decisión sobre la fuente.

``fallback_points``
    Detector de centroides de relleno. Es la única regla que no puede decidirse
    fila a fila —necesita ver el archivo entero— y por eso se aplica en una
    segunda pasada.

Un perfil no contiene umbrales: los umbrales viven en `config.yaml`, porque son
parámetros del análisis y tienen que poder barrerse. El perfil solo dice *qué*
columnas mirar.
"""

from __future__ import annotations

import re
import unicodedata
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Iterable, Sequence


@dataclass(frozen=True)
class SourceProfile:
    """Las particularidades de un proveedor de datos."""

    name: str

    #: canónica -> nombre exacto de la columna en el CSV. Gana sobre los alias.
    columns: dict[str, str] = field(default_factory=dict)

    #: Columnas del CSV que, concatenadas, identifican el hecho. Vacío = usar
    #: la columna `id` resuelta por alias.
    uid_columns: tuple[str, ...] = ()

    #: columna de origen -> valores que descalifican la fila.
    drop_where: dict[str, frozenset[str]] = field(default_factory=dict)

    #: columna de origen -> valores admitidos. Lo que no esté, se excluye.
    #:
    #: Es el complemento de `drop_where` y existe para separar dos cosas que en
    #: Lima no coinciden: **el alcance** del estudio y **el eje de categorías**.
    #: El alcance se decide sobre `tipo_hecho` (cinco familias de delito que
    #: ocurren en la vía pública); el eje que se ve en el mapa es
    #: `subtipo_hecho` (hurto, robo, extorsión, lesiones, homicidio…). Sin esta
    #: separación habría que elegir: o filtrar por la columna gruesa y mostrar
    #: cinco cajones, o enumerar a mano los treinta y tantos subtipos en el
    #: filtro de categorías y que la lista se desactualice sola.
    keep_where: dict[str, frozenset[str]] = field(default_factory=dict)

    #: Columna con la dirección escrita. La usa el detector de centroides para
    #: contar cuántas direcciones distintas comparten una misma coordenada.
    address_column: str | None = None

    #: Desfase horario de la ciudad respecto a UTC, en horas, cuando la fuente
    #: publica la fecha en UTC. `None` = la fecha ya viene en hora local.
    #:
    #: No es cosmético. El mes es la unidad de agregación del pipeline y la
    #: hora alimenta el perfil temporal del subgrafo: leer en UTC un hecho de
    #: Lima ocurrido a las 21:00 lo mueve al día siguiente y, si cae el último
    #: día del mes, al mes siguiente.
    local_utc_offset_h: float | None = None

    def uid(self, row: dict[str, str | None], fallback: str | None) -> str | None:
        """Identificador del hecho para deduplicar."""
        if not self.uid_columns:
            return fallback
        return "|".join((row.get(c) or "").strip() for c in self.uid_columns)

    def rejects(self, row: dict[str, str | None]) -> str | None:
        """Columna que descalifica la fila, o `None` si pasa."""
        for col, bad in self.drop_where.items():
            if (row.get(col) or "").strip().upper() in bad:
                return col
        for col, good in self.keep_where.items():
            if (row.get(col) or "").strip().upper() not in good:
                return col
        return None


# --------------------------------------------------------------------------- #
# Detector de centroides de relleno
# --------------------------------------------------------------------------- #


def normalize_address(raw: str) -> str:
    """Dirección comparable: sin acentos, sin puntuación, sin caja.

    `"AV. TÚPAC AMARU"`, `"av tupac amaru"` y `"Av Túpac Amaru."` cuentan como
    una sola dirección. Sin esto el detector inflaría el número de direcciones
    distintas por punto y marcaría como relleno esquinas que no lo son.
    """
    s = unicodedata.normalize("NFKD", raw)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    return re.sub(r"[^a-z0-9]+", " ", s.lower()).strip()


def find_fallback_points(
    coords: Sequence[tuple[str, str]],
    addresses: Iterable[str],
    *,
    min_rows: int,
    min_addresses: int,
) -> set[tuple[str, str]]:
    """Coordenadas que el geocodificador usa como cajón de sastre.

    Un punto se marca cuando cumple **las dos** condiciones: acumula al menos
    `min_rows` hechos y esos hechos vienen de al menos `min_addresses`
    direcciones escritas distintas.

    Exigir las dos es lo que separa el artefacto del fenómeno. Solo por volumen
    se tiraría la esquina genuinamente caliente —un mercado, un paradero— que
    es justo lo que el análisis busca. Solo por variedad de direcciones se
    tiraría cualquier manzana geocodificada a nivel de cuadra, que es la
    precisión normal de la fuente y perfectamente utilizable. El cajón de
    sastre es el único que cumple ambas: cientos de hechos que además vienen de
    cientos de direcciones que no tienen nada que ver entre sí.

    Se compara la coordenada como **texto crudo del CSV**, no como float: lo
    que se busca es la repetición exacta del mismo literal, que es la firma de
    un valor inyectado por el geocodificador y no medido.
    """
    rows: dict[tuple[str, str], int] = defaultdict(int)
    seen: dict[tuple[str, str], set[str]] = defaultdict(set)
    for pt, addr in zip(coords, addresses):
        rows[pt] += 1
        # Solo se acumulan direcciones mientras el punto siga siendo candidato:
        # el conjunto de un centroide de relleno tiene decenas de miles de
        # entradas y no hace falta guardarlas todas para decidir.
        if len(seen[pt]) <= min_addresses:
            seen[pt].add(normalize_address(addr))
    return {
        pt for pt, n in rows.items()
        if n >= min_rows and len(seen[pt]) >= min_addresses
    }


# --------------------------------------------------------------------------- #
# Perfiles conocidos
# --------------------------------------------------------------------------- #

#: Sin particularidades: todo se resuelve por alias. Es lo que había antes de
#: existir este módulo y sigue siendo el comportamiento por defecto.
GENERIC = SourceProfile(name="generic")

#: Chicago. El portal ya venía resolviéndose bien por alias; el perfil solo
#: fija el anclaje para que un cambio futuro en `COLUMN_ALIASES` no lo mueva.
CHICAGO = SourceProfile(
    name="chicago",
    columns={"tipo": "Primary Type", "crimen": "Description"},
)

#: Lima Metropolitana (PNP / DGIS).
#:
#: Las cinco decisiones:
#:
#: 1. **`tipo` = `subtipo_hecho`, `crimen` = `modalidad_hecho`.** La taxonomía
#:    de la PNP tiene cuatro niveles (materia -> tipo -> subtipo -> modalidad).
#:    El nivel que corresponde al `Primary Type` de Chicago —THEFT, ROBBERY,
#:    ASSAULT— **no** es `tipo_hecho` sino `subtipo_hecho`: `tipo_hecho` tiene
#:    52 valores tan gruesos que PATRIMONIO (DELITO) mete en la misma caja
#:    hurto, robo, extorsión y estafa, que en Chicago serían categorías
#:    distintas. `subtipo_hecho` distingue HURTO de ROBO de EXTORSION, que es
#:    la granularidad a la que el análisis dice algo. `modalidad_hecho` queda
#:    como descripción del hecho, equivalente al `Description` de Chicago.
#:
#: 2. **El alcance se decide sobre `tipo_hecho`, no sobre el eje de
#:    categorías.** `keep_where` retiene las cinco familias cuyo hecho ocurre
#:    en la vía pública; dentro de ellas entran **todos** sus subtipos, sin
#:    lista escrita a mano que se desactualice. Ver `keep_where` arriba.
#:
#: 3. **La fecha es `fecha_hora_hecho_iso_utc`.** El CSV trae también
#:    `fecha_hora_hecho` como epoch en milisegundos y `año/mes/dia_hecho`
#:    desglosados. La columna ISO es la única que no hay que reconstruir, y
#:    distingue el hecho del registro (`fecha_hora_registro_hecho`), que es una
#:    fecha administrativa y llega a estar meses por detrás.
#:
#: 4. **`lugar` = `distrito_hecho`.** Es el equivalente funcional del
#:    `Location Description` de Chicago para agregar y filtrar en el mapa.
#:    `direccion_hecho` es texto libre sin normalizar y no sirve para agrupar.
#:
#: 5. **El uid es compuesto.** `id_dgc` identifica la *denuncia*, no el hecho:
#:    el ETL emite una fila por `objectid` y repite la denuncia hasta nueve
#:    veces. Añadir subtipo y modalidad colapsa esa repetición sin fundir dos
#:    hechos distintos registrados bajo la misma denuncia.

#: Las cinco familias de `tipo_hecho` que se analizan. Quedan fuera, entre
#: otras, INTERVENCION POLICIALES (actividad policial, no delito, y solo 5.8 %
#: geocodificada) y LEY DE VIOLENCIA CONTRA LA MUJER… (ocurre en el domicilio y
#: se geocodifica a la vivienda de la víctima). Ver README.
LIMA_TIPOS = frozenset({
    "PATRIMONIO (DELITO)",
    "FALTAS",
    "SEGURIDAD PUBLICA (DELITO)",
    "VIDA, EL CUERPO Y LA SALUD (DELITO)",
    "LIBERTAD (DELITO)",
})

LIMA = SourceProfile(
    name="lima",
    columns={
        "lat": "lat_hecho",
        "lon": "long_hecho",
        "fecha": "fecha_hora_hecho_iso_utc",
        "tipo": "subtipo_hecho",
        "crimen": "modalidad_hecho",
        "lugar": "distrito_hecho",
        "id": "id_dgc",
    },
    uid_columns=("id_dgc", "subtipo_hecho", "modalidad_hecho"),
    drop_where={"observacion": frozenset({"GEO FORZADA AL CENTROIDE DE COMISARIA"})},
    keep_where={"tipo_hecho": LIMA_TIPOS},
    address_column="direccion_hecho",
    # Perú no observa horario de verano desde 1994, así que en toda la ventana
    # del archivo (2018-2025) el desfase es exactamente -5 h y no hay que
    # arrastrar una base de datos de zonas horarias para saberlo.
    local_utc_offset_h=-5.0,
)

PROFILES: dict[str, SourceProfile] = {p.name: p for p in (GENERIC, CHICAGO, LIMA)}


def get_profile(name: str | None) -> SourceProfile:
    """Perfil por nombre. Un nombre desconocido es un error de config, no un
    caso a tolerar en silencio: el análisis entero depende de esta elección."""
    if not name:
        return GENERIC
    try:
        return PROFILES[name]
    except KeyError:
        raise ValueError(
            f"source_profile {name!r} desconocido. Disponibles: "
            f"{', '.join(sorted(PROFILES))}"
        ) from None
