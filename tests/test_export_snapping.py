"""Artefacto binario del snapping para el dashboard.

Lo que se prueba aquí es lo que el navegador da por supuesto al leer
`snapping.bin`: el layout, las alineaciones que exigen los `TypedArray`, la
descuantización de las coordenadas y la huella que impide combinar dos
artefactos de cargas distintas.
"""

from __future__ import annotations

import numpy as np
import pytest

from pipeline.export import NO_NODE, QUANT_MAX, snapping_binary
from pipeline.ingest.crimes import fingerprint, load_crimes


@pytest.fixture
def snap_sample():
    """Tres nodos, cinco crímenes, uno de ellos sin snappear."""
    node_ids = [101, 202, 303]
    node_lat = [41.80, 41.90, 42.00]
    node_lon = [-87.70, -87.60, -87.50]
    crime_node = np.array([101, 303, -1, 101, 202], dtype=np.int64)
    buf, meta = snapping_binary(
        node_ids, node_lat, node_lon, crime_node,
        fingerprint="deadbeef", n_crimes=5,
    )
    return buf, meta, node_lat, node_lon


def views(buf, meta):
    """Reproduce exactamente lo que hace `snapViews` en el dashboard."""
    n, N = meta["n"], meta["nodes"]
    return {
        "node": np.frombuffer(buf, dtype="<u4", count=n, offset=0),
        "lat": np.frombuffer(buf, dtype="<u2", count=N, offset=4 * n),
        "lon": np.frombuffer(buf, dtype="<u2", count=N, offset=4 * n + 2 * N),
    }


def test_layout_declarado_coincide_con_el_real(snap_sample):
    buf, meta, _, _ = snap_sample
    n, N = meta["n"], meta["nodes"]
    assert len(buf) == 4 * n + 2 * N + 2 * N
    assert [l["offset"] for l in meta["layout"]] == [0, 4 * n, 4 * n + 2 * N]
    assert [l["type"] for l in meta["layout"]] == ["Uint32", "Uint16", "Uint16"]


def test_offsets_alineados_para_typedarray(snap_sample):
    """`new Uint16Array(buffer, offset, n)` lanza si el offset no es par."""
    _, meta, _, _ = snap_sample
    for layout in meta["layout"]:
        width = 4 if layout["type"] == "Uint32" else 2
        assert layout["offset"] % width == 0, layout


def test_asignacion_crimen_nodo(snap_sample):
    buf, meta, _, _ = snap_sample
    v = views(buf, meta)
    # ids OSM [101, 303, -1, 101, 202] -> índices densos [0, 2, NONE, 0, 1]
    assert list(v["node"]) == [0, 2, NO_NODE, 0, 1]
    assert meta["snapped"] == 4
    assert meta["unsnapped"] == 1
    assert meta["nodes_touched"] == 3


def test_coordenadas_recuperables(snap_sample):
    """La cuantización tiene que sobrevivir al viaje de ida y vuelta."""
    buf, meta, node_lat, node_lon = snap_sample
    v = views(buf, meta)
    lat0, lon0, lat1, lon1 = meta["bbox"]
    lat = lat0 + v["lat"] * (lat1 - lat0) / QUANT_MAX
    lon = lon0 + v["lon"] * (lon1 - lon0) / QUANT_MAX
    # A escala de ciudad, 65 536 niveles son submilimétricos.
    assert np.allclose(lat, node_lat, atol=1e-6)
    assert np.allclose(lon, node_lon, atol=1e-6)


def test_agregacion_por_nodo_como_el_navegador(snap_sample):
    """La suma por nodo del dashboard debe dar la carga real."""
    buf, meta, _, _ = snap_sample
    v = views(buf, meta)
    ok = v["node"] != NO_NODE
    counts = np.bincount(v["node"][ok], minlength=meta["nodes"])
    assert list(counts) == [2, 1, 1]
    assert counts.sum() == meta["snapped"]


def test_rechaza_longitud_incoherente():
    """Si el vector de snapping no mide lo mismo que la carga, es otra corrida."""
    with pytest.raises(ValueError, match="no son la misma corrida"):
        snapping_binary(
            [1], [41.8], [-87.6], np.array([1, 1], dtype=np.int64),
            fingerprint="x", n_crimes=5,
        )


def test_un_solo_nodo_no_divide_por_cero():
    """Con un nodo el bbox es degenerado: `dlat = 0` no debe reventar."""
    buf, meta = snapping_binary(
        [7], [41.88], [-87.63], np.array([7, 7], dtype=np.int64),
        fingerprint="x", n_crimes=2,
    )
    assert meta["snapped"] == 2
    assert len(buf) == 4 * 2 + 2 + 2


# --------------------------------------------------------------------------- #
# Huella de la carga
# --------------------------------------------------------------------------- #


def test_huella_estable_entre_cargas_iguales(crimes_csv):
    """Dos lecturas del mismo CSV con los mismos filtros dan la misma huella."""
    path, _ = crimes_csv
    a, _ = load_crimes(path)
    b, _ = load_crimes(path)
    assert fingerprint(a) == fingerprint(b)


def test_huella_cambia_con_el_alcance(crimes_csv):
    """Es justo el caso que tiene que detectar: mismo CSV, distinto recorte."""
    path, _ = crimes_csv
    todos, _ = load_crimes(path)
    robos, _ = load_crimes(path, categories=("THEFT",))
    assert len(robos) < len(todos)
    assert fingerprint(todos) != fingerprint(robos)


def test_huella_de_secuencia_vacia_no_revienta():
    assert isinstance(fingerprint([]), str)
