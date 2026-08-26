"""Fixtures sintéticas: una grilla vial pequeña y un CSV mínimo de crímenes."""

from __future__ import annotations

import math
import random

import networkx as nx
import numpy as np
import pytest

# Un grado de latitud ≈ 111.32 km. La grilla se define en metros y se convierte
# a lat/lon para que las fixtures vivan en el mismo marco que los datos reales.
M_PER_DEG_LAT = 111_320.0
LAT0, LON0 = 41.88, -87.63          # centro de Chicago, para que el UTM sea el real


def m_to_deg(dx_m: float, dy_m: float) -> tuple[float, float]:
    lat = LAT0 + dy_m / M_PER_DEG_LAT
    lon = LON0 + dx_m / (M_PER_DEG_LAT * math.cos(math.radians(LAT0)))
    return lat, lon


@pytest.fixture
def grid_graph() -> nx.Graph:
    """Damero de 7×7 con cuadras de 100 m, no dirigido y con `length`."""
    n, step = 7, 100.0
    G = nx.Graph()
    ids = {}
    for i in range(n):
        for j in range(n):
            nid = i * n + j
            lat, lon = m_to_deg(j * step, i * step)
            G.add_node(nid, x=lon, y=lat)
            ids[(i, j)] = nid
    for i in range(n):
        for j in range(n):
            if j + 1 < n:
                G.add_edge(ids[(i, j)], ids[(i, j + 1)], length=step, road_class="street")
            if i + 1 < n:
                G.add_edge(ids[(i, j)], ids[(i + 1, j)], length=step, road_class="street")
    G.graph["ids"] = ids
    G.graph["step"] = step
    return G


@pytest.fixture
def parallel_streets() -> nx.MultiDiGraph:
    """Dos calles paralelas este-oeste separadas 60 m, con nodos desalineados.

    Diseñada para que el nodo euclidianamente más cercano a un punto esté en la
    calle *equivocada*: es el caso que el snapping arista→nodo debe resolver
    bien y el snapping directo a nodo falla (§4.0, §11.2).
    """
    G = nx.MultiDiGraph()
    G.graph["crs"] = "EPSG:4326"

    # Calle A (y = 0): nodos en x = 0 y 200 -> tramo largo sin nodos intermedios.
    for nid, x in ((1, 0.0), (2, 200.0)):
        lat, lon = m_to_deg(x, 0.0)
        G.add_node(nid, x=lon, y=lat, street="A")
    # Calle B (y = 60): nodo justo enfrente del punto de prueba.
    for nid, x in ((3, 0.0), (4, 100.0), (5, 200.0)):
        lat, lon = m_to_deg(x, 60.0)
        G.add_node(nid, x=lon, y=lat, street="B")

    G.add_edge(1, 2, length=200.0, highway="residential")
    G.add_edge(3, 4, length=100.0, highway="residential")
    G.add_edge(4, 5, length=100.0, highway="residential")
    return G


@pytest.fixture
def crimes_csv(tmp_path):
    """CSV de ~200 crímenes con seed fijo, más filas inválidas inyectadas.

    Devuelve `(ruta, esperado)` con el desglose exacto de lo que el
    `ValidationReport` debe encontrar.
    """
    rng = random.Random(20240101)
    # Columnas de sobra a propósito (`Beat`, `Ward`, `FBI Code`, `Arrest`): el
    # cargador debe proyectar al esquema canónico y descartarlas.
    rows = ["ID,Case Number,Date,Beat,Ward,FBI Code,Arrest,Primary Type,"
            "Description,Location Description,Latitude,Longitude"]
    n_valid = 200
    for i in range(n_valid):
        lat, lon = m_to_deg(rng.uniform(0, 600), rng.uniform(0, 600))
        month = rng.choice(["01", "02"])
        day = f"{rng.randint(1, 28):02d}"
        tipo = rng.choice(["THEFT", "ASSAULT", "ROBBERY", "MOTOR VEHICLE THEFT"])
        rows.append(
            f"{i},JX{i:05d},{month}/{day}/2024 10:00:00 AM,1234,42,06,false,"
            f"{tipo},SIMPLE,STREET,{lat:.8f},{lon:.8f}"
        )

    bad = {
        "lat_invalida": 3,
        "lon_invalida": 2,
        "fecha_invalida": 2,
        "crimen_vacio": 1,
        "tipo_vacio": 1,
    }
    uid = n_valid

    def row(fecha="01/05/2024 10:00:00 AM", tipo="THEFT", desc="SIMPLE",
            lat="41.88", lon="-87.63"):
        nonlocal uid
        uid += 1
        return (f"{uid},JX9{uid:04d},{fecha},1234,42,06,false,"
                f"{tipo},{desc},STREET,{lat},{lon}")

    for _ in range(bad["lat_invalida"]):
        rows.append(row(lat=""))
    for _ in range(bad["lon_invalida"]):
        rows.append(row(lon="abc"))
    for _ in range(bad["fecha_invalida"]):
        rows.append(row(fecha="no-es-fecha"))
    for _ in range(bad["crimen_vacio"]):
        rows.append(row(desc=""))
    for _ in range(bad["tipo_vacio"]):
        rows.append(row(tipo=""))

    # Un duplicado exacto de la primera fila válida, con ID repetido.
    rows.append(rows[1])

    path = tmp_path / "crimes.csv"
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return path, {"valid": n_valid, "bad": bad, "duplicates": 1}


@pytest.fixture
def peaked_field(grid_graph):
    """Campo con dos picos claros y un tercero de baja persistencia.

    Devuelve `(csr, f, counts)` listos para `hotspots.segment`.
    """
    from pipeline.density import to_csr

    g = to_csr(grid_graph)
    ids = grid_graph.graph["ids"]
    f = np.full(g.n, 0.05, dtype=np.float64)
    counts = np.zeros(g.n, dtype=np.float64)

    def bump(cell, height, spread=1):
        ci, cj = cell
        for i in range(7):
            for j in range(7):
                d = max(abs(i - ci), abs(j - cj))
                if d <= spread:
                    f[g.index[ids[(i, j)]]] += height * (1.0 - 0.35 * d)

    bump((1, 1), 1.00)          # pico alto, esquina inferior izquierda
    bump((5, 5), 0.80)          # pico alto, esquina superior derecha
    bump((1, 3), 0.30)          # hombro del primero: poca persistencia

    counts[g.index[ids[(1, 1)]]] = 10
    counts[g.index[ids[(5, 5)]]] = 8
    counts[g.index[ids[(1, 3)]]] = 3
    return g, f, counts
