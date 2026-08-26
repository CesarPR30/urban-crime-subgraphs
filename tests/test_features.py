"""Fase E: caracterización, embedding y similitud (PROJECT_SPEC §5, §9).

El test que importa es `test_el_crimen_no_toca_la_similitud`. Los demás
comprueban propiedades; ese comprueba que el proyecto no es circular.
"""

from __future__ import annotations

import math

import networkx as nx
import numpy as np
import pytest

from pipeline.config import load_config
from pipeline.features.embedding import Graph2Vec, NetLSD, wl_documents
from pipeline.features.hierarchy import dims as hier_dims, fingerprint, hierarchy_matrix
from pipeline.features.structural import (
    GEOMETRY_DIMS,
    STRUCTURAL_DIMS,
    Subgraph,
    build_subgraphs,
    descriptor,
    structural_matrix,
)
from pipeline.similarity import (
    Blocks,
    cosine_matrix,
    fuse,
    jaccard_matrix,
    run,
    standardize,
    top_k,
)

M_PER_DEG = 111_320.0
LAT0, LON0 = 41.88, -87.63


def _ll(dx_m: float, dy_m: float) -> tuple[float, float]:
    lat = LAT0 + dy_m / M_PER_DEG
    lon = LON0 + dx_m / (M_PER_DEG * math.cos(math.radians(LAT0)))
    return lat, lon


@pytest.fixture
def city():
    """Red de juguete: un damero de 6×6 con cuadras de 100 m."""
    G = nx.MultiDiGraph()
    G.graph["crs"] = "EPSG:4326"
    ids = {}
    for i in range(6):
        for j in range(6):
            nid = 1000 + i * 6 + j
            lat, lon = _ll(j * 100.0, i * 100.0)
            G.add_node(nid, x=lon, y=lat)
            ids[(i, j)] = nid
    for i in range(6):
        for j in range(6):
            for di, dj, cls in ((0, 1, "street"), (1, 0, "avenue")):
                if i + di < 6 and j + dj < 6:
                    G.add_edge(ids[(i, j)], ids[(i + di, j + dj)],
                               length=100.0, road_class=cls)
    G.graph["ids"] = ids
    return G


def _hotspot(hid, month, ids, cells, crimes=100):
    """Un registro de hotspot con la forma del artefacto real."""
    nodes = [ids[c] for c in cells]
    sel = set(cells)
    edges = []
    for (i, j) in cells:
        for (di, dj) in ((0, 1), (1, 0)):
            if (i + di, j + dj) in sel:
                edges.append([ids[(i, j)], ids[(i + di, j + dj)]])
    return {
        "id": hid, "month": month, "nodes": nodes, "edges": edges,
        "crimes": crimes,
        "by_category": {"THEFT": crimes // 2, "ASSAULT": crimes - crimes // 2},
        "seed_node": nodes[0], "peak_node": nodes[0],
        "peak_f": 10.0, "persistence": 5.0,
        "n_nodes": len(nodes), "density": crimes / len(nodes),
    }


@pytest.fixture
def artifact(city):
    """Cuatro hotspots: una fila recta, una L, un bloque 2×2 y un damero 3×3."""
    ids = city.graph["ids"]
    # La recta y la L tienen los MISMOS 5 nodos y las mismas 4 aristas: las dos
    # son el camino `P₅`. Es la premisa del test que compara los dos bloques.
    recta = [(0, j) for j in range(5)]
    ele = [(0, 0), (0, 1), (0, 2), (1, 2), (2, 2)]
    bloque = [(2, 0), (2, 1), (3, 0), (3, 1)]
    damero = [(i, j) for i in range(3, 6) for j in range(3, 6)]
    return {
        "summary": {"top_k": 4},
        "monthly": [],
        "hotspots": [
            _hotspot("2024-01_h01", "2024-01", ids, recta, crimes=300),
            _hotspot("2024-01_h02", "2024-01", ids, ele, crimes=50),
            _hotspot("2024-02_h01", "2024-02", ids, bloque, crimes=10),
            _hotspot("2024-02_h02", "2024-02", ids, damero, crimes=999),
        ],
    }


# --------------------------------------------------------------------------- #
# §5.1 — la invariante del proyecto
# --------------------------------------------------------------------------- #


def test_el_crimen_no_toca_la_similitud(artifact, city):
    """Alterar los crímenes no puede mover ni un bit del vector de similitud.

    Es la comprobación de §5.1. Si esto falla, el proyecto se vuelve circular:
    estaría "descubriendo" que zonas con crimen parecido tienen crimen parecido.

    No se inspeccionan nombres de campos —eso se esquiva con un renombrado— sino
    el comportamiento: se retuercen todos los conteos del artefacto y se exige
    que la salida sea idéntica.
    """
    cfg = load_config()
    order = cfg.network.hierarchy_order

    def fused_for(art):
        subs = build_subgraphs(art, city)
        blocks = Blocks(
            structural=structural_matrix(subs),
            embedding=Graph2Vec(16, 2, 5, 42).fit_transform([s.graph() for s in subs]),
            hierarchy=hierarchy_matrix(subs, order),
        )
        return fuse(blocks, cfg.similarity.weights)

    base = fused_for(artifact)

    envenenado = {**artifact, "hotspots": []}
    for k, h in enumerate(artifact["hotspots"]):
        envenenado["hotspots"].append({
            **h,
            "crimes": 10 ** (k + 2),
            "by_category": {"THEFT": 10 ** k, "ROBBERY": 7 * k},
            "density": 1e6 / (k + 1),
            "peak_f": -k,
            "persistence": 999.0 * k,
        })

    assert np.array_equal(base, fused_for(envenenado)), (
        "el vector de similitud cambió al alterar los datos de crimen"
    )


def test_subgraph_no_lleva_datos_de_crimen(artifact, city):
    """Segunda barrera: quien tenga un `Subgraph` no puede contaminar aunque quiera."""
    subs = build_subgraphs(artifact, city)
    campos = set(Subgraph.__dataclass_fields__)
    prohibidos = {"crimes", "by_category", "density", "peak_f", "persistence",
                  "seed_node", "peak_node"}
    assert campos & prohibidos == set()


def test_nombres_de_dimensiones_sin_rastro_de_crimen():
    palabras = ("crim", "delito", "density_crime", "peak_f", "persistence")
    nombres = " ".join(STRUCTURAL_DIMS + hier_dims(["a", "b"])).lower()
    for p in palabras:
        assert p not in nombres


# --------------------------------------------------------------------------- #
# §5.2 — descriptor estructural
# --------------------------------------------------------------------------- #


def test_descriptor_tiene_doce_dims(artifact, city):
    subs = build_subgraphs(artifact, city)
    assert len(STRUCTURAL_DIMS) == 12
    assert descriptor(subs[0]).shape == (12,)


def test_normalizacion_min_max_en_cero_uno(artifact, city):
    X = structural_matrix(build_subgraphs(artifact, city))
    assert X.shape == (4, 12)
    assert X.min() >= 0.0 and X.max() <= 1.0


def test_la_recta_y_la_L_se_distinguen(artifact, city):
    """El caso que motiva el bloque métrico en §5.2.

    Una cadena recta y una en L son ambas el grafo camino `P_n`: misma
    conectividad, mismo histograma de grados. Solo la geometría las separa.
    """
    subs = {sg.id: sg for sg in build_subgraphs(artifact, city)}
    recta, ele = descriptor(subs["2024-01_h01"]), descriptor(subs["2024-01_h02"])
    i_elong = STRUCTURAL_DIMS.index("elongation")

    assert recta[i_elong] < 0.05, "una recta debe tener elongación ≈ 0"
    assert ele[i_elong] > 0.4, "una L no es lineal"

    # Y el bloque de conectividad por sí solo NO las distinguiría.
    n_conn = len(STRUCTURAL_DIMS) - len(GEOMETRY_DIMS)
    conn_recta, conn_ele = recta[:n_conn], ele[:n_conn]
    assert np.allclose(conn_recta[4:], conn_ele[4:], atol=0.02), (
        "los histogramas de grados de una recta y una L deberían coincidir"
    )


def test_hotspot_de_un_solo_nodo_no_revienta(city):
    ids = city.graph["ids"]
    art = {"hotspots": [_hotspot("2024-01_h01", "2024-01", ids, [(0, 0)])]}
    d = descriptor(build_subgraphs(art, city)[0])
    assert np.all(np.isfinite(d))
    assert d[STRUCTURAL_DIMS.index("density")] == 0.0


# --------------------------------------------------------------------------- #
# §5.4 — jerarquía vial
# --------------------------------------------------------------------------- #


def test_huella_de_jerarquia_suma_uno(artifact, city):
    order = ["expressway", "avenue", "collector", "street", "service"]
    subs = build_subgraphs(artifact, city)
    H = hierarchy_matrix(subs, order)
    assert H.shape == (4, len(order) + 1)
    for row in H:
        assert row[:-1].sum() == pytest.approx(1.0)
        assert row[-1] == pytest.approx(row[:-1].max())


def test_una_fila_de_calles_es_monoclase(artifact, city):
    order = ["expressway", "avenue", "collector", "street", "service"]
    subs = {sg.id: sg for sg in build_subgraphs(artifact, city)}
    h = fingerprint(subs["2024-01_h01"], order)      # la fila recta, toda `street`
    assert h[order.index("street")] == pytest.approx(1.0)
    assert h[-1] == pytest.approx(1.0)


def test_sin_aristas_la_huella_es_cero(city):
    ids = city.graph["ids"]
    art = {"hotspots": [_hotspot("x", "2024-01", ids, [(0, 0)])]}
    sg = build_subgraphs(art, city)[0]
    assert np.all(fingerprint(sg, ["street", "avenue"]) == 0.0)


# --------------------------------------------------------------------------- #
# §5.3 / §9 — embedding
# --------------------------------------------------------------------------- #


def _isomorfo(g, seed=11):
    perm = list(g.nodes)
    np.random.RandomState(seed).shuffle(perm)
    return nx.relabel_nodes(g, dict(zip(g.nodes, perm)), copy=True)


@pytest.fixture
def corpus():
    base = [nx.grid_2d_graph(3, 4), nx.path_graph(9), nx.cycle_graph(7),
            nx.barbell_graph(5, 2), nx.star_graph(6)]
    base = [nx.convert_node_labels_to_integers(g) for g in base]
    out = []
    for g in base:
        out.append(g)
        out.append(_isomorfo(g))     # cada grafo seguido de su copia renumerada
    return out


def test_wl_es_invariante_a_la_numeracion(corpus):
    """La propiedad que hace correcto a graph2vec: el `sort` borra los ids."""
    docs = wl_documents(corpus, 2)
    for i in range(0, len(corpus), 2):
        assert sorted(docs[i]) == sorted(docs[i + 1])


def test_isomorfos_reciben_el_mismo_embedding(corpus):
    """Criterio de aceptación de la fase E (§9).

    No sale idéntico bit a bit: PV-DBOW inicializa cada documento por separado y
    el descenso los visita en orden distinto, así que queda ruido de SGD. Lo que
    sí debe cumplirse es que el par isomorfo esté más junto que cualquier par no
    isomorfo, con margen.
    """
    E = Graph2Vec(64, 2, 50, 42).fit_transform(corpus)
    U = E / np.linalg.norm(E, axis=1, keepdims=True)
    S = U @ U.T

    iso = [S[i, i + 1] for i in range(0, len(corpus), 2)]
    distintos = [S[i, j] for i in range(len(corpus)) for j in range(len(corpus))
                 if i // 2 != j // 2]

    assert min(iso) > 0.999
    assert min(iso) > max(distintos)


def test_embedding_reproducible_con_el_mismo_seed(corpus):
    a = Graph2Vec(32, 2, 10, 42).fit_transform(corpus)
    b = Graph2Vec(32, 2, 10, 42).fit_transform(corpus)
    assert np.array_equal(a, b)


def test_seeds_distintos_dan_resultados_distintos(corpus):
    a = Graph2Vec(32, 2, 10, 42).fit_transform(corpus)
    b = Graph2Vec(32, 2, 10, 7).fit_transform(corpus)
    assert not np.array_equal(a, b)


def test_netlsd_es_determinista_y_finito(corpus):
    a = NetLSD(8).fit_transform(corpus)
    b = NetLSD(8).fit_transform(corpus)
    assert a.shape == (len(corpus), 8)
    assert np.all(np.isfinite(a))
    assert np.array_equal(a, b)


def test_fallback_a_netlsd_cuando_falta_la_dependencia(caplog):
    """§5.3: nunca debe fallar la generación de coordenadas."""
    from pipeline.features.embedding import build_embedder

    cfg = load_config().model_copy(deep=True)
    cfg.features.embedding.method = "gcn"
    with caplog.at_level("WARNING"):
        emb = build_embedder(cfg)
    assert emb.name == "netlsd"
    assert "fallback" in caplog.text.lower()


# --------------------------------------------------------------------------- #
# §5.5 — fusión, coseno, top-K
# --------------------------------------------------------------------------- #


def test_standardize_no_produce_nan_en_columnas_constantes():
    X = np.array([[1.0, 5.0], [2.0, 5.0], [3.0, 5.0]])
    Z = standardize(X)
    assert np.all(np.isfinite(Z))
    assert np.all(Z[:, 1] == 0.0)


def test_los_pesos_reparten_norma_no_columnas():
    """Un bloque de 128 dims no debe pesar 10× uno de 12 solo por ser más ancho."""
    rng = np.random.RandomState(0)
    blocks = Blocks(
        structural=rng.rand(20, 12),
        embedding=rng.rand(20, 128),
        hierarchy=rng.rand(20, 6),
    )

    class W:
        structural, embedding, hierarchy = 1.0, 1.0, 1.0

    F = fuse(blocks, W)
    a = np.linalg.norm(F[:, :12], axis=1).mean()
    b = np.linalg.norm(F[:, 12:140], axis=1).mean()
    assert 0.7 < a / b < 1.4


def test_coseno_es_simetrico_y_acotado():
    rng = np.random.RandomState(1)
    S = cosine_matrix(rng.randn(30, 8))
    assert np.allclose(S, S.T)
    assert np.allclose(np.diag(S), 1.0)
    assert S.min() >= -1.0 and S.max() <= 1.0


def test_coseno_tolera_el_vector_cero():
    S = cosine_matrix(np.array([[0.0, 0.0], [1.0, 0.0]]))
    assert np.all(np.isfinite(S))


def test_jaccard_mide_solape_de_huella():
    J = jaccard_matrix([{1, 2, 3}, {2, 3, 4}, {9}])
    assert J[0, 1] == pytest.approx(2 / 4)
    assert J[0, 2] == 0.0
    assert np.allclose(np.diag(J), 1.0)


def test_top_k_ordena_excluye_al_propio_y_respeta_k():
    ids = ["a", "b", "c", "d"]
    months = ["2024-01"] * 4
    S = np.array([
        [1.0, 0.9, 0.5, 0.7],
        [0.9, 1.0, 0.2, 0.1],
        [0.5, 0.2, 1.0, 0.3],
        [0.7, 0.1, 0.3, 1.0],
    ])
    J = np.eye(4, dtype=np.float32)
    top = top_k(S, ids, months, 2, J, exclude_same_place=False)
    assert [s["id"] for s in top["a"]] == ["b", "d"]
    assert all(len(v) == 2 for v in top.values())
    assert all(s["id"] != k for k, v in top.items() for s in v)


def test_top_k_distinto_excluye_el_mismo_lugar():
    ids = ["a", "b", "c"]
    months = ["2024-01", "2024-02", "2024-03"]
    S = np.array([[1.0, 0.99, 0.4], [0.99, 1.0, 0.3], [0.4, 0.3, 1.0]])
    J = np.array([[1.0, 0.8, 0.0], [0.8, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)

    con = top_k(S, ids, months, 2, J, exclude_same_place=False)
    sin = top_k(S, ids, months, 2, J, exclude_same_place=True)
    assert con["a"][0]["id"] == "b"          # el más parecido es el mismo sitio
    assert [s["id"] for s in sin["a"]] == ["c"]


def test_similitud_completa_sobre_el_corpus_de_juguete(artifact, city):
    cfg = load_config()
    subs = build_subgraphs(artifact, city)
    blocks = Blocks(
        structural=structural_matrix(subs),
        embedding=Graph2Vec(16, 2, 5, 42).fit_transform([s.graph() for s in subs]),
        hierarchy=hierarchy_matrix(subs, cfg.network.hierarchy_order),
    )
    res = run(subs, blocks, cfg)

    assert res.coords.shape == (4, 2)
    assert np.all(np.isfinite(res.coords))
    assert res.labels.shape == (4,)
    assert set(res.top) == {sg.id for sg in subs}
    assert res.meta["fused_dims"] == 12 + 16 + len(cfg.network.hierarchy_order) + 1
