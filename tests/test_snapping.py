"""Snapping arista→nodo y campo de densidad (§4.0, §4.1)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from pipeline.config import load_config
from pipeline.density import KernelCache, bounded_dijkstra, density_field, to_csr
from pipeline.snapping import haversine, snap_crimes

from .conftest import m_to_deg


@pytest.fixture(scope="module")
def cfg():
    return load_config()


class TestHaversine:
    def test_distancia_cero(self):
        assert haversine(np.array([41.88]), np.array([-87.63]),
                         np.array([41.88]), np.array([-87.63]))[0] == pytest.approx(0.0)

    def test_un_grado_de_latitud(self):
        d = haversine(np.array([0.0]), np.array([0.0]),
                      np.array([1.0]), np.array([0.0]))[0]
        assert d == pytest.approx(111_195.0, rel=1e-3)

    def test_coincide_con_la_conversion_local(self):
        """100 m en la fixture deben medir 100 m con haversine."""
        lat1, lon1 = m_to_deg(0.0, 0.0)
        lat2, lon2 = m_to_deg(100.0, 0.0)
        d = haversine(np.array([lat1]), np.array([lon1]),
                      np.array([lat2]), np.array([lon2]))[0]
        assert d == pytest.approx(100.0, rel=0.01)

    def test_es_simetrica(self):
        a = haversine(np.array([41.0]), np.array([-87.0]),
                      np.array([42.0]), np.array([-88.0]))
        b = haversine(np.array([42.0]), np.array([-88.0]),
                      np.array([41.0]), np.array([-87.0]))
        assert a[0] == pytest.approx(b[0])


class TestSnapping:
    def test_punto_entre_calles_paralelas_va_a_la_correcta(self, parallel_streets, cfg):
        """El caso que el snapping directo a nodo falla (§4.0, §11.2).

        El punto está a 10 m de la calle A y a 50 m de la calle B, pero el nodo
        *euclidianamente* más cercano es el 4, que está en la calle B. Ir al
        nodo más próximo lo asignaría a la calle equivocada; ir primero a la
        arista más próxima y luego a su extremo, no.
        """
        G = parallel_streets
        lat, lon = m_to_deg(100.0, 10.0)     # a mitad de cuadra de la calle A

        # El nodo más cercano está, en efecto, en la calle equivocada.
        d = {
            n: haversine(np.array([lat]), np.array([lon]),
                         np.array([G.nodes[n]["y"]]), np.array([G.nodes[n]["x"]]))[0]
            for n in G.nodes
        }
        assert min(d, key=d.get) == 4
        assert G.nodes[4]["street"] == "B"

        res = snap_crimes(G, np.array([lat]), np.array([lon]), cfg)
        assert G.nodes[int(res.node[0])]["street"] == "A"
        assert res.dist_edge_m[0] == pytest.approx(10.0, abs=2.0)

    def test_descarta_los_puntos_demasiado_lejos(self, parallel_streets, cfg):
        G = parallel_streets
        lat, lon = m_to_deg(100.0, 5000.0)      # 5 km al norte de la red
        res = snap_crimes(G, np.array([lat]), np.array([lon]), cfg)
        assert res.node[0] == -1
        assert res.dropped_far == 1
        assert not res.mask[0]

    def test_asigna_al_extremo_mas_cercano_de_su_arista(self, parallel_streets, cfg):
        G = parallel_streets
        lat, lon = m_to_deg(20.0, 2.0)          # cerca del nodo 1, sobre la calle A
        res = snap_crimes(G, np.array([lat]), np.array([lon]), cfg)
        assert int(res.node[0]) == 1


class TestDensidad:
    def test_dijkstra_respeta_la_red_no_la_linea_recta(self, grid_graph):
        """En un damero, la distancia geodésica es Manhattan, no euclidiana."""
        g = to_csr(grid_graph)
        ids = grid_graph.graph["ids"]
        src = g.index[ids[(0, 0)]]
        nodes, dists = bounded_dijkstra(g, src, cutoff=1000.0)
        d = dict(zip(nodes.tolist(), dists.tolist()))

        # (3,4): 3 cuadras arriba + 4 a la derecha = 700 m por la red;
        # en línea recta serían 500 m.
        assert d[g.index[ids[(3, 4)]]] == pytest.approx(700.0)
        assert math.hypot(300, 400) == pytest.approx(500.0)

    def test_el_truncamiento_acota_el_vecindario(self, grid_graph):
        g = to_csr(grid_graph)
        src = g.index[grid_graph.graph["ids"][(3, 3)]]
        nodes, dists = bounded_dijkstra(g, src, cutoff=250.0)
        assert dists.max() <= 250.0
        assert nodes.size < g.n

    def test_el_kernel_vale_uno_en_la_fuente_y_decae(self, grid_graph):
        g = to_csr(grid_graph)
        ids = grid_graph.graph["ids"]
        kern = KernelCache(g, sigma_m=120.0, radius_m=360.0)
        nodes, w = kern.get(g.index[ids[(3, 3)]])
        peso = dict(zip(nodes.tolist(), w.tolist()))

        assert peso[g.index[ids[(3, 3)]]] == pytest.approx(1.0)
        # A 100 m: exp(−100²/(2·120²))
        assert peso[g.index[ids[(3, 4)]]] == pytest.approx(
            math.exp(-(100.0 ** 2) / (2 * 120.0 ** 2))
        )
        assert peso[g.index[ids[(3, 4)]]] > peso[g.index[ids[(3, 5)]]]

    def test_el_campo_es_lineal_en_los_conteos(self, grid_graph):
        g = to_csr(grid_graph)
        ids = grid_graph.graph["ids"]
        kern = KernelCache(g, sigma_m=120.0, radius_m=360.0)
        src = np.array([g.index[ids[(2, 2)]]])
        f1 = density_field(kern, src, np.array([1.0]), g.n)
        f3 = density_field(kern, src, np.array([3.0]), g.n)
        assert np.allclose(f3, 3.0 * f1)

    def test_el_campo_es_maximo_en_la_fuente(self, grid_graph):
        g = to_csr(grid_graph)
        src_id = grid_graph.graph["ids"][(3, 3)]
        kern = KernelCache(g, sigma_m=120.0, radius_m=360.0)
        f = density_field(kern, np.array([g.index[src_id]]), np.array([5.0]), g.n)
        assert int(np.argmax(f)) == g.index[src_id]

    def test_el_cache_de_kernels_no_cambia_el_resultado(self, grid_graph):
        g = to_csr(grid_graph)
        ids = grid_graph.graph["ids"]
        src = np.array([g.index[ids[(1, 1)]], g.index[ids[(4, 4)]]])
        c = np.array([2.0, 3.0])
        k1 = KernelCache(g, 120.0, 360.0)
        f_a = density_field(k1, src, c, g.n)
        f_b = density_field(k1, src, c, g.n)      # segunda vez, ya cacheado
        assert np.array_equal(f_a, f_b)


class TestCSR:
    def test_conserva_nodos_y_aristas(self, grid_graph):
        g = to_csr(grid_graph)
        assert g.n == grid_graph.number_of_nodes()
        assert g.indices.size == 2 * grid_graph.number_of_edges()

    def test_la_adyacencia_coincide_con_networkx(self, grid_graph):
        g = to_csr(grid_graph)
        for nid in list(grid_graph.nodes)[:20]:
            i = g.index[nid]
            vecinos = {int(g.ids[g.indices[k]])
                       for k in range(g.indptr[i], g.indptr[i + 1])}
            assert vecinos == set(grid_graph.neighbors(nid))
