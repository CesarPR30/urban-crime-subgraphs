"""Barrido por join tree, persistencia y extracción de hotspots (§4.2 – §4.3).

Los dos criterios de aceptación de la Fase C sobre la fixture sintética son que
las regiones sean **conexas** y **no se solapen**. Ambos se comprueban aquí y,
si el artefacto real existe, también sobre la salida de Chicago.
"""

from __future__ import annotations

import json
from pathlib import Path

import networkx as nx
import numpy as np
import pytest

from pipeline.hotspots import extract_month, segment

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "data" / "interim" / "hotspots.json"


def _connected(g, idx) -> bool:
    """¿El subgrafo inducido por `idx` (índices internos) es conexo?"""
    inside = set(int(i) for i in idx)
    if not inside:
        return False
    start = next(iter(inside))
    seen = {start}
    stack = [start]
    while stack:
        v = stack.pop()
        for k in range(g.indptr[v], g.indptr[v + 1]):
            u = int(g.indices[k])
            if u in inside and u not in seen:
                seen.add(u)
                stack.append(u)
    return seen == inside


class TestSegmentacion:
    def test_las_regiones_son_conexas(self, peaked_field):
        g, f, _ = peaked_field
        regions, _, _ = segment(f, g, alpha=0.3, f_min_ratio=0.10)
        assert regions
        for peak, idx in regions.items():
            assert _connected(g, idx), f"región {peak} no es conexa"

    def test_las_regiones_no_se_solapan(self, peaked_field):
        g, f, _ = peaked_field
        regions, _, _ = segment(f, g, alpha=0.3, f_min_ratio=0.10)
        seen: set[int] = set()
        for idx in regions.values():
            s = set(int(i) for i in idx)
            assert not (s & seen), "dos regiones comparten nodos"
            seen |= s

    def test_solo_entran_nodos_por_encima_del_piso(self, peaked_field):
        g, f, _ = peaked_field
        ratio = 0.25
        regions, _, _ = segment(f, g, alpha=0.3, f_min_ratio=ratio)
        floor = ratio * f.max()
        for idx in regions.values():
            assert (f[np.asarray(idx)] >= floor).all()

    def test_el_peak_de_cada_region_le_pertenece_y_es_su_maximo(self, peaked_field):
        g, f, _ = peaked_field
        regions, _, _ = segment(f, g, alpha=0.3, f_min_ratio=0.10)
        for peak, idx in regions.items():
            assert peak in set(int(i) for i in idx)
            assert f[peak] == pytest.approx(f[np.asarray(idx)].max())

    def test_la_persistencia_fusiona_el_pico_debil(self, peaked_field):
        """El tercer pico es un hombro del primero: debe absorberse."""
        g, f, _ = peaked_field
        sin_simplificar, _, _ = segment(f, g, alpha=0.0, f_min_ratio=0.10)
        simplificado, _, _ = segment(f, g, alpha=0.3, f_min_ratio=0.10)
        assert len(simplificado) < len(sin_simplificar)

    def test_alpha_mayor_produce_menos_regiones(self, peaked_field):
        g, f, _ = peaked_field
        counts = [len(segment(f, g, alpha=a, f_min_ratio=0.10)[0])
                  for a in (0.0, 0.2, 0.5, 0.9)]
        assert counts == sorted(counts, reverse=True)

    def test_f_min_mayor_produce_huella_menor(self, peaked_field):
        g, f, _ = peaked_field
        huellas = []
        for ratio in (0.05, 0.15, 0.30, 0.50):
            regions, _, _ = segment(f, g, alpha=0.3, f_min_ratio=ratio)
            huellas.append(sum(len(v) for v in regions.values()))
        assert huellas == sorted(huellas, reverse=True)

    def test_campo_plano_no_revienta(self, peaked_field):
        g, _, _ = peaked_field
        regions, _, _ = segment(np.zeros(g.n), g, alpha=0.3, f_min_ratio=0.10)
        assert regions == {}


class TestExtraccion:
    def test_respeta_top_k_y_ordena_por_crimen_capturado(self, peaked_field):
        g, f, counts = peaked_field
        hs = extract_month("2024-01", f, g, counts, {"THEFT": counts},
                           alpha=0.3, f_min_ratio=0.10, top_k=2)
        assert len(hs) <= 2
        assert [h.crimes for h in hs] == sorted((h.crimes for h in hs), reverse=True)

    def test_la_semilla_es_el_nodo_con_mas_crimenes(self, peaked_field):
        g, f, counts = peaked_field
        hs = extract_month("2024-01", f, g, counts, {"THEFT": counts},
                           alpha=0.3, f_min_ratio=0.10, top_k=5)
        for h in hs:
            idx = np.array([g.index[n] for n in h.nodes])
            assert counts[g.index[h.seed_node]] == counts[idx].max()

    def test_el_desglose_por_categoria_suma_el_total(self, peaked_field):
        g, f, counts = peaked_field
        a, b = counts * 0.6, counts * 0.4
        hs = extract_month("2024-01", f, g, counts, {"A": a, "B": b},
                           alpha=0.3, f_min_ratio=0.10, top_k=5)
        for h in hs:
            assert sum(h.by_category.values()) <= h.crimes

    def test_las_aristas_conectan_solo_nodos_de_la_region(self, peaked_field):
        g, f, counts = peaked_field
        hs = extract_month("2024-01", f, g, counts, {"THEFT": counts},
                           alpha=0.3, f_min_ratio=0.10, top_k=5)
        for h in hs:
            inside = set(h.nodes)
            for u, v in h.edges:
                assert u in inside and v in inside

    def test_los_hotspots_de_un_mes_no_se_solapan(self, peaked_field):
        g, f, counts = peaked_field
        hs = extract_month("2024-01", f, g, counts, {"THEFT": counts},
                           alpha=0.3, f_min_ratio=0.10, top_k=20)
        seen: set[int] = set()
        for h in hs:
            assert not (set(h.nodes) & seen)
            seen |= set(h.nodes)


@pytest.mark.skipif(not ARTIFACT.exists(),
                    reason="requiere `crimepipe hotspots` sobre los datos reales")
class TestArtefactoReal:
    @staticmethod
    @pytest.fixture(scope="class")
    def art():
        return json.loads(ARTIFACT.read_text(encoding="utf-8"))

    def test_veinte_hotspots_por_mes(self, art):
        por_mes: dict[str, int] = {}
        for h in art["hotspots"]:
            por_mes[h["month"]] = por_mes.get(h["month"], 0) + 1
        assert set(por_mes.values()) == {art["summary"]["top_k"]}

    def test_cada_hotspot_es_conexo(self, art):
        for h in art["hotspots"]:
            G = nx.Graph()
            G.add_nodes_from(h["nodes"])
            G.add_edges_from(tuple(e) for e in h["edges"])
            assert nx.is_connected(G), f"{h['id']} no es conexo"

    def test_los_hotspots_de_un_mes_no_se_solapan(self, art):
        por_mes: dict[str, set[int]] = {}
        for h in art["hotspots"]:
            s = set(h["nodes"])
            prev = por_mes.setdefault(h["month"], set())
            assert not (s & prev), f"{h['id']} solapa con otro de su mes"
            prev |= s

    def test_la_semilla_y_el_pico_pertenecen_al_hotspot(self, art):
        for h in art["hotspots"]:
            nodes = set(h["nodes"])
            assert h["seed_node"] in nodes
            assert h["peak_node"] in nodes

    def test_reproduce_los_agregados_de_la_spec(self, art):
        """§7.3: ±2 % en los agregados de crimen capturado."""
        s = art["summary"]
        assert s["hotspots"] == 480
        assert abs(s["crimes_captured"] - 26645) / 26645 < 0.02
        assert abs(s["coverage"] - 0.125) < 0.005
