"""Línea base, métricas de evaluación y calibración automática (§7, §4.3)."""

from __future__ import annotations

import json
from pathlib import Path

import networkx as nx
import numpy as np
import pytest

from pipeline.baseline import extract_month_bfs, grow_bfs, grow_greedy
from pipeline.calibrate import SweepPoint, knee_point, pareto_front
from pipeline.density import to_csr
from pipeline.evaluate import build_result, csv_columns, month_row, write_csv

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "data" / "processed" / "evaluation_report.json"
MONTHLY_CSV = ROOT / "data" / "processed" / "evaluation_monthly.csv"


@pytest.fixture
def counted(grid_graph):
    """Grilla con crimen concentrado en dos focos y ruido alrededor."""
    g = to_csr(grid_graph)
    ids = grid_graph.graph["ids"]
    c = np.zeros(g.n, dtype=np.float64)
    c[g.index[ids[(1, 1)]]] = 20
    c[g.index[ids[(1, 2)]]] = 12
    c[g.index[ids[(5, 5)]]] = 15
    c[g.index[ids[(5, 4)]]] = 9
    c[g.index[ids[(3, 3)]]] = 2
    return g, c


class TestCrecimiento:
    def test_ambos_producen_regiones_conexas(self, counted):
        g, c = counted
        for grow in (grow_bfs, grow_greedy):
            taken = np.zeros(g.n, dtype=bool)
            region = grow(g, 0, 8, taken, c)
            G = nx.Graph()
            G.add_nodes_from(region)
            for v in region:
                for k in range(g.indptr[v], g.indptr[v + 1]):
                    u = int(g.indices[k])
                    if u in set(region):
                        G.add_edge(v, u)
            assert nx.is_connected(G), grow.__name__

    def test_respetan_el_presupuesto(self, counted):
        g, c = counted
        for grow in (grow_bfs, grow_greedy):
            taken = np.zeros(g.n, dtype=bool)
            assert len(grow(g, 0, 6, taken, c)) <= 6

    def test_el_voraz_captura_mas_que_la_anchura(self, counted):
        """Es la razón de ser de las dos variantes: la voraz es el suelo alto."""
        g, c = counted
        seed = int(np.argmax(c))
        a = grow_greedy(g, seed, 10, np.zeros(g.n, dtype=bool), c)
        b = grow_bfs(g, seed, 10, np.zeros(g.n, dtype=bool), c)
        assert c[np.asarray(a)].sum() >= c[np.asarray(b)].sum()

    def test_no_reutiliza_nodos_ya_tomados(self, counted):
        g, c = counted
        taken = np.zeros(g.n, dtype=bool)
        first = grow_greedy(g, int(np.argmax(c)), 8, taken, c)
        second = grow_greedy(g, int(np.argsort(-c)[8]), 8, taken, c)
        assert not (set(first) & set(second))


class TestExtraccionBaseline:
    def test_regiones_disjuntas_y_dentro_de_top_k(self, counted):
        g, c = counted
        hs = extract_month_bfs("2024-01", g, c, {"THEFT": c}, top_k=3, budget=5)
        assert len(hs) <= 3
        seen: set[int] = set()
        for h in hs:
            assert not (set(h.nodes) & seen)
            seen |= set(h.nodes)

    def test_ordenadas_por_crimen_capturado(self, counted):
        g, c = counted
        hs = extract_month_bfs("2024-01", g, c, {"THEFT": c}, top_k=4, budget=5)
        assert [h.crimes for h in hs] == sorted((h.crimes for h in hs), reverse=True)

    def test_es_reproducible(self, counted):
        g, c = counted
        a = extract_month_bfs("2024-01", g, c, {"T": c}, top_k=4, budget=6)
        b = extract_month_bfs("2024-01", g, c, {"T": c}, top_k=4, budget=6)
        assert [h.nodes for h in a] == [h.nodes for h in b]

    def test_presupuesto_cero_no_produce_nada(self, counted):
        g, c = counted
        assert extract_month_bfs("2024-01", g, c, {"T": c}, top_k=4, budget=0) == []


class TestMetricas:
    def _row(self):
        from pipeline.hotspots import Hotspot
        mk = lambda n, cr, nodes: Hotspot(id=n, month="2024-01", nodes=list(range(nodes)),
                                          edges=[], crimes=cr)
        return month_row("2024-01", 1000, {
            "topo": [mk("a", 100, 20), mk("b", 50, 10)],
            "greedy": [mk("c", 80, 20), mk("d", 40, 10)],
        })

    def test_la_fila_mensual_tiene_las_cuatro_metricas(self):
        r = self._row()
        for k in ("topo", "greedy"):
            assert {f"{k}_captured", f"{k}_coverage", f"{k}_nodes", f"{k}_density"} <= set(r)
        assert r["topo_captured"] == 150
        assert r["topo_nodes"] == 30
        assert r["topo_coverage"] == pytest.approx(0.15)
        assert r["topo_density"] == pytest.approx(5.0)

    def test_los_deltas_van_contra_topo(self):
        r = self._row()
        assert r["delta_greedy_captured"] == 30
        assert r["delta_greedy_coverage_pts"] == pytest.approx(3.0)

    def test_los_agregados_suman_los_meses(self):
        rows = [self._row(), self._row()]
        res = build_result(rows, ["topo", "greedy"], meta={})
        assert res.topo.captured == 300
        assert res.topo.crimes == 2000
        assert res.topo.coverage == pytest.approx(0.15)
        assert res.compare("greedy")["gain_absolute"] == 60

    def test_el_csv_lleva_una_fila_por_mes(self, tmp_path):
        rows = [self._row() for _ in range(24)]
        for i, r in enumerate(rows):
            r["month"] = f"2024-{i % 12 + 1:02d}"
            r["bfs_budget"] = 15
        p = write_csv(tmp_path / "m.csv", rows, ["topo", "greedy"])
        lines = p.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 25                      # cabecera + 24 meses
        assert lines[0].split(",") == csv_columns(["topo", "greedy"])


class TestCalibracion:
    def _pt(self, cov, den, **kw):
        return SweepPoint(sigma=kw.get("sigma", 120), alpha=kw.get("alpha", .3),
                          f_min=kw.get("f_min", .1), k=20, captured=int(cov * 1000),
                          coverage=cov, nodes=100, nodes_per_month=100,
                          density=den, lift=den / .3, hotspots=480)

    def test_pareto_descarta_los_dominados(self):
        pts = [self._pt(.10, 4.0), self._pt(.20, 3.0), self._pt(.15, 2.0),
               self._pt(.05, 1.0)]
        front = pareto_front(pts)
        covs = [p.coverage for p in front]
        assert covs == sorted(covs)
        assert .15 not in covs and .05 not in covs   # dominados por (.20, 3.0)

    def test_la_frontera_esta_ordenada_por_cobertura(self):
        pts = [self._pt(.30, 1.0), self._pt(.10, 5.0), self._pt(.20, 3.0)]
        front = pareto_front(pts)
        assert [p.coverage for p in front] == [.10, .20, .30]

    def test_la_rodilla_cae_en_el_codo(self):
        """Curva con codo claro en (0.12, 3.0): el knee debe encontrarlo."""
        pts = [self._pt(.04, 6.0), self._pt(.08, 4.5), self._pt(.12, 3.0),
               self._pt(.30, 1.4), self._pt(.50, 1.0)]
        knee, curvature = knee_point(pareto_front(pts))
        assert knee.coverage == pytest.approx(.12)
        assert curvature > 0.1

    def test_una_frontera_recta_no_tiene_rodilla(self):
        pts = [self._pt(.10, 5.0), self._pt(.20, 4.0), self._pt(.30, 3.0),
               self._pt(.40, 2.0)]
        _, curvature = knee_point(pareto_front(pts))
        assert curvature < 0.05

    def test_frontera_degenerada_no_revienta(self):
        knee, curvature = knee_point([self._pt(.1, 3.0)])
        assert knee is not None and curvature == 0.0


@pytest.mark.skipif(not REPORT.exists(),
                    reason="requiere `crimepipe evaluate` sobre los datos reales")
class TestArtefactoReal:
    @staticmethod
    @pytest.fixture(scope="class")
    def rep():
        return json.loads(REPORT.read_text(encoding="utf-8"))

    def test_hay_veinticuatro_filas_mensuales(self, rep):
        assert len(rep["monthly"]) == 24

    def test_el_csv_mensual_existe_y_tiene_24_filas(self):
        lines = MONTHLY_CSV.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 25

    def test_la_huella_esta_igualada_entre_metodos(self, rep):
        """El punto de §7.1: sin huella comparable, la cobertura no mide nada."""
        topo = rep["methods"]["topo"]["nodes"]
        for key, m in rep["methods"].items():
            if key == "topo":
                continue
            assert abs(m["nodes"] - topo) / topo < 0.05, key

    def test_el_topologico_gana_a_las_dos_lineas_base(self, rep):
        for c in rep["comparisons"]:
            assert c["gain_absolute"] > 0, c["against"]

    def test_el_voraz_domina_a_la_anchura_pura(self, rep):
        assert (rep["methods"]["greedy"]["captured"]
                > rep["methods"]["bfs"]["captured"])
