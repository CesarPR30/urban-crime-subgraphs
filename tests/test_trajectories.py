"""Enlace de subgrafos entre meses y estudios de caso.

Lo que se comprueba aquí es lo que puede fallar en silencio: que el
seguimiento no encadene sitios distintos, que la frecuencia cuente meses y no
apariciones, y que las medidas de contraste se comporten como dicen sus
docstrings. Un error en cualquiera de las tres no rompe nada, solo produce
números plausibles y falsos.
"""

from __future__ import annotations

import numpy as np
import pytest

from pipeline.casestudies import (
    benjamini_hochberg, cliffs_delta, matched_pairs,
)
from pipeline.trajectories import iou, link, quadrants


def _hs(month, hid, nodes, crimes=10, seed=None):
    return {
        "id": f"{month}_{hid}", "month": month, "nodes": list(nodes),
        "crimes": crimes, "density": crimes / max(len(nodes), 1),
        "n_nodes": len(nodes), "seed_node": nodes[0] if seed is None else seed,
    }


# ── IoU ───────────────────────────────────────────────────────────────────

def test_iou_basico():
    assert iou({1, 2, 3}, {1, 2, 3}) == 1.0
    assert iou({1, 2}, {3, 4}) == 0.0
    assert iou(set(), {1}) == 0.0
    assert iou({1, 2, 3, 4}, {3, 4, 5, 6}) == pytest.approx(2 / 6)


def test_iou_no_premia_al_grande():
    """Treinta nodos compartidos valen mucho entre zonas pequeñas y poco entre
    grandes. Es la razón de usar Jaccard y no el recuento crudo."""
    chico = iou(set(range(35)), set(range(5, 40)))
    grande = iou(set(range(300)), set(range(270, 570)))
    assert chico > grande


# ── Enlace ────────────────────────────────────────────────────────────────

def test_enlace_sigue_el_mismo_sitio():
    months = ["2024-01", "2024-02", "2024-03"]
    hs = [_hs("2024-01", "h1", range(0, 10)),
          _hs("2024-02", "h1", range(1, 11)),
          _hs("2024-03", "h1", range(2, 12))]
    trajs = link(hs, months, min_iou=0.2, max_gap=1)
    assert len(trajs) == 1
    assert trajs[0].k == 3
    assert trajs[0].frequency(3) == 1.0


def test_enlace_separa_sitios_distintos():
    months = ["2024-01", "2024-02"]
    hs = [_hs("2024-01", "h1", range(0, 10)),
          _hs("2024-02", "h1", range(100, 110))]
    trajs = link(hs, months, min_iou=0.2)
    assert len(trajs) == 2
    assert all(t.k == 1 for t in trajs)


def test_no_encadena_por_transitividad():
    """A solapa con B y B con C, pero A y C no comparten nada.

    Con componentes conexas los tres caerían en una trayectoria y la frecuencia
    saldría 1.0 para un sitio que se ha desplazado entero. El seguimiento
    temporal sí los une —es un desplazamiento gradual, no dos sitios— pero la
    ESTABILIDAD tiene que delatarlo, que es justo para lo que existe.
    """
    months = ["2024-01", "2024-02", "2024-03"]
    hs = [_hs("2024-01", "h1", range(0, 10)),
          _hs("2024-02", "h1", range(7, 17)),
          _hs("2024-03", "h1", range(14, 24))]
    trajs = link(hs, months, min_iou=0.1)
    assert len(trajs) == 1
    # Se movió: comparte poco entre pasos consecutivos y nada entre extremos.
    assert trajs[0].stability < 0.3
    assert iou(set(range(0, 10)), set(range(14, 24))) == 0.0


def test_hueco_maximo():
    months = ["2024-01", "2024-02", "2024-03", "2024-04"]
    hs = [_hs("2024-01", "h1", range(0, 10)),
          _hs("2024-04", "h1", range(0, 10))]
    # Con hueco de 3 sobrevive; con hueco de 1 se parte en dos.
    assert len(link(hs, months, max_gap=3)) == 1
    assert len(link(hs, months, max_gap=1)) == 2


def test_emparejamiento_uno_a_uno():
    """Dos subgrafos del mes siguiente compiten por la misma trayectoria.

    Solo uno puede continuarla, y tiene que ser el de mayor solape; el otro
    abre trayectoria propia.
    """
    months = ["2024-01", "2024-02"]
    hs = [_hs("2024-01", "h1", range(0, 10)),
          _hs("2024-02", "h1", range(0, 9)),     # IoU alto
          _hs("2024-02", "h2", range(6, 16))]    # IoU bajo
    trajs = link(hs, months, min_iou=0.1)
    assert len(trajs) == 2
    cont = [t for t in trajs if t.k == 2][0]
    assert cont.members == ["2024-01_h1", "2024-02_h1"]


def test_frecuencia_cuenta_meses_no_apariciones():
    """Dos subgrafos del MISMO mes en la misma trayectoria no valen por dos."""
    months = ["2024-01", "2024-02"]
    hs = [_hs("2024-01", "h1", range(0, 10)),
          _hs("2024-02", "h1", range(0, 10))]
    t = link(hs, months)[0]
    assert t.frequency(2) == 1.0
    assert t.frequency(10) == 0.2


def test_estabilidad_none_con_una_aparicion():
    """`None` y no `0`: una aparición única no es inestable, es que no aplica.

    Con 0 caería en el cubo «muy móvil» y contaminaría el eje entero.
    """
    t = link([_hs("2024-01", "h1", range(0, 10))], ["2024-01"])[0]
    assert t.stability is None
    assert t.movement_m is None


def test_quadrants_excluye_las_de_una_aparicion():
    months = [f"2024-{m:02d}" for m in range(1, 5)]
    hs = [_hs("2024-01", "h1", range(0, 10)),
          _hs("2024-02", "h1", range(0, 10)),
          _hs("2024-03", "h9", range(500, 510))]
    trajs = link(hs, months)
    q = quadrants(trajs, len(months))
    assert q["single_appearance"] == 1
    assert sum(q["counts"].values()) == 1


# ── Contraste ─────────────────────────────────────────────────────────────

def test_cliffs_delta_extremos():
    a = np.array([10.0, 11.0, 12.0])
    b = np.array([1.0, 2.0, 3.0])
    assert cliffs_delta(a, b) == pytest.approx(1.0)
    assert cliffs_delta(b, a) == pytest.approx(-1.0)
    assert cliffs_delta(a, a) == pytest.approx(0.0)


def test_cliffs_delta_con_empates():
    """Los empates no cuentan ni a favor ni en contra."""
    a = np.array([1.0, 2.0, 3.0])
    b = np.array([2.0, 2.0, 2.0])
    # Uno mayor, uno menor, uno empatado -> (1 - 1) / 9 = 0
    assert cliffs_delta(a, b) == pytest.approx(0.0)


def test_benjamini_hochberg_monotono_y_acotado():
    p = [0.001, 0.01, 0.02, 0.5, 0.9]
    adj = benjamini_hochberg(p)
    assert all(0.0 <= v <= 1.0 for v in adj)
    assert all(x <= y + 1e-12 for x, y in zip(adj, adj[1:])), "no monótono"
    assert all(a >= b - 1e-12 for a, b in zip(adj, p)), "ajustado menor que crudo"


def test_benjamini_hochberg_vacio():
    assert benjamini_hochberg([]) == []


def test_matched_pairs_excluye_solapados():
    """Dos recortes de la misma esquina no son un contraste."""
    X = np.array([[0.0, 0.0], [0.0, 0.0], [5.0, 5.0]])
    crimes = np.array([100.0, 10.0, 50.0])
    ids = ["a", "b", "c"]
    months = ["2024-01", "2024-02", "2024-03"]
    solapan = [set(range(10)), set(range(10)), {999}]
    assert matched_pairs(X, crimes, ids, months, solapan, min_ratio=3.0) == []

    disjuntos = [{1, 2}, {3, 4}, {999}]
    out = matched_pairs(X, crimes, ids, months, disjuntos, min_ratio=3.0)
    assert len(out) == 1
    assert out[0]["high"] == "a" and out[0]["low"] == "b"


def test_matched_pairs_limita_repeticiones():
    """Un subgrafo extremo no puede copar la lista entera."""
    n = 12
    X = np.zeros((n, 2))
    crimes = np.array([1000.0] + [10.0] * (n - 1))
    ids = [f"h{i}" for i in range(n)]
    months = ["2024-01"] * n
    sets = [{i} for i in range(n)]
    out = matched_pairs(X, crimes, ids, months, sets,
                        n_pairs=20, min_ratio=3.0, topo_quantile=1.0)
    veces = sum(1 for r in out if r["high"] == "h0" or r["low"] == "h0")
    assert veces <= 2
