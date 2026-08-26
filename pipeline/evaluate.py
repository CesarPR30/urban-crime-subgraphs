"""Métricas y comparación contra la línea base (PROJECT_SPEC §7.1 – §7.4).

Cuatro métricas por mes y en total, para el extractor y para cada línea base:

1. crímenes crudos dentro de la unión de los top-K subgrafos
2. cobertura como fracción de todos los crímenes de ese mes
3. **huella total de nodos** (compacidad)
4. **densidad de crímenes por nodo**

(3) y (4) no son opcionales. Sin ellas cualquier método gana la métrica de
cobertura simplemente haciendo los subgrafos más grandes — y esa es justo la
comparación que no se quiere hacer.

## Por qué hay dos líneas base y no una

§7.2 pide un «region growing voraz por BFS», que admite dos lecturas con
resultados muy distintos:

* **voraz** — la frontera se ordena por crimen del vecino, así que la región
  elige hacia dónde crecer;
* **anchura pura** — cola FIFO, crece igual en todas direcciones y se traga
  tantos nodos vacíos como calientes.

La primera es una línea base fuerte, la segunda un suelo. Se evalúan las dos
porque la conclusión del proyecto —«el método topológico gana a igual huella»—
tiene que sostenerse contra la versión fuerte, no solo contra la ingenua.
"""

from __future__ import annotations

import csv
import logging
from dataclasses import dataclass, field
from pathlib import Path

from .hotspots import Hotspot

logger = logging.getLogger(__name__)

#: Etiqueta legible de cada método, y el prefijo que usa en el CSV.
METHOD_LABELS = {
    "topo": "Topológico",
    "greedy": "BFS voraz",
    "bfs": "BFS anchura",
}


@dataclass
class MethodMetrics:
    """Agregados de un método sobre toda la ventana."""

    key: str
    label: str
    captured: int = 0
    crimes: int = 0
    nodes: int = 0
    hotspots: int = 0

    @property
    def coverage(self) -> float:
        return self.captured / self.crimes if self.crimes else 0.0

    @property
    def density(self) -> float:
        return self.captured / self.nodes if self.nodes else 0.0

    def to_dict(self) -> dict:
        return {
            "method": self.key,
            "label": self.label,
            "captured": self.captured,
            "coverage": round(self.coverage, 5),
            "nodes": self.nodes,
            "density": round(self.density, 4),
            "hotspots": self.hotspots,
        }


@dataclass
class EvaluationResult:
    """Comparación cabeza a cabeza más la serie mensual completa (§7.4)."""

    methods: dict[str, MethodMetrics]
    monthly: list[dict] = field(default_factory=list)
    meta: dict = field(default_factory=dict)

    @property
    def topo(self) -> MethodMetrics:
        return self.methods["topo"]

    def compare(self, key: str) -> dict:
        """Ganancia del extractor topológico sobre la línea base `key`."""
        base = self.methods[key]
        return {
            "against": key,
            "gain_absolute": self.topo.captured - base.captured,
            "gain_relative": round(
                (self.topo.captured - base.captured) / base.captured, 5
            ) if base.captured else None,
            "coverage_points": round((self.topo.coverage - base.coverage) * 100, 3),
            "density_delta": round(self.topo.density - base.density, 4),
            "footprint_delta": self.topo.nodes - base.nodes,
            "footprint_ratio": round(self.topo.nodes / base.nodes, 4) if base.nodes else None,
        }

    def to_dict(self) -> dict:
        return {
            "meta": self.meta,
            "methods": {k: m.to_dict() for k, m in self.methods.items()},
            "comparisons": [self.compare(k) for k in self.methods if k != "topo"],
            "monthly": self.monthly,
        }

    def summary(self) -> str:
        out = ["", "=" * 74,
               "  EVALUACIÓN · extractor topológico vs líneas base", "=" * 74,
               f"  {'Método':<15}{'Crímenes':>11}{'Cobertura':>12}"
               f"{'Nodos':>10}{'cr/nodo':>10}{'Subgrafos':>11}"]
        for m in self.methods.values():
            out.append(f"  {m.label:<15}{m.captured:>11,}{m.coverage*100:>11.1f}%"
                       f"{m.nodes:>10,}{m.density:>10.2f}{m.hotspots:>11,}")
        out.append("  " + "─" * 72)
        for key in self.methods:
            if key == "topo":
                continue
            c = self.compare(key)
            rel = f"{c['gain_relative']*100:+.1f}%" if c["gain_relative"] is not None else "—"
            out.append(f"  vs {self.methods[key].label:<12}{c['gain_absolute']:>+11,}"
                       f"{c['coverage_points']:>+11.1f}p"
                       f"{c['footprint_delta']:>+10,}{c['density_delta']:>+10.2f}"
                       f"{rel:>11}")
        out.append("=" * 74)
        out.append("  Lectura: la huella de nodos está igualada por construcción, así que")
        out.append("  la diferencia en crimen capturado mide *dónde* se ponen los nodos,")
        out.append("  no cuántos. La mejora no viene de inflar los subgrafos.")
        out.append("=" * 74)
        return "\n".join(out)


def csv_columns(methods: list[str]) -> list[str]:
    cols = ["month", "crimes"]
    for k in methods:
        cols += [f"{k}_captured", f"{k}_coverage", f"{k}_nodes", f"{k}_density"]
    cols.append("bfs_budget")
    for k in methods:
        if k != "topo":
            cols += [f"delta_{k}_captured", f"delta_{k}_coverage_pts"]
    return cols


def month_row(month: str, crimes: int, by_method: dict[str, list[Hotspot]]) -> dict:
    """Fila de la serie mensual con las cuatro métricas para cada método."""
    row = {"month": month, "crimes": crimes}
    agg = {}
    for key, hs in by_method.items():
        cap = sum(h.crimes for h in hs)
        nod = sum(h.n_nodes for h in hs)
        cov = cap / crimes if crimes else 0.0
        den = cap / nod if nod else 0.0
        agg[key] = (cap, cov)
        row[f"{key}_captured"] = cap
        row[f"{key}_coverage"] = round(cov, 5)
        row[f"{key}_nodes"] = nod
        row[f"{key}_density"] = round(den, 4)
    t_cap, t_cov = agg["topo"]
    for key, (cap, cov) in agg.items():
        if key == "topo":
            continue
        row[f"delta_{key}_captured"] = t_cap - cap
        row[f"delta_{key}_coverage_pts"] = round((t_cov - cov) * 100, 3)
    return row


def build_result(monthly: list[dict], methods: list[str], meta: dict) -> EvaluationResult:
    ms = {k: MethodMetrics(k, METHOD_LABELS.get(k, k)) for k in methods}
    for r in monthly:
        for k, m in ms.items():
            m.crimes += r["crimes"]
            m.captured += r[f"{k}_captured"]
            m.nodes += r[f"{k}_nodes"]
            m.hotspots += r.get(f"{k}_hotspots", 0)
    return EvaluationResult(methods=ms, monthly=monthly, meta=meta)


def write_csv(path: Path, monthly: list[dict], methods: list[str]) -> Path:
    """Serie mensual completa: una fila por mes, todos los métodos (§7.4)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = csv_columns(methods)
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(monthly)
    logger.info("Escrito %s (%d filas, %d columnas)", path, len(monthly), len(cols))
    return path
