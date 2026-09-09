"""Barrido del piso de densidad `f_min` contra los objetivos de §7.3.

La especificación fija σ (120 m) y α (0.3) con valores explícitos, pero de
`f_min` solo dice que existe y para qué sirve: «mantener las regiones
compactas» (§4.3). Es el único parámetro libre del extractor, así que se
determina midiendo, no eligiéndolo a ojo.

`f_min` es el piso por debajo del cual un nodo no entra en el barrido, expresado
como fracción de `max f`. Controla directamente la huella: cuanto más bajo, más
lejos se derraman las regiones por la cola del kernel.

Uso:
    .venv/Scripts/python scripts/calibrate_fmin.py [--values 0.05 0.1 ...]
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pipeline.cli import _load_crimes, _node_counts, _snap  # noqa: E402
from pipeline.config import load_config  # noqa: E402
from pipeline.density import KernelCache, density_field, to_csr  # noqa: E402
from pipeline.hotspots import extract_month  # noqa: E402
from pipeline.io import Paths  # noqa: E402

logger = logging.getLogger("calibrate")

# Objetivos de §7.3
TARGET = {
    "captured": 26645,
    "coverage": 0.125,
    "nodes": 8562,
    "density": 3.11,
}
DEFAULT_VALUES = [0.02, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.60]


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", default=None,
                   help="ciudad a calibrar (bloque de `datasets:` en config.yaml)")
    p.add_argument("--values", type=float, nargs="+", default=DEFAULT_VALUES)
    p.add_argument("--alpha", type=float, default=None, help="sobrescribe α")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)-7s %(name)-18s %(message)s")
    cfg = load_config(dataset=args.dataset)
    paths = Paths.from_config(cfg).ensure()
    alpha = args.alpha if args.alpha is not None else cfg.hotspots.alpha

    arr, months, cats, report = _load_crimes(cfg, paths)
    H, _, snapped, _ = _snap(cfg, paths, arr, force=False)
    g = to_csr(H)
    counts, cat_counts, n_snapped = _node_counts(g, snapped, arr, months, cats)

    # Los kernels no dependen de f_min: se construyen una vez para todo el barrido.
    sources = np.unique(np.concatenate([np.nonzero(c)[0] for c in counts.values()]))
    kernels = KernelCache(g, cfg.density.sigma_m, cfg.density.radius_m).build(sources)
    fields = {m: density_field(kernels, np.nonzero(counts[m])[0],
                               counts[m][np.nonzero(counts[m])[0]], g.n)
              for m in months}
    total_crimes = sum(int(counts[m].sum()) for m in months)

    print()
    print(f"  α = {alpha}   σ = {cfg.density.sigma_m} m   K = {cfg.hotspots.top_k}   "
          f"crímenes snappeados = {n_snapped:,}")
    print("  " + "─" * 78)
    print(f"  {'f_min':>7} {'capturados':>12} {'cobertura':>11} {'nodos':>9} "
          f"{'nod/mes':>9} {'cr/nodo':>9} {'lift':>7}")
    print("  " + "─" * 78)

    city_mean = total_crimes / len(months) / g.n
    best = None
    for ratio in args.values:
        cap = nodes = 0
        for m in months:
            hs, _ = extract_month(m, fields[m], g, counts[m], cat_counts[m],
                               alpha=alpha, f_min_ratio=ratio,
                               top_k=cfg.hotspots.top_k)
            cap += sum(h.crimes for h in hs)
            nodes += sum(h.n_nodes for h in hs)
        cov = cap / total_crimes if total_crimes else 0
        dens = cap / nodes if nodes else 0
        lift = dens / city_mean if city_mean else 0
        # Error relativo conjunto sobre las dos métricas que la spec marca como
        # no opcionales: huella y densidad (§7.1).
        err = (abs(nodes - TARGET["nodes"]) / TARGET["nodes"]
               + abs(dens - TARGET["density"]) / TARGET["density"])
        flag = ""
        if best is None or err < best[0]:
            best = (err, ratio)
        print(f"  {ratio:>7.2f} {cap:>12,} {cov*100:>10.1f}% {nodes:>9,} "
              f"{nodes/len(months):>9.0f} {dens:>9.2f} {lift:>6.1f}×{flag}")

    print("  " + "─" * 78)
    print(f"  {'§7.3':>7} {TARGET['captured']:>12,} {TARGET['coverage']*100:>10.1f}% "
          f"{TARGET['nodes']:>9,} {TARGET['nodes']/24:>9.0f} {TARGET['density']:>9.2f} "
          f"{'~10.0×':>7}")
    print("  " + "─" * 78)
    print(f"  Mejor ajuste sobre huella+densidad: f_min = {best[1]:.2f} "
          f"(error relativo conjunto {best[0]:.3f})")
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
