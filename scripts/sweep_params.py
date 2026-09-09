"""Barrido de σ, α, f_min y K: ¿qué valores capturan más crimen? (§7.1)

## La trampa que este barrido evita

Maximizar la cobertura a secas no dice nada: cualquier método gana esa métrica
simplemente **haciendo los subgrafos más grandes**. Bajar `f_min` de 0.10 a 0.02
sube la cobertura del 12.5 % al 56.7 %, pero la densidad se desploma de 2.82 a
0.89 cr/nodo y el *lift* sobre la media de la ciudad cae de 9.6× a 3.0×: a esa
altura los "hotspots" son media ciudad y ya no señalan nada.

Por eso cada combinación se mide con las cuatro métricas de §7.1 —crimen
capturado, cobertura, **huella de nodos** y **densidad**— y el resultado se
presenta como frontera de Pareto entre cobertura y densidad, no como un ranking.

## Coste

Los kernels gaussianos dependen solo de σ, así que se construyen una vez por
valor de σ y se reutilizan en todas las combinaciones de α y f_min que cuelgan
de él. Sin eso el barrido sería inviable.

Uso:
    .venv/Scripts/python scripts/sweep_params.py [--quick]
"""

from __future__ import annotations

import argparse
import itertools
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pipeline.cli import _load_crimes, _node_counts, _snap  # noqa: E402
from pipeline.config import load_config  # noqa: E402
from pipeline.density import KernelCache, density_field, to_csr  # noqa: E402
from pipeline.hotspots import extract_month  # noqa: E402
from pipeline.io import Paths, save_json  # noqa: E402

logger = logging.getLogger("sweep")

SIGMAS = [60.0, 90.0, 120.0, 150.0, 200.0, 250.0]
ALPHAS = [0.1, 0.2, 0.3, 0.4, 0.5]
FMINS = [0.05, 0.075, 0.10, 0.125, 0.15, 0.20]
KS = [5, 10, 20, 30, 50, 100]

QUICK = {"sigmas": [90.0, 120.0, 150.0], "alphas": [0.2, 0.3, 0.4],
         "fmins": [0.075, 0.10, 0.15]}

TARGETS = {"captured": 26645, "coverage": 0.125, "nodes": 8562, "density": 3.11}


def evaluate(months, fields, g, counts, cat_counts, alpha, fmin, k):
    """Agrega las cuatro métricas de §7.1 para una combinación."""
    cap = nodes = n_hs = 0
    for m in months:
        hs, _ = extract_month(m, fields[m], g, counts[m], cat_counts[m],
                           alpha=alpha, f_min_ratio=fmin, top_k=k)
        cap += sum(h.crimes for h in hs)
        nodes += sum(h.n_nodes for h in hs)
        n_hs += len(hs)
    return cap, nodes, n_hs


def pareto_front(rows: list[dict]) -> list[int]:
    """Índices no dominados en (cobertura ↑, densidad ↑).

    Un punto domina a otro si es al menos igual en ambas métricas y
    estrictamente mejor en una. La frontera es el conjunto de configuraciones
    donde no se puede ganar cobertura sin perder densidad.
    """
    front = []
    for i, a in enumerate(rows):
        dominated = any(
            b["coverage"] >= a["coverage"] and b["density"] >= a["density"]
            and (b["coverage"] > a["coverage"] or b["density"] > a["density"])
            for j, b in enumerate(rows) if i != j
        )
        if not dominated:
            front.append(i)
    return front


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--quick", action="store_true", help="malla reducida (3×3×3)")
    p.add_argument("--dataset", default=None,
                   help="ciudad a barrer (bloque de `datasets:` en config.yaml)")
    # Por defecto escribe en la carpeta del dataset, no en la raiz de `data/`:
    # el barrido es de una ciudad concreta y machacar el de la otra dejaria al
    # dashboard mostrando una frontera de Pareto que no es la suya.
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)-7s %(name)-16s %(message)s")
    t0 = time.perf_counter()
    cfg = load_config(dataset=args.dataset)
    paths = Paths.from_config(cfg).ensure()
    out_path = args.out or (cfg.dashboard_data / "param_sweep.json")

    sigmas = QUICK["sigmas"] if args.quick else SIGMAS
    alphas = QUICK["alphas"] if args.quick else ALPHAS
    fmins = QUICK["fmins"] if args.quick else FMINS

    arr, months, cats, report = _load_crimes(cfg, paths)
    H, _, snapped, _ = _snap(cfg, paths, arr, force=False)
    g = to_csr(H)
    counts, cat_counts, n_snapped = _node_counts(g, snapped, arr, months, cats)
    total = sum(int(counts[m].sum()) for m in months)
    city_mean = total / len(months) / g.n

    logger.info("Malla: %d σ × %d α × %d f_min = %d combinaciones",
                len(sigmas), len(alphas), len(fmins),
                len(sigmas) * len(alphas) * len(fmins))

    grid: list[dict] = []
    fields_by_sigma = {}
    for sigma in sigmas:
        radius = sigma * cfg.density.truncate_sigmas
        logger.info("σ = %.0f m (r = %.0f m): construyendo kernels…", sigma, radius)
        kern = KernelCache(g, sigma, radius)
        src_all = np.unique(np.concatenate([np.nonzero(counts[m])[0] for m in months]))
        kern.build(src_all)
        fields = {}
        for m in months:
            src = np.nonzero(counts[m])[0]
            fields[m] = density_field(kern, src, counts[m][src], g.n)
        fields_by_sigma[sigma] = fields

        for alpha, fmin in itertools.product(alphas, fmins):
            cap, nodes, n_hs = evaluate(months, fields, g, counts, cat_counts,
                                        alpha, fmin, cfg.hotspots.top_k)
            dens = cap / nodes if nodes else 0.0
            grid.append({
                "sigma": sigma, "alpha": alpha, "f_min": fmin, "k": cfg.hotspots.top_k,
                "captured": cap, "coverage": round(cap / total, 5),
                "nodes": nodes, "nodes_per_month": round(nodes / len(months), 1),
                "density": round(dens, 4),
                "lift": round(dens / city_mean, 3) if city_mean else 0.0,
                "hotspots": n_hs,
            })
        logger.info("  σ = %.0f m listo (%d combinaciones, %.0f s acumulados)",
                    sigma, len(alphas) * len(fmins), time.perf_counter() - t0)

    # Barrido de K sobre la configuración base: K controla directamente el
    # trade-off cobertura/huella sin tocar la forma de las regiones.
    base_fields = fields_by_sigma.get(cfg.density.sigma_m)
    k_sweep = []
    if base_fields is not None:
        for k in KS:
            cap, nodes, n_hs = evaluate(months, base_fields, g, counts, cat_counts,
                                        cfg.hotspots.alpha, cfg.hotspots.f_min_ratio, k)
            dens = cap / nodes if nodes else 0.0
            k_sweep.append({
                "k": k, "captured": cap, "coverage": round(cap / total, 5),
                "nodes": nodes, "nodes_per_month": round(nodes / len(months), 1),
                "density": round(dens, 4),
                "lift": round(dens / city_mean, 3) if city_mean else 0.0,
                "hotspots": n_hs,
            })
            logger.info("  K = %d: %s capturados (%.1f %%), %.0f nodos/mes, %.2f cr/nodo",
                        k, f"{cap:,}", 100 * cap / total, nodes / len(months), dens)

    front = pareto_front(grid)
    for i in front:
        grid[i]["pareto"] = True

    payload = {
        "meta": {
            "crimes_total": total,
            "crimes_snapped": n_snapped,
            "network_nodes": g.n,
            "months": len(months),
            "city_mean_density": round(city_mean, 4),
            "truncate_sigmas": cfg.density.truncate_sigmas,
            "elapsed_s": round(time.perf_counter() - t0, 1),
            "quick": args.quick,
        },
        "baseline": {
            "sigma": cfg.density.sigma_m, "alpha": cfg.hotspots.alpha,
            "f_min": cfg.hotspots.f_min_ratio, "k": cfg.hotspots.top_k,
        },
        "targets": TARGETS,
        "axes": {"sigma": sigmas, "alpha": alphas, "f_min": fmins, "k": KS},
        "grid": grid,
        "k_sweep": k_sweep,
    }
    save_json(out_path, payload, indent=None)
    logger.info("Escrito %s (%d combinaciones, %d en la frontera de Pareto) en %.0f s",
                out_path, len(grid), len(front), time.perf_counter() - t0)

    best_cov = max(grid, key=lambda r: r["coverage"])
    best_dens = max(grid, key=lambda r: r["density"])
    print()
    print("  Máxima cobertura : σ=%.0f α=%.1f f_min=%.3f -> %.1f %% cobertura, "
          "%.2f cr/nodo, %.0f nodos/mes"
          % (best_cov["sigma"], best_cov["alpha"], best_cov["f_min"],
             100 * best_cov["coverage"], best_cov["density"], best_cov["nodes_per_month"]))
    print("  Máxima densidad  : σ=%.0f α=%.1f f_min=%.3f -> %.1f %% cobertura, "
          "%.2f cr/nodo, %.0f nodos/mes"
          % (best_dens["sigma"], best_dens["alpha"], best_dens["f_min"],
             100 * best_dens["coverage"], best_dens["density"], best_dens["nodes_per_month"]))
    print("  Configuración actual está %sen la frontera de Pareto."
          % ("" if any(
              r.get("pareto") and r["sigma"] == cfg.density.sigma_m
              and r["alpha"] == cfg.hotspots.alpha
              and r["f_min"] == cfg.hotspots.f_min_ratio for r in grid) else "NO "))
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
