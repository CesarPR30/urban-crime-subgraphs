"""Entrypoint del pipeline: `crimepipe <comando>` (PROJECT_SPEC §2).

Cada paso es invocable de forma independiente, cachea su salida y detecta si su
entrada no cambió para saltarse el recómputo. `--force` ignora la caché.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np

from .config import load_config
from .io import Manifest, Paths, cache_key, save_json, timed

logger = logging.getLogger("crimepipe")


# --------------------------------------------------------------------------- #
# Carga compartida
# --------------------------------------------------------------------------- #


def _load_crimes(cfg):
    from .ingest.crimes import fingerprint, load_crimes

    records, report = load_crimes(
        cfg.path(cfg.data.crimes_csv),
        categories=cfg.data.categories,
        month_from=cfg.data.month_from,
        month_to=cfg.data.month_to,
        dedupe=cfg.data.dedupe,
    )
    months = sorted({r.mes for r in records})
    mindex = {m: i for i, m in enumerate(months)}
    cats = sorted({r.tipo for r in records})
    cindex = {c: i for i, c in enumerate(cats)}

    arr = {
        "lat": np.fromiter((r.lat for r in records), dtype=np.float64, count=len(records)),
        "lon": np.fromiter((r.lon for r in records), dtype=np.float64, count=len(records)),
        "month": np.fromiter((mindex[r.mes] for r in records), dtype=np.int32, count=len(records)),
        "cat": np.fromiter((cindex[r.tipo] for r in records), dtype=np.int32, count=len(records)),
        # Firma del orden de `records`. Los artefactos que se indexan por
        # posición (`snapping.bin` contra `crimes.bin`) la llevan para que el
        # navegador pueda negarse a combinarlos si no vienen de la misma carga.
        "fingerprint": fingerprint(records),
    }
    return arr, months, cats, report


def _snap(cfg, paths, arr, *, force: bool):
    """Red vial + snapping, con caché en npz."""
    import json

    from .ingest.network import load_network, to_undirected_simple

    cache = paths.interim / "snapped.npz"
    G, info = load_network(cfg, force=force)
    H = to_undirected_simple(G)

    if cache.exists() and not force:
        z = np.load(cache)
        if int(z["n"]) == arr["lat"].size and int(z["graph_nodes"]) == H.number_of_nodes():
            logger.info("Snapping desde caché %s", cache)
            # Las métricas del snapping (distancias mediana/p95/máx) se guardan
            # con el caché: sin ellas, cualquier consumidor que dé con el caché
            # las perdería y solo las vería quien pagase el recómputo.
            stats = json.loads(str(z["stats"])) if "stats" in z.files else {}
            return H, info, z["node"], {**stats, "source": "cache"}
        logger.info("Caché de snapping obsoleto (cambió el input); se recalcula")

    from .snapping import snap_crimes

    res = snap_crimes(G, arr["lat"], arr["lon"], cfg)
    np.savez_compressed(
        cache, node=res.node, n=arr["lat"].size, graph_nodes=H.number_of_nodes(),
        stats=np.array(json.dumps(res.stats)),
    )
    return H, info, res.node, res.stats


def _node_counts(g, snapped_nodes, arr, months, cats):
    """`c_m(v)` y su desglose por categoría, en índices internos del CSR."""
    n = g.n
    valid = snapped_nodes >= 0
    idx = np.full(snapped_nodes.shape, -1, dtype=np.int64)
    lookup = g.index
    idx[valid] = [lookup.get(int(x), -1) for x in snapped_nodes[valid]]
    valid &= idx >= 0

    counts = {}
    cat_counts = {}
    for mi, m in enumerate(months):
        sel = valid & (arr["month"] == mi)
        c = np.zeros(n, dtype=np.float64)
        np.add.at(c, idx[sel], 1.0)
        counts[m] = c
        per_cat = {}
        for ci, cat in enumerate(cats):
            s2 = sel & (arr["cat"] == ci)
            cc = np.zeros(n, dtype=np.float64)
            np.add.at(cc, idx[s2], 1.0)
            per_cat[cat] = cc
        cat_counts[m] = per_cat
    return counts, cat_counts, int(valid.sum())


# --------------------------------------------------------------------------- #
# Comandos
# --------------------------------------------------------------------------- #


def cmd_ingest(cfg, args) -> int:
    paths = Paths.from_config(cfg).ensure()
    from .ingest.network import load_network

    arr, months, cats, report = _load_crimes(cfg)
    print(report.summary())
    _, info = load_network(cfg, force=args.force)

    mani = Manifest(paths.manifest)
    mani.record("ingest", cache_key(cfg.data.model_dump(), [cfg.path(cfg.data.crimes_csv)]),
                [paths.network], crimes=report.kept_rows, network=info)
    logger.info("Red: %d nodos, %d aristas (%s)", info["nodes"], info["edges"], info["source"])
    return 0


def _prepare(cfg, paths, *, force: bool):
    """Carga común de los pasos pesados: crímenes, red, snapping, conteos."""
    from .density import to_csr

    with timed("Carga de crímenes", logger):
        arr, months, cats, report = _load_crimes(cfg)
    logger.info("Crímenes: %s en %d meses, %d categorías",
                f"{report.kept_rows:,}", len(months), len(cats))

    with timed("Red vial y snapping", logger):
        H, net_info, snapped, snap_stats = _snap(cfg, paths, arr, force=force)

    with timed("Conversión a CSR", logger):
        g = to_csr(H)
    logger.info("Grafo: %d nodos, %d aristas", g.n, g.indices.size // 2)

    with timed("Conteos por nodo y mes", logger):
        counts, cat_counts, n_snapped = _node_counts(g, snapped, arr, months, cats)
    logger.info("Crímenes snappeados a la red: %s", f"{n_snapped:,}")

    return g, months, cats, counts, cat_counts, n_snapped, report, net_info, snap_stats


def _load_calibration(paths):
    from .io import load_json
    return load_json(paths.calibration) if paths.calibration.exists() else None


def _resolve_params(cfg, paths) -> dict:
    """Valores efectivos de σ, α y f_min, resolviendo los puestos en `auto`."""
    cal = _load_calibration(paths)
    sigma = cfg.density.resolved_sigma(cal)
    params = {
        "sigma_m": sigma,
        "radius_m": cfg.density.radius_for(sigma),
        "alpha": cfg.hotspots.resolved("alpha", cal),
        "f_min_ratio": cfg.hotspots.resolved("f_min_ratio", cal),
        "top_k": cfg.hotspots.top_k,
        "auto": [k for k, v in (("sigma_m", cfg.density.sigma_m),
                                ("alpha", cfg.hotspots.alpha),
                                ("f_min_ratio", cfg.hotspots.f_min_ratio))
                 if v == "auto"],
    }
    if params["auto"]:
        logger.info("Parámetros calibrados automáticamente: %s", ", ".join(params["auto"]))
    return params


def cmd_calibrate(cfg, args) -> int:
    """Determina σ, α y f_min desde los datos, sin usar §7.3 (§4.1, §4.3)."""
    paths = Paths.from_config(cfg).ensure()
    from .calibrate import calibrate

    g, months, cats, counts, cat_counts, n_snapped, report, net_info, _ = \
        _prepare(cfg, paths, force=args.force)

    base = {"sigma": 120.0, "alpha": 0.3, "f_min": 0.10, "k": cfg.hotspots.top_k}
    cal = calibrate(
        g, counts, cat_counts, months,
        truncate_sigmas=cfg.density.truncate_sigmas,
        base=base,
        progress=lambda s, t: logger.info("  σ = %.0f m listo (%.0f s)", s, t),
    )
    cal.meta.update(crimes_snapped=n_snapped, network=net_info)

    save_json(paths.calibration, cal.to_dict())
    out = cfg.root / "dashboard" / "public" / "data" / "param_sweep.json"
    save_json(out, _sweep_payload(cal, cfg))
    logger.info("Escrito %s y %s", paths.calibration, out)

    print()
    print(_format_calibration(cal, cfg))
    return 0


def _sweep_payload(cal, cfg) -> dict:
    """Artefacto para el dashboard, con la forma que ya consume la pestaña."""
    from dataclasses import asdict
    return {
        "meta": {**cal.meta, "curvature": round(cal.curvature, 4),
                 "alpha_sensitivity": round(cal.alpha_sensitivity, 4)},
        "baseline": {"sigma": cfg.density.sigma_m if cfg.density.sigma_m != "auto"
                     else cal.sigma_m,
                     "alpha": cfg.hotspots.alpha if cfg.hotspots.alpha != "auto"
                     else cal.alpha,
                     "f_min": cfg.hotspots.f_min_ratio
                     if cfg.hotspots.f_min_ratio != "auto" else cal.f_min_ratio,
                     "k": cfg.hotspots.top_k},
        "recommended": {"sigma": cal.sigma_m, "alpha": cal.alpha,
                        "f_min": cal.f_min_ratio, "curvature": round(cal.curvature, 4)},
        "targets": {"captured": 26645, "coverage": 0.125, "nodes": 8562, "density": 3.11},
        "axes": cal.meta["axes"],
        "grid": [{**asdict(p), "f_min": p.f_min} for p in cal.grid],
        "k_sweep": [asdict(p) for p in cal.k_sweep],
    }


def _format_calibration(cal, cfg) -> str:
    k = cal.knee
    out = ["", "=" * 66, "  CALIBRACIÓN AUTOMÁTICA", "=" * 66,
           f"  Malla evaluada        {len(cal.grid)} combinaciones",
           f"  Frontera de Pareto    {len(cal.front)} configuraciones no dominadas",
           f"  Curvatura de la rodilla {cal.curvature:.3f}"
           f"  ({'codo marcado' if cal.curvature > 0.15 else 'frontera casi recta'})",
           "",
           "  RECOMENDADO",
           f"    σ         {cal.sigma_m:g} m",
           f"    α         {cal.alpha:g}",
           f"    f_min     {cal.f_min_ratio:g}",
           f"    -> {100*k.coverage:.1f} % cobertura · {k.density:.2f} cr/nodo · "
           f"{k.nodes_per_month:.0f} nodos/mes · lift {k.lift:.1f}×",
           "",
           f"  α mueve la cobertura un {100*cal.alpha_sensitivity:.1f} % "
           f"({'relevante' if cal.alpha_sensitivity > 0.1 else 'irrelevante en este régimen'})",
           "=" * 66]
    cur = {"sigma": cfg.density.sigma_m, "alpha": cfg.hotspots.alpha,
           "f_min": cfg.hotspots.f_min_ratio}
    out.append(f"  config.yaml actual: σ={cur['sigma']} α={cur['alpha']} "
               f"f_min={cur['f_min']}")
    if cur["sigma"] != "auto" and float(cur["sigma"]) != cal.sigma_m:
        out.append(f"  ⚠ σ difiere del recomendado ({cur['sigma']} vs {cal.sigma_m:g}).")
    out.append("=" * 66)
    return "\n".join(out)


def cmd_hotspots(cfg, args) -> int:
    """Pasos 1 y 2: campo de densidad por mes y extracción de los top-K."""
    paths = Paths.from_config(cfg).ensure()
    from .density import KernelCache, density_field, to_csr
    from .hotspots import extract_month

    t_all = time.perf_counter()
    P = _resolve_params(cfg, paths)

    g, months, cats, counts, cat_counts, n_snapped, report, net_info, snap_stats = \
        _prepare(cfg, paths, force=args.force)

    # Un kernel por nodo fuente, compartido por los 24 meses.
    all_sources = np.unique(
        np.concatenate([np.nonzero(c)[0] for c in counts.values()])
    )
    logger.info("Nodos con al menos un crimen: %s", f"{all_sources.size:,}")
    kernels = KernelCache(g, P["sigma_m"], P["radius_m"]).build(all_sources)

    all_hotspots = []
    monthly = []
    for m in months:
        c = counts[m]
        src = np.nonzero(c)[0]
        f = density_field(kernels, src, c[src], g.n)
        hs = extract_month(
            m, f, g, c, cat_counts[m],
            alpha=P["alpha"], f_min_ratio=P["f_min_ratio"], top_k=P["top_k"],
        )
        all_hotspots.extend(hs)
        captured = sum(h.crimes for h in hs)
        nodes = sum(h.n_nodes for h in hs)
        total = int(c.sum())
        monthly.append({
            "month": m, "crimes": total, "captured": captured,
            "coverage": round(captured / total, 4) if total else 0.0,
            "nodes": nodes,
            "density": round(captured / nodes, 4) if nodes else 0.0,
            "hotspots": len(hs),
        })
        logger.info(
            "%s: %s crímenes, %s capturados (%.1f %%), %d nodos, %.2f cr/nodo",
            m, f"{total:,}", f"{captured:,}",
            100 * captured / total if total else 0, nodes,
            captured / nodes if nodes else 0,
        )

    tot_c = sum(r["crimes"] for r in monthly)
    tot_cap = sum(r["captured"] for r in monthly)
    tot_n = sum(r["nodes"] for r in monthly)
    hottest = max((int(c.max()) for c in counts.values()), default=0)
    summary = {
        "hotspots": len(all_hotspots),
        "months": len(months),
        "top_k": cfg.hotspots.top_k,
        "network_nodes": g.n,
        "network_edges": int(g.indices.size // 2),
        "crimes_loaded": report.kept_rows,
        "crimes_snapped": n_snapped,
        "crimes_captured": tot_cap,
        "coverage": round(tot_cap / tot_c, 4) if tot_c else 0.0,
        "nodes_in_hotspots": tot_n,
        "nodes_per_month": round(tot_n / len(months), 1) if months else 0,
        "footprint_fraction": round(tot_n / len(months) / g.n, 5) if months else 0,
        "density_hotspots": round(tot_cap / tot_n, 4) if tot_n else 0.0,
        "city_mean_density": round(tot_c / len(months) / g.n, 4) if months else 0.0,
        "hottest_node_crimes": hottest,
        "sigma_m": P["sigma_m"],
        "alpha": P["alpha"],
        "f_min_ratio": P["f_min_ratio"],
        "auto_params": P["auto"],
        "network": net_info,
        "snapping": snap_stats,
        "elapsed_s": round(time.perf_counter() - t_all, 1),
    }
    summary["lift"] = (
        round(summary["density_hotspots"] / summary["city_mean_density"], 2)
        if summary["city_mean_density"] else None
    )

    with timed("Serie mensual por subgrafo", logger):
        series = _footprint_series(g, counts, months, all_hotspots)

    save_json(paths.hotspots, {
        "summary": summary,
        "monthly": monthly,
        "months": months,
        "hotspots": [{**h.to_dict(), "series": series[h.id]} for h in all_hotspots],
    })
    Manifest(paths.manifest).record(
        "hotspots",
        cache_key({"params": {k: P[k] for k in
                              ("sigma_m", "radius_m", "alpha", "f_min_ratio", "top_k")}},
                  [cfg.path(cfg.data.crimes_csv), paths.network]),
        [paths.hotspots], summary=summary,
    )

    print()
    print(_format_summary(summary, monthly))
    return 0


def _footprint_series(g, counts, months, hotspots) -> dict[str, list[int]]:
    """Crímenes de cada subgrafo mes a mes, sobre TODA la ventana (§6).

    Un hotspot se extrae en un mes concreto, pero su huella —el conjunto de
    nodos— existe los 24. Contar el crimen de esa huella en cada mes es lo que
    convierte una foto en una serie, y es lo que compara el panel de contraste:
    dos zonas de estructura parecida pueden tener perfiles temporales muy
    distintos, y esa diferencia es justo el objeto del estudio.

    Ojo: la serie NO se recorta al mes de extracción. Si lo hiciera, sería un
    solo número y no habría nada que comparar.
    """
    lookup = g.index
    out: dict[str, list[int]] = {}
    for h in hotspots:
        idx = np.fromiter((lookup[n] for n in h.nodes if n in lookup),
                          dtype=np.int64)
        out[h.id] = [int(counts[m][idx].sum()) if idx.size else 0 for m in months]
    return out


def _format_summary(s: dict, monthly: list[dict]) -> str:
    exp = {
        "crimes_snapped": 213602, "network_nodes": 29537, "hotspots": 480,
        "crimes_captured": 26645, "coverage": 0.125, "nodes_in_hotspots": 8562,
        "density_hotspots": 3.11, "hottest_node_crimes": 59,
    }
    rows = [
        ("Crímenes snappeados a la red", s["crimes_snapped"], exp["crimes_snapped"]),
        ("Nodos en la red vial", s["network_nodes"], exp["network_nodes"]),
        ("Subgrafos extraídos", s["hotspots"], exp["hotspots"]),
        ("Crímenes en los top-K/mes", s["crimes_captured"], exp["crimes_captured"]),
        ("Nodos en subgrafos (24 meses)", s["nodes_in_hotspots"], exp["nodes_in_hotspots"]),
        ("Nodo más caliente", s["hottest_node_crimes"], exp["hottest_node_crimes"]),
    ]
    out = ["", "=" * 66, "  RESULTADO vs §7.3", "=" * 66,
           f"  {'Métrica':<32}{'Obtenido':>12}{'Esperado':>11}{'Δ':>9}"]
    for label, got, want in rows:
        d = (got - want) / want * 100 if want else 0
        out.append(f"  {label:<32}{got:>12,}{want:>11,}{d:>8.1f}%")
    out.append(f"  {'Cobertura':<32}{s['coverage']*100:>11.1f}%{exp['coverage']*100:>10.1f}%"
               f"{(s['coverage']-exp['coverage'])*100:>8.1f}p")
    out.append(f"  {'Densidad en hotspots':<32}{s['density_hotspots']:>12.2f}"
               f"{exp['density_hotspots']:>11.2f}"
               f"{(s['density_hotspots']-exp['density_hotspots'])/exp['density_hotspots']*100:>8.1f}%")
    out.append(f"  {'Media de la ciudad':<32}{s['city_mean_density']:>12.2f}{'~0.30':>11}")
    out.append(f"  {'Lift':<32}{s['lift']:>11.1f}×{'~10×':>11}")
    out.append("=" * 66)
    cov = [r["coverage"] for r in monthly]
    lo = min(monthly, key=lambda r: r["coverage"])
    hi = max(monthly, key=lambda r: r["coverage"])
    out.append(f"  Cobertura mensual: media {100*sum(cov)/len(cov):.1f} %, "
               f"mín {100*lo['coverage']:.1f} % ({lo['month']}), "
               f"máx {100*hi['coverage']:.1f} % ({hi['month']})")
    out.append(f"  Tiempo total: {s['elapsed_s']} s")
    return "\n".join(out)


def cmd_density(cfg, args) -> int:
    logger.info("`density` está integrado en `hotspots` (comparten el caché de kernels).")
    return cmd_hotspots(cfg, args)


def cmd_evaluate(cfg, args) -> int:
    """Fase D: línea base BFS, las cuatro métricas y la serie mensual (§7)."""
    paths = Paths.from_config(cfg).ensure()
    from .baseline import extract_month_bfs
    from .density import KernelCache, density_field
    from .evaluate import build_result, month_row, write_csv
    from .hotspots import extract_month

    t_all = time.perf_counter()
    P = _resolve_params(cfg, paths)

    g, months, cats, counts, cat_counts, n_snapped, report, net_info, _ = \
        _prepare(cfg, paths, force=args.force)

    all_sources = np.unique(np.concatenate([np.nonzero(c)[0] for c in counts.values()]))
    kernels = KernelCache(g, P["sigma_m"], P["radius_m"]).build(all_sources)

    # Se evalúan las dos lecturas de «region growing voraz por BFS» (§7.2): la
    # conclusión tiene que sostenerse contra la línea base fuerte, no solo
    # contra la ingenua.
    METHODS = ["topo", "greedy", "bfs"]
    monthly = []
    for m in months:
        c = counts[m]
        src = np.nonzero(c)[0]
        f = density_field(kernels, src, c[src], g.n)
        topo = extract_month(m, f, g, c, cat_counts[m],
                             alpha=P["alpha"], f_min_ratio=P["f_min_ratio"],
                             top_k=P["top_k"])

        # Presupuesto de nodos por región: la media de las topológicas de ese
        # mes, para que las familias ocupen la misma huella (§7.2). Sin igualar
        # la huella, comparar cobertura no mide nada.
        if cfg.baseline.match_footprint and topo:
            budget = max(1, round(sum(h.n_nodes for h in topo) / len(topo)))
        else:
            budget = cfg.baseline.node_budget

        by_method = {"topo": topo}
        for growth in ("greedy", "bfs"):
            by_method[growth] = extract_month_bfs(
                m, g, c, cat_counts[m], top_k=P["top_k"],
                budget=budget, growth=growth)

        row = month_row(m, int(c.sum()), by_method)
        row["bfs_budget"] = budget
        for k, hs in by_method.items():
            row[f"{k}_hotspots"] = len(hs)
        monthly.append(row)
        logger.info(
            "%s: topo %s · voraz %s · anchura %s cr "
            "(%d nodos topo, presupuesto %d)",
            m, f"{row['topo_captured']:,}", f"{row['greedy_captured']:,}",
            f"{row['bfs_captured']:,}", row["topo_nodes"], budget,
        )

    result = build_result(monthly, METHODS, meta={
        "crimes_snapped": n_snapped,
        "network_nodes": g.n,
        "months": len(months),
        "top_k": P["top_k"],
        "params": {k: P[k] for k in ("sigma_m", "alpha", "f_min_ratio")},
        "auto_params": P["auto"],
        "match_footprint": cfg.baseline.match_footprint,
        "network": net_info,
        "elapsed_s": round(time.perf_counter() - t_all, 1),
    })

    save_json(paths.evaluation, result.to_dict(), indent=2)
    write_csv(paths.evaluation_csv, monthly, METHODS)
    out = cfg.root / "dashboard" / "public" / "data" / "evaluation.json"
    save_json(out, result.to_dict())
    logger.info("Escrito %s, %s y %s", paths.evaluation, paths.evaluation_csv, out)

    print()
    print(result.summary())
    print(_format_eval_vs_spec(result))
    return 0


def _format_eval_vs_spec(res) -> str:
    """Contraste con la tabla de comparación de §7.3."""
    exp = {"topo_c": 26645, "topo_cov": 0.125, "topo_d": 3.11,
           "bfs_c": 23064, "bfs_cov": 0.108, "bfs_d": 3.02,
           "gain": 3581, "gain_rel": 0.155}
    gr = res.methods["greedy"]
    cmp_g = res.compare("greedy")
    rows = [
        ("Topológico · crímenes", res.topo.captured, exp["topo_c"]),
        ("Topológico · cobertura", res.topo.coverage * 100, exp["topo_cov"] * 100),
        ("Topológico · cr/nodo", res.topo.density, exp["topo_d"]),
        ("BFS voraz · crímenes", gr.captured, exp["bfs_c"]),
        ("BFS voraz · cobertura", gr.coverage * 100, exp["bfs_cov"] * 100),
        ("BFS voraz · cr/nodo", gr.density, exp["bfs_d"]),
        ("Ganancia absoluta", cmp_g["gain_absolute"], exp["gain"]),
        ("Ganancia relativa", cmp_g["gain_relative"] * 100, exp["gain_rel"] * 100),
    ]
    out = ["", "  Contraste con §7.3  (la línea base de referencia es la voraz)",
           f"  {'Métrica':<26}{'Obtenido':>12}{'Esperado':>11}{'Δ':>9}"]
    for label, got, want in rows:
        d = (got - want) / want * 100 if want else 0
        fm = (lambda v: f"{v:,.0f}") if abs(want) > 100 else (lambda v: f"{v:.2f}")
        out.append(f"  {label:<26}{fm(got):>12}{fm(want):>11}{d:>8.1f}%")
    naive = res.compare("bfs")
    out.append("")
    out.append(f"  Contra la anchura pura la ganancia es "
               f"{naive['gain_absolute']:+,} crímenes "
               f"({naive['gain_relative']*100:+.1f} %): el +15.5 % de §7.3 cae")
    out.append("  entre las dos lecturas de «voraz por BFS», que es la ambigüedad")
    out.append("  que esta evaluación deja explícita en vez de resolverla a ojo.")
    return "\n".join(out)


def cmd_export(cfg, args) -> int:
    """Traduce los hotspots a GeoJSON para el dashboard (§8)."""
    paths = Paths.from_config(cfg).ensure()
    from .export import hotspots_geojson
    from .ingest.network import load_network
    from .io import load_json

    if not paths.hotspots.exists():
        logger.error("Falta %s. Ejecuta antes `crimepipe hotspots`.", paths.hotspots)
        return 1

    art = load_json(paths.hotspots)
    G, _ = load_network(cfg)
    out_dir = cfg.root / "dashboard" / "public" / "data"

    with timed("Construcción de GeoJSON", logger):
        gj = hotspots_geojson(art, G, cfg)

    out = out_dir / "hotspots.geojson"
    save_json(out, gj)
    mb = out.stat().st_size / 1e6
    logger.info("Escrito %s (%.2f MB, %d features de %d hotspots)",
                out, mb, len(gj["features"]), len(art["hotspots"]))
    if mb > 8:
        logger.warning("El GeoJSON pesa %.1f MB; considera simplificar geometría (§8).", mb)

    _export_snapping(cfg, paths, G, out_dir, force=args.force)
    _export_pois(cfg, out_dir)
    return 0


def _export_pois(cfg, out_dir) -> None:
    """Emite `pois.bin` + `pois_meta.json` para la capa de POIs del mapa."""
    import json

    from .export import pois_binary
    from .ingest.pois import load_pois

    try:
        df, _, taxonomy = load_pois(cfg)
    except Exception as exc:
        logger.warning("Sin POIs (%s): el mapa no podrá dibujarlos. "
                       "Ejecuta `crimepipe pois`.", exc)
        return

    buf, meta = pois_binary(df["lat"].to_numpy(), df["lon"].to_numpy(),
                            df["category"].to_numpy(), taxonomy.categories,
                            taxonomy.labels)
    (out_dir / "pois.bin").write_bytes(buf)
    (out_dir / "pois_meta.json").write_text(
        json.dumps(meta, separators=(",", ":")), encoding="utf-8")
    logger.info("Escrito pois.bin (%.2f MB · %s POIs · %d categorías)",
                len(buf) / 1e6, f"{meta['n']:,}", len(meta["categories"]))


def _export_snapping(cfg, paths, G, out_dir, *, force: bool) -> None:
    """Emite `snapping.bin` + `snapping_meta.json` para la vista «por nodos».

    Es el mismo snapping que consume el resto del pipeline (misma caché
    `snapped.npz`), así que lo que dibuja el dashboard es literalmente lo que
    alimenta el campo de densidad, no una reconstrucción aparte.
    """
    import json

    from .export import snapping_binary
    from .ingest.network import to_undirected_simple

    with timed("Snapping para el dashboard", logger):
        arr, _, _, _ = _load_crimes(cfg)
        _, _, snapped, stats = _snap(cfg, paths, arr, force=force)

    H = to_undirected_simple(G)
    node_ids = list(H.nodes)
    buf, meta = snapping_binary(
        node_ids,
        [float(H.nodes[n]["y"]) for n in node_ids],
        [float(H.nodes[n]["x"]) for n in node_ids],
        snapped,
        fingerprint=arr["fingerprint"],
        n_crimes=arr["lat"].size,
        stats=stats,
    )

    (out_dir / "snapping.bin").write_bytes(buf)
    (out_dir / "snapping_meta.json").write_text(
        json.dumps(meta, separators=(",", ":")), encoding="utf-8"
    )
    logger.info(
        "Escrito snapping.bin (%.2f MB · %s de %s crímenes en red · %s nodos con carga de %s)",
        len(buf) / 1e6, f"{meta['snapped']:,}", f"{meta['n']:,}",
        f"{meta['nodes_touched']:,}", f"{meta['nodes']:,}",
    )


def cmd_pois(cfg, args) -> int:
    """Descarga, clasifica y cachea los puntos de interés (§3.2)."""
    paths = Paths.from_config(cfg).ensure()
    from .ingest.pois import load_pois

    df, report, taxonomy = load_pois(cfg, force=args.force)
    save_json(paths.processed / "poi_report.json", report.to_dict(), indent=2)
    print()
    print(report.summary())
    print()
    logger.info("Taxonomía: %d categorías, %d etiquetas OSM mapeadas",
                len(taxonomy.categories), len(taxonomy.lookup))
    return 0


def _load_poi_profiles(cfg, subs):
    """Perfiles de POI por subgrafo (§5.6), o `None` si no hay POIs.

    No es fatal que falten: el descriptor, el embedding y la similitud no los
    usan —§5.1 lo prohíbe— así que la fase E entera funciona sin ellos. Lo que
    se pierde es el contraste de la fase 3, y eso se dice en vez de fallar.
    """
    from .features.pois import assign
    from .ingest.pois import load_pois

    try:
        df, report, taxonomy = load_pois(cfg)
    except Exception as exc:
        logger.warning("No hay POIs (%s). La fase E sigue; el contraste de la "
                       "fase 3 queda pendiente. Ejecuta `crimepipe pois`.", exc)
        return None, None, None

    with timed("Asociación POI -> subgrafo", logger):
        profiles = assign(subs, df["lat"].to_numpy(), df["lon"].to_numpy(),
                          df["category"].to_numpy(), taxonomy.categories,
                          cfg.pois.buffer_m)
    tot = sum(p.total for p in profiles)
    empty = sum(1 for p in profiles if p.total == 0)
    logger.info("POIs: %s asociaciones en %d zonas (%d zonas sin ningún POI), "
                "entropía media %.3f",
                f"{tot:,}", len(profiles), empty,
                float(np.mean([p.entropy_norm for p in profiles])))
    return profiles, taxonomy, report


def cmd_features(cfg, args) -> int:
    """Caracterización y similitud entre subgrafos (§5)."""
    paths = Paths.from_config(cfg).ensure()
    from .features.embedding import build_embedder
    from .features.hierarchy import dims as hier_dims, hierarchy_matrix
    from .features.structural import (
        STRUCTURAL_DIMS, build_subgraphs, structural_matrix,
    )
    from .io import load_json
    from .similarity import Blocks, run

    if not paths.hotspots.exists():
        logger.error("Falta %s. Ejecuta antes `crimepipe hotspots`.", paths.hotspots)
        return 1

    from .ingest.network import load_network

    art = load_json(paths.hotspots)
    G, _ = load_network(cfg)

    with timed("Construcción de subgrafos", logger):
        subs = build_subgraphs(art, G)
    logger.info("Subgrafos: %d, de %d a %d nodos",
                len(subs), min(s.n for s in subs), max(s.n for s in subs))

    order = cfg.network.hierarchy_order
    with timed("Descriptor estructural (12 dims)", logger):
        S = structural_matrix(subs)
    with timed("Huella de jerarquía vial", logger):
        H = hierarchy_matrix(subs, order)

    embedder = build_embedder(cfg)
    with timed(f"Embedding {embedder.name}", logger):
        E = embedder.fit_transform([sg.graph() for sg in subs])
    logger.info("Embedding: %s (%s)", E.shape, embedder.name)

    blocks = Blocks(structural=S, embedding=E, hierarchy=H)
    with timed("Fusión, coseno, top-K y proyección", logger):
        res = run(subs, blocks, cfg)

    # Los POIs van DESPUÉS de la similitud, no antes. No es casualidad de
    # orden: §5.1 prohíbe que entren en la comparación, y calcularlos aquí deja
    # explícito que `run()` ya terminó sin haberlos visto.
    profiles, taxonomy, poi_report = _load_poi_profiles(cfg, subs)

    res.meta.update(
        embedding_method=embedder.name,
        embedding_requested=cfg.features.embedding.method,
        wl_iterations=cfg.features.embedding.wl_iterations,
        hierarchy_order=order,
        pois=(
            {
                "categories": taxonomy.categories,
                "labels": taxonomy.labels,
                "buffer_m": cfg.pois.buffer_m,
                "total_assigned": int(sum(p.total for p in profiles)),
                "zones_without_pois": int(sum(1 for p in profiles if p.total == 0)),
                "mean_entropy_norm": round(
                    float(np.mean([p.entropy_norm for p in profiles])), 4),
            }
            if profiles else None
        ),
    )

    payload = _similarity_payload(res, subs, S, H, order, profiles, art)
    save_json(paths.similarity, payload)
    if profiles and poi_report:
        save_json(paths.processed / "poi_profiles.json", {
            "meta": res.meta["pois"],
            "profiles": {sg.id: {
                "counts": [int(v) for v in p.counts],
                "total": p.total,
                "per_node": round(p.per_node, 4),
                "entropy": round(p.entropy, 4),
                "entropy_norm": round(p.entropy_norm, 4),
            } for sg, p in zip(subs, profiles)},
        })
    save_json(paths.embedding_2d, {
        "meta": {k: res.meta[k] for k in ("projection", "clustering", "n_clusters",
                                          "n_noise", "random_state")},
        "methods": {embedder.name: [
            {"id": i, "x": round(float(x), 4), "y": round(float(y), 4),
             "cluster": int(c)}
            for i, (x, y), c in zip(res.ids, res.coords, res.labels)
        ]},
    })
    out = cfg.root / "dashboard" / "public" / "data" / "similarity.json"
    save_json(out, payload)

    logger.info("Escrito %s y %s (%.2f MB)", paths.similarity, out,
                out.stat().st_size / 1e6)
    print()
    print(_format_similarity(res, subs))
    return 0


def _similarity_payload(res, subs, S, H, order, profiles=None, art=None) -> dict:
    """Artefacto único que consumen el dashboard y la fase 3.

    Junta en un solo archivo lo que describe a cada subgrafo (forma), lo que lo
    compara (similitud) y lo que se contrasta después (crimen y POIs). Van
    juntos porque el panel de comparación los necesita a la vez; el que no
    entren en la similitud lo garantiza `run()`, que ya terminó.
    """
    from .features.hierarchy import dims as hier_dims
    from .features.structural import STRUCTURAL_DIMS

    r5 = lambda v: [round(float(x), 5) for x in v]
    crime = {h["id"]: h for h in (art or {}).get("hotspots", [])}
    cats = sorted({c for h in crime.values() for c in h["by_category"]})

    out = {
        "meta": res.meta,
        "dims": {
            "structural": list(STRUCTURAL_DIMS),
            "hierarchy": list(hier_dims(order)),
            "crime_categories": cats,
        },
        "months": (art or {}).get("months", []),
        "hotspots": [],
    }
    for i, sg in enumerate(subs):
        h = crime.get(sg.id, {})
        rec = {
            "id": sg.id,
            "month": sg.month,
            "n_nodes": sg.n,
            "n_edges": len(sg.edges),
            "x": round(float(res.coords[i, 0]), 4),
            "y": round(float(res.coords[i, 1]), 4),
            "cluster": int(res.labels[i]),
            "structural": r5(S[i]),
            "hierarchy": r5(H[i]),
            "similar": res.top[sg.id],
            "similar_distinct": res.top_distinct[sg.id],
            # A partir de aquí, variables de contraste (§5.1): reporte, nunca
            # similitud.
            "crimes": h.get("crimes", 0),
            "by_category": [h.get("by_category", {}).get(c, 0) for c in cats],
            "series": h.get("series", []),
        }
        if profiles:
            p = profiles[i]
            rec["pois"] = {
                "counts": [int(v) for v in p.counts],
                "total": p.total,
                "per_node": round(p.per_node, 3),
                "entropy": round(p.entropy, 4),
                "entropy_norm": round(p.entropy_norm, 4),
            }
        out["hotspots"].append(rec)
    return out


def _format_similarity(res, subs) -> str:
    """Resumen legible: qué se calculó y un par de emparejamientos de muestra."""
    m = res.meta
    lines = [
        "Fase E · caracterización y similitud",
        "─" * 66,
        f"  hotspots                 {m['n']}",
        f"  bloques                  " + ", ".join(
            f"{k} {tuple(v)}" for k, v in m["blocks"].items()),
        f"  vector fusionado         {m['fused_dims']} dims  "
        f"(pesos {m['weights']})",
        f"  embedding                {m['embedding_method']}"
        + ("" if m["embedding_method"] == m["embedding_requested"]
           else f"  (pedido: {m['embedding_requested']}, fallback §5.3)"),
        f"  proyección / clustering  {m['projection']} / {m['clustering']}",
        f"  clústeres                {m['n_clusters']}  ({m['n_noise']} en ruido)",
        "",
        "  Top-3 de un hotspot de muestra, excluyendo el mismo lugar:",
    ]
    sample = subs[0].id
    for s in res.top_distinct[sample][:3]:
        lines.append(f"    {sample} → {s['id']}   coseno {s['score']:.4f}")
    same = res.top[sample][0] if res.top[sample] else None
    if same:
        lines.append(f"    sin excluir, el primero es {same['id']} "
                     f"(coseno {same['score']:.4f}, jaccard {same['jaccard']:.2f})")
    return "\n".join(lines)


COMMANDS = {
    "ingest": cmd_ingest,
    "pois": cmd_pois,
    "calibrate": cmd_calibrate,
    "density": cmd_density,
    "hotspots": cmd_hotspots,
    "evaluate": cmd_evaluate,
    "features": cmd_features,
    "export": cmd_export,
}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="crimepipe", description=__doc__)
    p.add_argument("command", choices=sorted(COMMANDS))
    p.add_argument("--config", type=Path, default=None)
    p.add_argument("--force", action="store_true", help="ignora la caché")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s  %(levelname)-7s %(name)-22s %(message)s",
        stream=sys.stdout,
    )
    cfg = load_config(args.config)
    return COMMANDS[args.command](cfg, args)


if __name__ == "__main__":
    raise SystemExit(main())
