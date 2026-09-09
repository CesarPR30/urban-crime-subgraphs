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


def _load_crimes(cfg, paths=None):
    """Crímenes del dataset, como vectores paralelos, con caché en disco.

    La carga es lectura de CSV en stdlib puro: sobre el export de Chicago
    cuesta segundos y no merecía caché. Sobre el CSV de Lima —1.8 GB, 3.1
    millones de filas, 58 columnas— cuesta diez minutos, y *cada* paso del
    pipeline la repite. Sin caché, una corrida completa gasta más tiempo
    releyendo el mismo CSV que calculando.

    La clave es la misma que usa el manifest: los parámetros de `data` más la
    huella del CSV. Cambia la ventana, las categorías, el perfil o el archivo y
    el caché se invalida solo.
    """
    import dataclasses
    import json

    from .ingest.crimes import ValidationReport, fingerprint, load_crimes

    key = cache_key(cfg.data.model_dump(), [cfg.path(cfg.data.crimes_csv)])
    cache = (paths.interim / "crimes_load.npz") if paths else None
    if cache is not None and cache.exists():
        z = np.load(cache, allow_pickle=False)
        if str(z["key"]) == key:
            logger.info("Crímenes desde caché %s", cache)
            payload = json.loads(str(z["payload"]))
            # `to_dict` no es la inversa del constructor: aplana y añade campos
            # derivados. Se filtra por los campos reales del dataclass para no
            # colgarle atributos que no le pertenecen.
            names = {f.name for f in dataclasses.fields(ValidationReport)}
            fields = {k: v for k, v in payload["report"].items() if k in names}
            if fields.get("bbox"):
                fields["bbox"] = tuple(fields["bbox"])
            report = ValidationReport(**fields)
            return (
                {"lat": z["lat"], "lon": z["lon"], "month": z["month"],
                 "cat": z["cat"], "fingerprint": str(z["fingerprint"])},
                payload["months"], payload["cats"], report,
            )
        logger.info("Caché de crímenes obsoleto (cambió `data`); se recarga")

    records, report = load_crimes(
        cfg.path(cfg.data.crimes_csv),
        categories=cfg.data.categories,
        month_from=cfg.data.month_from,
        month_to=cfg.data.month_to,
        dedupe=cfg.data.dedupe,
        profile=cfg.data.source_profile,
        cleaning=cfg.data.cleaning,
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
    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            cache, key=key, fingerprint=arr["fingerprint"],
            lat=arr["lat"], lon=arr["lon"], month=arr["month"], cat=arr["cat"],
            payload=json.dumps(
                {"months": months, "cats": cats, "report": report.to_dict()},
                ensure_ascii=False,
            ),
        )
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


def _kernels_for_selection(cfg, P, g, counts):
    """Caché de kernels y pool de nodos, dimensionados según el criterio.

    Lo devuelve `cmd_hotspots` y `cmd_evaluate` por igual, y eso es el punto:
    si la extracción y su evaluación no usan exactamente el mismo criterio de
    selección, la comparación con la línea base mide la diferencia entre los
    dos criterios en vez de la diferencia entre los dos métodos.
    """
    from .density import KernelCache

    sel_cfg = cfg.hotspots.selection
    mc = sel_cfg.method == "montecarlo"
    all_sources = np.unique(
        np.concatenate([np.nonzero(c)[0] for c in counts.values()])
    )
    # Con el nulo uniforme, las réplicas colocan crimen en nodos que nunca lo
    # tuvieron, y el kernel de cada uno cuesta una Dijkstra. Construirlos dentro
    # del bucle de réplicas sería pagarlos una y otra vez.
    uniform = mc and sel_cfg.null == "uniform"
    pool = np.arange(g.n, dtype=np.int64) if uniform else all_sources
    logger.info("Nodos con al menos un crimen: %s%s", f"{all_sources.size:,}",
                f" (kernels para los {g.n:,} de la red: nulo uniforme)" if uniform else "")
    kernels = KernelCache(g, P["sigma_m"], P["radius_m"]).build(pool)
    return kernels, pool, sel_cfg, mc


def _month_hotspots(cfg, P, g, kernels, pool, counts, cat_counts, m, *,
                    sel_cfg, mc, rng):
    """Campo, distribución nula y hotspots de un mes, con el criterio activo."""
    from .density import density_field
    from .hotspots import extract_month, segment
    from .selection import null_max_statistic

    c = counts[m]
    src = np.nonzero(c)[0]
    f = density_field(kernels, src, c[src], g.n)

    null_max = None
    if mc:
        null_max = null_max_statistic(
            int(c.sum()), pool, c,
            kernels=kernels, g=g, segment_fn=segment,
            alpha=P["alpha"], f_min_ratio=P["f_min_ratio"],
            replicates=sel_cfg.replicates, null=sel_cfg.null, rng=rng,
        )
    hs, sel = extract_month(
        m, f, g, c, cat_counts[m],
        alpha=P["alpha"], f_min_ratio=P["f_min_ratio"],
        selection=sel_cfg, null_max=null_max,
    )
    return f, hs, sel


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

    arr, months, cats, report = _load_crimes(cfg, paths)
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
        arr, months, cats, report = _load_crimes(cfg, paths)
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

    # El barrido son 180 combinaciones y cada una recorre todos los meses. Sobre
    # Chicago (24 meses, 29 537 nodos) es viable entero; sobre Lima (90 meses,
    # 135 633) son horas. `--sweep-months` toma una muestra **regular** de la
    # ventana —no los primeros N, que serían todos del mismo año y de la misma
    # estación— y la sensibilidad de un parámetro se mide igual de bien: lo que
    # se compara es cómo responden las métricas al moverlo, no su nivel absoluto.
    if args.sweep_months and args.sweep_months < len(months):
        step = len(months) / args.sweep_months
        picked = [months[int(i * step)] for i in range(args.sweep_months)]
        logger.info("Barrido sobre %d de %d meses (muestra regular): %s … %s",
                    len(picked), len(months), picked[0], picked[-1])
        months = picked

    base = {"sigma": 120.0, "alpha": 0.3, "f_min": 0.10, "k": cfg.hotspots.top_k}
    cal = calibrate(
        g, counts, cat_counts, months,
        truncate_sigmas=cfg.density.truncate_sigmas,
        base=base,
        progress=lambda s, t: logger.info("  σ = %.0f m listo (%.0f s)", s, t),
    )
    cal.meta.update(crimes_snapped=n_snapped, network=net_info)

    save_json(paths.calibration, cal.to_dict())
    out = cfg.dashboard_data / "param_sweep.json"
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
        # El dashboard dibuja estas marcas sobre la frontera de Pareto. Sin
        # tabla de referencia no se emiten: una marca prestada de otra ciudad
        # se leería como un objetivo que este barrido no alcanza.
        "targets": None if cfg.reference is None else {
            "captured": cfg.reference.crimes_captured,
            "coverage": cfg.reference.coverage,
            "nodes": cfg.reference.nodes_in_hotspots,
            "density": cfg.reference.density_hotspots,
        },
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
    """Pasos 1 y 2: campo de densidad por mes y extracción de las regiones."""
    paths = Paths.from_config(cfg).ensure()
    from .density import to_csr

    t_all = time.perf_counter()
    P = _resolve_params(cfg, paths)

    g, months, cats, counts, cat_counts, n_snapped, report, net_info, snap_stats = \
        _prepare(cfg, paths, force=args.force)

    kernels, pool, sel_cfg, mc = _kernels_for_selection(cfg, P, g, counts)

    if mc:
        logger.info(
            "Selección por significancia: %d réplicas, nulo %r, alpha=%.3f "
            "(%d meses -> %s extracciones nulas)",
            sel_cfg.replicates, sel_cfg.null, sel_cfg.alpha_sig, len(months),
            f"{sel_cfg.replicates * len(months):,}",
        )
    rng = np.random.default_rng(cfg.project.random_state)

    all_hotspots = []
    monthly = []
    for m in months:
        c = counts[m]
        _, hs, sel = _month_hotspots(cfg, P, g, kernels, pool, counts, cat_counts, m,
                                     sel_cfg=sel_cfg, mc=mc, rng=rng)
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
            # Con K adaptativo, cuántas regiones sobrevivieron y de cuántas
            # candidatas es un resultado del mes, no un parámetro. Va al
            # artefacto para poder graficarlo.
            "selection": sel.summary(),
        })
        logger.info(
            "%s: %s crímenes, %s capturados (%.1f %%), %d nodos, %.2f cr/nodo, "
            "%d de %d regiones",
            m, f"{total:,}", f"{captured:,}",
            100 * captured / total if total else 0, nodes,
            captured / nodes if nodes else 0,
            len(hs), sel.n_candidates,
        )

    tot_c = sum(r["crimes"] for r in monthly)
    tot_cap = sum(r["captured"] for r in monthly)
    tot_n = sum(r["nodes"] for r in monthly)
    hottest = max((int(c.max()) for c in counts.values()), default=0)
    per_month = [r["hotspots"] for r in monthly]
    summary = {
        "hotspots": len(all_hotspots),
        "months": len(months),
        "top_k": cfg.hotspots.top_k,
        "selection": {
            **sel_cfg.model_dump(),
            "hotspots_min": min(per_month) if per_month else 0,
            "hotspots_max": max(per_month) if per_month else 0,
            "hotspots_mean": round(sum(per_month) / len(per_month), 2) if per_month else 0,
            "candidates_mean": round(
                sum(r["selection"]["candidates"] for r in monthly) / len(monthly), 1
            ) if monthly else 0,
        },
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
    print(_format_summary(summary, monthly, cfg.reference))
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


def _format_summary(s: dict, monthly: list[dict], ref=None) -> str:
    """Resumen de la extracción, contrastado con §7.3 si la ciudad tiene tabla.

    Sin `ref` se imprimen los mismos valores sin columnas de contraste. Restar
    contra los números de otra ciudad daría un delta con formato impecable y
    sin significado, que es peor que no dar ninguno.
    """
    rows = [
        ("Crímenes snappeados a la red", s["crimes_snapped"], "crimes_snapped"),
        ("Nodos en la red vial", s["network_nodes"], "network_nodes"),
        ("Subgrafos extraídos", s["hotspots"], "hotspots"),
        ("Crímenes en los top-K/mes", s["crimes_captured"], "crimes_captured"),
        ("Nodos en subgrafos", s["nodes_in_hotspots"], "nodes_in_hotspots"),
        ("Nodo más caliente", s["hottest_node_crimes"], "hottest_node_crimes"),
    ]
    title = "  RESULTADO vs §7.3" if ref else "  RESULTADO DE LA EXTRACCIÓN"
    out = ["", "=" * 66, title, "=" * 66]
    if ref:
        out.append(f"  {'Métrica':<32}{'Obtenido':>12}{'Esperado':>11}{'Δ':>9}")
        for label, got, key in rows:
            want = getattr(ref, key)
            d = (got - want) / want * 100 if want else 0
            out.append(f"  {label:<32}{got:>12,}{want:>11,}{d:>8.1f}%")
        out.append(f"  {'Cobertura':<32}{s['coverage']*100:>11.1f}%{ref.coverage*100:>10.1f}%"
                   f"{(s['coverage']-ref.coverage)*100:>8.1f}p")
        out.append(f"  {'Densidad en hotspots':<32}{s['density_hotspots']:>12.2f}"
                   f"{ref.density_hotspots:>11.2f}"
                   f"{(s['density_hotspots']-ref.density_hotspots)/ref.density_hotspots*100:>8.1f}%")
        out.append(f"  {'Media de la ciudad':<32}{s['city_mean_density']:>12.2f}{'~0.30':>11}")
        out.append(f"  {'Lift':<32}{s['lift']:>11.1f}×{'~10×':>11}")
    else:
        out.append(f"  {'Métrica':<32}{'Obtenido':>12}")
        for label, got, _ in rows:
            out.append(f"  {label:<32}{got:>12,}")
        out.append(f"  {'Cobertura':<32}{s['coverage']*100:>11.1f}%")
        out.append(f"  {'Densidad en hotspots':<32}{s['density_hotspots']:>12.2f}")
        out.append(f"  {'Media de la ciudad':<32}{s['city_mean_density']:>12.2f}")
        out.append(f"  {'Lift':<32}{s['lift']:>11.1f}×")
        out.append("  (este dataset no tiene tabla de referencia en §7.3: no hay")
        out.append("   valores esperados contra los que contrastar)")
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
    from .evaluate import build_result, month_row, write_csv

    t_all = time.perf_counter()
    P = _resolve_params(cfg, paths)

    g, months, cats, counts, cat_counts, n_snapped, report, net_info, _ = \
        _prepare(cfg, paths, force=args.force)

    kernels, pool, sel_cfg, mc = _kernels_for_selection(cfg, P, g, counts)
    rng = np.random.default_rng(cfg.project.random_state)

    # Se evalúan las dos lecturas de «region growing voraz por BFS» (§7.2): la
    # conclusión tiene que sostenerse contra la línea base fuerte, no solo
    # contra la ingenua.
    METHODS = ["topo", "greedy", "bfs"]
    monthly = []
    for m in months:
        c = counts[m]
        _, topo, _sel = _month_hotspots(cfg, P, g, kernels, pool, counts, cat_counts, m,
                                        sel_cfg=sel_cfg, mc=mc, rng=rng)

        # Presupuesto de nodos por región: la media de las topológicas de ese
        # mes, para que las familias ocupen la misma huella (§7.2). Sin igualar
        # la huella, comparar cobertura no mide nada.
        if cfg.baseline.match_footprint and topo:
            budget = max(1, round(sum(h.n_nodes for h in topo) / len(topo)))
        else:
            budget = cfg.baseline.node_budget

        # La línea base recibe **tantas regiones como sacó el extractor ese
        # mes**, no `top_k`. Con K adaptativo el número de hotspots es un
        # resultado del mes; dejar la base en 20 fijas mientras el topológico
        # saca 5 compararía cobertura entre familias de tamaño distinto, que es
        # exactamente lo que §7.1 advierte que no mide nada.
        k_month = len(topo)
        by_method = {"topo": topo}
        for growth in ("greedy", "bfs"):
            by_method[growth] = extract_month_bfs(
                m, g, c, cat_counts[m], top_k=k_month,
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

    # Con K adaptativo «top_k» ya no describe la corrida: el número de regiones
    # es un resultado de cada mes. Se publica el criterio y el rango observado
    # para que el dashboard no anuncie un 20 que no se usó en ningún mes.
    k_obs = [r["topo_hotspots"] for r in monthly]
    result = build_result(monthly, METHODS, meta={
        "crimes_snapped": n_snapped,
        "network_nodes": g.n,
        "months": len(months),
        "top_k": P["top_k"],
        "selection": {
            **sel_cfg.model_dump(),
            "hotspots_min": min(k_obs) if k_obs else 0,
            "hotspots_max": max(k_obs) if k_obs else 0,
            "hotspots_mean": round(sum(k_obs) / len(k_obs), 2) if k_obs else 0,
        },
        "params": {k: P[k] for k in ("sigma_m", "alpha", "f_min_ratio")},
        "auto_params": P["auto"],
        "match_footprint": cfg.baseline.match_footprint,
        "network": net_info,
        "elapsed_s": round(time.perf_counter() - t_all, 1),
    })

    save_json(paths.evaluation, result.to_dict(), indent=2)
    write_csv(paths.evaluation_csv, monthly, METHODS)
    out = cfg.dashboard_data / "evaluation.json"
    save_json(out, result.to_dict())
    logger.info("Escrito %s, %s y %s", paths.evaluation, paths.evaluation_csv, out)

    print()
    print(result.summary())
    print(_format_eval_vs_spec(result, cfg.reference))
    return 0


def _format_eval_vs_spec(res, ref=None) -> str:
    """Contraste con la tabla de comparación de §7.3.

    Vacío si la ciudad no tiene tabla. La comparación con la línea base sigue
    imprimiéndose siempre —esa es interna al dataset y siempre significa algo—;
    lo que desaparece es la columna de valores esperados.
    """
    naive = res.compare("bfs")
    if ref is None:
        return (
            f"\n  Contra la anchura pura la ganancia es "
            f"{naive['gain_absolute']:+,} crímenes "
            f"({naive['gain_relative']*100:+.1f} %).\n"
            "  Sin tabla de referencia en §7.3 para este dataset: los números de\n"
            "  arriba se leen contra su propia línea base, no contra Chicago."
        )

    gr = res.methods["greedy"]
    cmp_g = res.compare("greedy")
    rows = [
        ("Topológico · crímenes", res.topo.captured, ref.crimes_captured),
        ("Topológico · cobertura", res.topo.coverage * 100, ref.coverage * 100),
        ("Topológico · cr/nodo", res.topo.density, ref.density_hotspots),
        ("BFS voraz · crímenes", gr.captured, ref.baseline_captured),
        ("BFS voraz · cobertura", gr.coverage * 100, ref.baseline_coverage * 100),
        ("BFS voraz · cr/nodo", gr.density, ref.baseline_density),
        ("Ganancia absoluta", cmp_g["gain_absolute"], ref.gain_absolute),
        ("Ganancia relativa", cmp_g["gain_relative"] * 100, ref.gain_relative * 100),
    ]
    out = ["", "  Contraste con §7.3  (la línea base de referencia es la voraz)",
           f"  {'Métrica':<26}{'Obtenido':>12}{'Esperado':>11}{'Δ':>9}"]
    for label, got, want in rows:
        d = (got - want) / want * 100 if want else 0
        fm = (lambda v: f"{v:,.0f}") if abs(want) > 100 else (lambda v: f"{v:.2f}")
        out.append(f"  {label:<26}{fm(got):>12}{fm(want):>11}{d:>8.1f}%")
    out.append("")
    out.append(f"  Contra la anchura pura la ganancia es "
               f"{naive['gain_absolute']:+,} crímenes "
               f"({naive['gain_relative']*100:+.1f} %): el "
               f"{ref.gain_relative*100:+.1f} % de §7.3 cae")
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
    out_dir = cfg.dashboard_data

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
    _export_pois(cfg, out_dir, skip=getattr(args, 'no_pois', False))
    return 0


def _export_pois(cfg, out_dir, *, skip: bool = False) -> None:
    """Emite `pois.bin` + `pois_meta.json` para la capa de POIs del mapa."""
    import json

    from .export import pois_binary
    from .ingest.pois import load_pois

    if skip:
        logger.info("POIs omitidos por --no-pois; no se emite pois.bin.")
        return
    try:
        df, _, taxonomy = load_pois(cfg)
    except Exception as exc:
        logger.warning("Sin POIs (%s): el mapa no podrá dibujarlos. "
                       "Ejecuta `crimepipe pois`.", exc)
        return

    buf, meta = pois_binary(df["lat"].to_numpy(), df["lon"].to_numpy(),
                            df["category"].to_numpy(), taxonomy.categories,
                            taxonomy.labels,
                            year=df["year"].to_numpy() if "year" in df else None)
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
        arr, _, _, _ = _load_crimes(cfg, paths)
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


def _load_poi_profiles(cfg, subs, *, skip: bool = False, city_nodes=None):
    """Perfiles de POI por subgrafo (§5.6), o `None` si no hay POIs.

    No es fatal que falten: el descriptor, el embedding y la similitud no los
    usan —§5.1 lo prohíbe— así que la fase E entera funciona sin ellos. Lo que
    se pierde es el contraste de la fase 3, y eso se dice en vez de fallar.
    """
    from .features.pois import assign
    from .ingest.pois import load_pois

    if skip:
        logger.info("POIs omitidos por --no-pois; el perfil funcional de los "
                    "subgrafos queda pendiente.")
        return None, None, None
    try:
        df, report, taxonomy = load_pois(cfg)
    except Exception as exc:
        logger.warning("No hay POIs (%s). La fase E sigue; el contraste de la "
                       "fase 3 queda pendiente. Ejecuta `crimepipe pois`.", exc)
        return None, None, None

    # Con instantáneas anuales cada subgrafo se caracteriza contra el mapa de
    # **su** año. Sin ellas, contra el único que hay.
    temporal = "year" in df.columns and df["year"].nunique() > 1
    poi_year = df["year"].to_numpy() if temporal else None
    sub_year = [int(sg.month[:4]) for sg in subs] if temporal else None
    if temporal:
        logger.info("POIs temporales: %d instantáneas (%s)",
                    df["year"].nunique(),
                    ", ".join(str(int(y)) for y in sorted(df["year"].unique())))

    with timed("Asociación POI -> subgrafo", logger):
        profiles = assign(subs, df["lat"].to_numpy(), df["lon"].to_numpy(),
                          df["category"].to_numpy(), taxonomy.categories,
                          cfg.pois.buffer_m,
                          poi_year=poi_year, sub_year=sub_year,
                          city_nodes=city_nodes)
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

    from .features.embedding import build_embedders

    graphs = [sg.graph() for sg in subs]
    embedders = build_embedders(cfg)          # el principal primero
    results: dict[str, object] = {}
    for emb in embedders:
        with timed(f"Embedding {emb.name}", logger):
            Ek = emb.fit_transform(graphs)
        logger.info("Embedding: %s (%s)", Ek.shape, emb.name)
        with timed(f"Fusión, coseno, top-K y proyección ({emb.name})", logger):
            rk = run(subs, Blocks(structural=S, embedding=Ek, hierarchy=H), cfg)
        rk.meta["embedding_dims"] = int(Ek.shape[1])
        results[emb.name] = rk

    embedder = embedders[0]
    res = results[embedder.name]

    # Los POIs van DESPUÉS de la similitud, no antes. No es casualidad de
    # orden: §5.1 prohíbe que entren en la comparación, y calcularlos aquí deja
    # explícito que `run()` ya terminó sin haberlos visto.
    profiles, taxonomy, poi_report = _load_poi_profiles(
        cfg, subs, skip=getattr(args, 'no_pois', False),
        city_nodes=G.number_of_nodes())

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
                "source": cfg.pois.source,
                "years": sorted({p.year for p in profiles if p.year is not None}),
                "total_assigned": int(sum(p.total for p in profiles)),
                "zones_without_pois": int(sum(1 for p in profiles if p.total == 0)),
                "mean_entropy_norm": round(
                    float(np.mean([p.entropy_norm for p in profiles])), 4),
            }
            if profiles else None
        ),
    )

    payload = _similarity_payload(res, subs, S, H, order, profiles, art)

    # Modo comparativo: cada método con su top-5 y sus coordenadas 2D, para que
    # el cajón de embeddings del dashboard pueda conmutar entre ellos. El
    # principal ya está en el nivel superior; aquí van todos, con la misma forma.
    if len(results) > 1:
        payload["methods"] = {
            name: {
                "meta": {
                    "embedding_dims": rk.meta.get("embedding_dims"),
                    "fused_dims": rk.meta["fused_dims"],
                    "projection": rk.meta["projection"],
                    "clustering": rk.meta["clustering"],
                    "n_clusters": rk.meta["n_clusters"],
                    "n_noise": rk.meta["n_noise"],
                },
                "hotspots": {
                    sg.id: {
                        "x": round(float(rk.coords[i, 0]), 4),
                        "y": round(float(rk.coords[i, 1]), 4),
                        "cluster": int(rk.labels[i]),
                        "similar": rk.top[sg.id],
                        "similar_distinct": rk.top_distinct[sg.id],
                    }
                    for i, sg in enumerate(subs)
                },
            }
            for name, rk in results.items()
        }
        payload["meta"]["methods"] = list(results)

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
        "methods": {name: [
            {"id": rk.ids[i], "x": round(float(rk.coords[i, 0]), 4),
             "y": round(float(rk.coords[i, 1]), 4), "cluster": int(rk.labels[i])}
            for i in range(len(rk.ids))
        ] for name, rk in results.items()},
    })
    out = cfg.dashboard_data / "similarity.json"
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
            if p.lq is not None:
                # El LQ es lo comparable entre años; los recuentos crudos no.
                rec["pois"]["lq"] = [round(float(v), 3) for v in p.lq]
                rec["pois"]["year"] = p.year
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



def cmd_benchmark(cfg, args) -> int:
    """Validación con datos sintéticos (Shiode & Shiode, 2020, §3 y §5).

    Siembra concentraciones en calles conocidas, se las pasa al extractor sin
    decirle dónde están, y mide cuánto de lo que encuentra es de verdad (PPV) y
    cuánto de lo que hay encuentra (sensibilidad).

    Es la única métrica **absoluta** del proyecto. Cobertura, densidad y
    ganancia sobre la línea base dicen que el extractor topológico captura más
    crimen que hacer crecer regiones a lo bruto; ninguna dice si acierta *dónde
    está* el hotspot, porque sobre datos reales no hay verdad conocida.
    """
    paths = Paths.from_config(cfg).ensure()
    from .baseline import extract_month_bfs
    from .density import KernelCache, density_field, to_csr
    from .hotspots import extract_month, segment
    from .ingest.network import load_network, to_undirected_simple
    from .selection import null_max_statistic
    from .synthetic import Benchmark, poisson_cluster, score, subnetwork

    t_all = time.perf_counter()
    P = _resolve_params(cfg, paths)
    if args.sigma:
        P["sigma_m"] = args.sigma
        P["radius_m"] = cfg.density.radius_for(args.sigma)

    with timed("Red vial", logger):
        G, net_info = load_network(cfg, force=False)
        g = to_csr(to_undirected_simple(G))
    logger.info("Grafo completo: %d nodos, %d aristas", g.n, g.indices.size // 2)

    rng = np.random.default_rng(args.seed)

    # Shiode & Shiode no corren el experimento sobre Buffalo entera sino sobre
    # un recorte con 394 puntos de referencia y 14 segmentos sembrados. Repetir
    # sus 300 puntos sobre una ciudad completa cambia el experimento: 30 nodos
    # sembrados entre 29 832 son el 0.1 % de la red, con un fondo tan diluido
    # que cualquier método acierta y la comparación no mide nada.
    if args.subnetwork:
        g = subnetwork(g, args.subnetwork, rng)
        logger.info("Recorte del experimento: %d nodos, %d aristas",
                    g.n, g.indices.size // 2)
    sel_cfg = cfg.hotspots.selection
    mc = sel_cfg.method == "montecarlo"

    # El fondo se reparte por toda la red, así que hacen falta kernels para
    # cualquier nodo: aquí no vale con los que tienen crimen observado.
    pool = np.arange(g.n, dtype=np.int64)
    with timed("Kernels", logger):
        kernels = KernelCache(g, P["sigma_m"], P["radius_m"]).build(pool)

    bench = Benchmark(params={
        "realisations": args.realisations,
        "parents": args.parents,
        "offspring": args.offspring,
        "background": args.background,
        "network": {"nodes": g.n, "edges": int(g.indices.size // 2),
                    "place": cfg.network.place, "subnetwork": args.subnetwork},
        "sigma_m": P["sigma_m"], "alpha": P["alpha"],
        "f_min_ratio": P["f_min_ratio"],
        "selection": sel_cfg.model_dump(),
        "seed": args.seed,
    })
    bench.methods = {"topologico": [], "voraz": [], "anchura": []}

    for r in range(args.realisations):
        truth = poisson_cluster(
            g, n_parents=args.parents, n_offspring=args.offspring,
            n_background=args.background, rng=rng,
        )
        c = truth.counts
        src = np.nonzero(c)[0]
        f = density_field(kernels, src, c[src], g.n)

        null_max = None
        if mc:
            null_max = null_max_statistic(
                truth.n_crimes, pool, c, kernels=kernels, g=g, segment_fn=segment,
                alpha=P["alpha"], f_min_ratio=P["f_min_ratio"],
                replicates=sel_cfg.replicates, null=sel_cfg.null, rng=rng,
            )
        hs, sel = extract_month(
            f"synth_{r:02d}", f, g, c, {"synthetic": c},
            alpha=P["alpha"], f_min_ratio=P["f_min_ratio"],
            selection=sel_cfg, null_max=null_max,
        )

        index = g.index
        topo_nodes = np.array(
            [index[n] for h in hs for n in h.nodes], dtype=np.int64)

        # Las líneas base reciben el mismo número de regiones y la misma huella
        # media que el extractor topológico. Sin igualar las dos cosas, el PPV
        # compararía métodos que marcan cantidades distintas de red y la
        # comparación no mediría acierto sino tamaño (§7.1).
        k = len(hs)
        budget = max(1, round(sum(h.n_nodes for h in hs) / k)) if k else 1
        by = {"topologico": topo_nodes}
        for name, growth in (("voraz", "greedy"), ("anchura", "bfs")):
            base = extract_month_bfs(f"synth_{r:02d}", g, c, {"synthetic": c},
                                     top_k=k, budget=budget, growth=growth)
            by[name] = np.array(
                [index[n] for h in base for n in h.nodes], dtype=np.int64)

        for name, nodes in by.items():
            bench.methods[name].append(score(nodes, truth))

        logger.info(
            "  realización %2d/%d: %d regiones (de %d candidatas), huella %d nodos"
            "  ·  PPV topo %.2f  sens %.2f",
            r + 1, args.realisations, k, sel.n_candidates, budget,
            bench.methods["topologico"][-1].ppv,
            bench.methods["topologico"][-1].sensitivity,
        )

    payload = bench.to_dict()
    payload["elapsed_s"] = round(time.perf_counter() - t_all, 2)
    out = paths.processed / "benchmark_synthetic.json"
    save_json(out, payload)
    dash = cfg.dashboard_data / "benchmark.json"
    save_json(dash, payload)
    logger.info("Escrito %s y %s", out, dash)

    print()
    print(_format_benchmark(payload))
    return 0


def _format_benchmark(b: dict) -> str:
    p = b["params"]
    out = [
        "=" * 74,
        "  VALIDACIÓN CON DATOS SINTÉTICOS  (Shiode & Shiode 2020, §3 y §5)",
        "=" * 74,
        f"  Red            {p['network']['place']} · {p['network']['nodes']:,} nodos",
        f"  Realizaciones  {b['realisations']}",
        f"  Siembra        {p['parents']} aristas · {p['offspring']:,} hechos en clúster"
        f" · {p['background']:,} de fondo",
        f"  Selección      {p['selection']['method']}",
        "",
        f"  {'Método':<14}{'PPV':>18}{'Sensibilidad':>20}{'F1':>14}{'Nodos':>9}",
        f"  {'':<14}{'media':>8}{'CV':>10}{'media':>10}{'CV':>10}{'media':>10}{'CV':>4}{'':>9}",
    ]
    for name, m in b["methods"].items():
        s = m["summary"]
        out.append(
            f"  {name:<14}{s['ppv']['mean']:>8.3f}{s['ppv']['cv']:>10.3f}"
            f"{s['sensitivity']['mean']:>10.3f}{s['sensitivity']['cv']:>10.3f}"
            f"{s['f1']['mean']:>10.3f}{s['f1']['cv']:>4.2f}"
            f"{s['detected_mean']:>9.0f}"
        )
    if b.get("tests"):
        out += ["", "  U de Mann-Whitney (dos colas) sobre F1"]
        for pair, row in b["tests"].items():
            f1 = row.get("f1") or {}
            if f1.get("p") is None:
                out.append(f"    {pair:<28} no definido")
            else:
                star = " *" if f1["p"] < 0.05 else ""
                out.append(f"    {pair:<28} U={f1['U']:>6.1f}  p={f1['p']:.4f}{star}")
    out += [
        "",
        "  PPV mide sobredisparo: de lo marcado, cuánto es hotspot de verdad.",
        "  Sensibilidad mide subdisparo: de lo que hay, cuánto se encuentra.",
        "  Las dos por separado se maximizan haciendo trampa —marcar toda la",
        "  ciudad da sensibilidad 1—, así que se leen juntas o vía F1.",
        "=" * 74,
    ]
    return "\n".join(out)



def cmd_casestudies(cfg, args) -> int:
    """Trayectorias y los tres estudios de caso de la nota de tesis.

    Produce `casestudies.json`, que es lo que alimenta las pestañas nuevas del
    dashboard: el plano Intensidad x Frecuencia, el plano Frecuencia x
    Estabilidad con sus cuatro cubos, el contraste topológico entre hotspots
    persistentes y episódicos, y los pares emparejados por forma.
    """
    paths = Paths.from_config(cfg).ensure()
    from .casestudies import (
        EXTRA_DIMS, contrast_groups, extra_descriptors, matched_pairs,
    )
    from .density import to_csr
    from .features.structural import (
        STRUCTURAL_DIMS, build_subgraphs, structural_matrix,
    )
    from .ingest.network import load_network, to_undirected_simple
    from .io import load_json
    from .trajectories import link, quadrants

    if not paths.hotspots.exists():
        logger.error("Falta %s. Ejecuta antes `crimepipe hotspots`.", paths.hotspots)
        return 1

    t_all = time.perf_counter()
    art = load_json(paths.hotspots)
    months = art["months"]
    hs = art["hotspots"]

    with timed("Red vial", logger):
        G, _ = load_network(cfg)
        g = to_csr(to_undirected_simple(G))

    # ── Estudio 1 · trayectorias ──────────────────────────────────────────
    with timed("Enlace de trayectorias", logger):
        trajs = link(hs, months, min_iou=args.min_iou, max_gap=args.max_gap,
                     g=g, cutoff_m=args.movement_cutoff_m)
    q = quadrants(trajs, len(months))
    multi = [t for t in trajs if t.k > 1]
    logger.info(
        "Trayectorias: %d (%d con más de una aparición, %d de un solo mes)",
        len(trajs), len(multi), len(trajs) - len(multi))
    if multi:
        logger.info("  frecuencia media %.3f · estabilidad media %.3f · "
                    "desplazamiento medio %.0f m",
                    float(np.mean([t.frequency(len(months)) for t in multi])),
                    float(np.mean([t.stability for t in multi])),
                    float(np.mean([t.movement_m for t in multi])))

    # ── Estudio 2 · topología de persistentes vs episódicos ───────────────
    with timed("Construcción de subgrafos", logger):
        subs = build_subgraphs(art, G)
    S = structural_matrix(subs)
    with timed(f"Descriptores extra ({', '.join(EXTRA_DIMS)})", logger):
        E = extra_descriptors(subs)
    X = np.hstack([S, E])
    names = list(STRUCTURAL_DIMS) + list(EXTRA_DIMS)

    # La frecuencia es una propiedad de la trayectoria; se hereda a cada uno de
    # sus subgrafos para poder contrastar descriptores, que son por subgrafo.
    freq_of: dict[str, float] = {}
    for t in trajs:
        f = t.frequency(len(months))
        for mid in t.members:
            freq_of[mid] = f
    freq = np.array([freq_of.get(sg.id, 0.0) for sg in subs])

    # Terciles y no mediana: comparar el tercio de arriba con el de abajo deja
    # fuera la franja ambigua del medio, donde «persistente» y «episódico» no
    # se distinguen y solo añadirían ruido al contraste.
    lo_q, hi_q = np.quantile(freq, [1 / 3, 2 / 3])
    persistent = freq >= hi_q
    episodic = freq <= lo_q
    logger.info("Contraste topológico: %d persistentes (f>=%.3f) vs %d "
                "episódicos (f<=%.3f)",
                int(persistent.sum()), hi_q, int(episodic.sum()), lo_q)
    topo = contrast_groups(X, names, persistent, episodic)

    # Mismo contraste sobre los POIs, que es lo que la nota pide a continuación.
    poi_rows, poi_meta = [], None
    sim_path = cfg.dashboard_data / "similarity.json"
    if sim_path.exists():
        sim = load_json(sim_path)
        poi_meta = sim.get("meta", {}).get("pois")
        if poi_meta:
            by_id = {h["id"]: h for h in sim["hotspots"]}
            cats = poi_meta["categories"]
            # El LQ es lo comparable entre años; los recuentos crudos no, porque
            # OSM absorbió un padrón escolar entero a mitad de la ventana.
            has_lq = any((by_id.get(sg.id, {}).get("pois") or {}).get("lq")
                         for sg in subs)
            key = "lq" if has_lq else "counts"
            P = np.array([
                (by_id.get(sg.id, {}).get("pois") or {}).get(key, [0] * len(cats))
                for sg in subs], dtype=np.float64)
            if not has_lq:
                # Sin LQ hay que normalizar igualmente. Los persistentes son
                # más grandes —`log_n_nodes` los separa— así que contrastar
                # recuentos crudos mediría tamaño y no función, que es
                # exactamente lo que §7.1 advierte que no mide nada. Se pasa a
                # proporciones sobre el total de la propia zona.
                tot = P.sum(axis=1, keepdims=True)
                P = np.divide(P, tot, out=np.zeros_like(P), where=tot > 0)
                key = "share"
            extra = np.array([
                [(by_id.get(sg.id, {}).get("pois") or {}).get("entropy_norm", 0.0),
                 (by_id.get(sg.id, {}).get("pois") or {}).get("per_node", 0.0)]
                for sg in subs], dtype=np.float64)
            poi_rows = contrast_groups(
                np.hstack([P, extra]),
                [f"{key}:{c}" for c in cats] + ["entropy_norm", "per_node"],
                persistent, episodic)
            logger.info("Contraste de POIs sobre %s (%d columnas)", key, P.shape[1])
    else:
        logger.info("Sin similarity.json: el contraste de POIs se omite. "
                    "Ejecuta `crimepipe features`.")

    # ── Estudio 3 · matching ──────────────────────────────────────────────
    crimes = np.array([h["crimes"] for h in hs], dtype=np.float64)
    order = {h["id"]: i for i, h in enumerate(hs)}
    crimes_sub = np.array([crimes[order[sg.id]] for sg in subs])
    with timed("Emparejamiento por forma", logger):
        pairs = matched_pairs(S, crimes_sub, [sg.id for sg in subs],
                              [sg.month for sg in subs],
                              [set(sg.nodes) for sg in subs],
                              n_pairs=args.pairs, min_ratio=args.min_ratio)
    logger.info("Pares con topología casi igual y crimen >=%.0fx: %d",
                args.min_ratio, len(pairs))

    payload = {
        "meta": {
            "months": months,
            "n_months": len(months),
            "hotspots": len(hs),
            "min_iou": args.min_iou,
            "max_gap": args.max_gap,
            "movement_cutoff_m": args.movement_cutoff_m,
            "freq_terciles": [round(float(lo_q), 4), round(float(hi_q), 4)],
            "n_persistent": int(persistent.sum()),
            "n_episodic": int(episodic.sum()),
            "dims": names,
            "poi_meta": poi_meta,
            "elapsed_s": round(time.perf_counter() - t_all, 1),
        },
        "trajectories": [t.to_dict(len(months)) for t in trajs],
        "quadrants": q,
        "topology_contrast": [r.to_dict() for r in topo],
        "poi_contrast": [r.to_dict() for r in poi_rows],
        "matched_pairs": pairs,
    }
    save_json(paths.processed / "casestudies.json", payload, indent=2)
    out = cfg.dashboard_data / "casestudies.json"
    save_json(out, payload)
    logger.info("Escrito %s y %s", paths.processed / "casestudies.json", out)

    print()
    print(_format_casestudies(payload))
    return 0


def _format_casestudies(d: dict) -> str:
    m = d["meta"]
    q = d["quadrants"]
    tr = d["trajectories"]
    multi = [t for t in tr if t["k"] > 1]
    out = [
        "=" * 74,
        "  ESTUDIOS DE CASO",
        "=" * 74,
        f"  Ventana        {m['n_months']} meses · {m['hotspots']:,} subgrafos",
        f"  Enlace         IoU >= {m['min_iou']} · hueco máximo {m['max_gap']} meses",
        "",
        "  1 · ¿Persistencia implica estabilidad espacial?",
        f"     trayectorias            {len(tr):,}",
        f"     con más de un mes       {len(multi):,}",
        f"     de un solo mes          {len(tr) - len(multi):,}",
    ]
    if q.get("counts"):
        out.append(f"     cortes: frecuencia {q['freq_split']} · "
                   f"estabilidad {q['stab_split']}")
        for k in ("frecuente_estable", "frecuente_movil",
                  "episodico_estable", "episodico_movil"):
            out.append(f"       {k:<22}{q['counts'].get(k, 0):>7,}")

    out += ["", "  2 · ¿Los persistentes tienen otra topología?",
            f"     {m['n_persistent']:,} persistentes vs {m['n_episodic']:,} episódicos",
            f"     {'descriptor':<24}{'persist.':>10}{'episód.':>10}{'δ':>8}{'p aj.':>10}"]
    for r in d["topology_contrast"][:8]:
        pa = "—" if r["p_adj"] is None else f"{r['p_adj']:.4f}"
        out.append(f"     {r['name']:<24}{r['mean_persistent']:>10.3f}"
                   f"{r['mean_episodic']:>10.3f}{r['delta']:>8.3f}{pa:>10}")
    out.append("     δ es la delta de Cliff: <0.15 despreciable, <0.33 pequeño.")

    if d["poi_contrast"]:
        out += ["", "     Y en POIs:",
                f"     {'categoría':<24}{'persist.':>10}{'episód.':>10}{'δ':>8}{'p aj.':>10}"]
        for r in d["poi_contrast"][:6]:
            pa = "—" if r["p_adj"] is None else f"{r['p_adj']:.4f}"
            out.append(f"     {r['name']:<24}{r['mean_persistent']:>10.3f}"
                       f"{r['mean_episodic']:>10.3f}{r['delta']:>8.3f}{pa:>10}")

    out += ["", "  3 · Matching: misma forma, muy distinto crimen",
            f"     pares encontrados       {len(d['matched_pairs']):,}"]
    for r in d["matched_pairs"][:5]:
        out.append(f"       {r['high']} ({r['crimes_high']:,}) vs "
                   f"{r['low']} ({r['crimes_low']:,})  ×{r['ratio']}")
    out.append("=" * 74)
    return "\n".join(out)


COMMANDS = {
    "ingest": cmd_ingest,
    "casestudies": cmd_casestudies,
    "benchmark": cmd_benchmark,
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
    p.add_argument(
        "--dataset",
        default=None,
        help="ciudad a procesar (bloque de `datasets:` en config.yaml). "
             "Por defecto, `active_dataset`.",
    )
    p.add_argument("--force", action="store_true", help="ignora la caché")
    p.add_argument("--sweep-months", type=int, default=None,
                   help="submuestra regular de meses para `calibrate` "
                        "(por defecto, todos)")
    c = p.add_argument_group("estudios de caso", "solo para `crimepipe casestudies`")
    c.add_argument("--min-iou", type=float, default=0.2,
                   help="solape mínimo para considerar dos subgrafos el mismo sitio")
    c.add_argument("--max-gap", type=int, default=3,
                   help="meses que una trayectoria sobrevive sin aparecer")
    c.add_argument("--movement-cutoff-m", type=float, default=3000.0,
                   help="tope del Dijkstra al medir desplazamiento de la semilla")
    c.add_argument("--pairs", type=int, default=25,
                   help="pares a devolver en el estudio de matching")
    c.add_argument("--min-ratio", type=float, default=3.0,
                   help="cociente mínimo de crimen entre los dos del par")
    b = p.add_argument_group("benchmark", "solo para `crimepipe benchmark`")
    b.add_argument("--realisations", type=int, default=10,
                   help="realizaciones sintéticas (Shiode usa 10)")
    b.add_argument("--parents", type=int, default=15,
                   help="aristas donde se siembra concentración")
    b.add_argument("--offspring", type=int, default=200,
                   help="hechos repartidos entre esas aristas")
    b.add_argument("--background", type=int, default=100,
                   help="hechos uniformes sobre el resto de la red")
    b.add_argument("--subnetwork", type=int, default=400,
                   help="recorta la red a N nodos conexos antes de sembrar. "
                        "0 = ciudad entera, que hace el problema trivial")
    b.add_argument("--sigma", type=float, default=None,
                   help="sobrescribe σ, para barrer la resolución del kernel")
    b.add_argument("--seed", type=int, default=20200,
                   help="semilla del generador sintético")
    p.add_argument(
        "--no-pois",
        action="store_true",
        help="no descargar POIs en `features`/`export`. Los POIs son el único "
             "paso que depende de un servicio ajeno (Overpass) y el único que "
             "puede tardar media hora o fallar por cuota; con esta bandera el "
             "resto del cálculo no queda detrás de él. Se añaden después con "
             "`crimepipe pois` y reejecutando `features` y `export`.",
    )
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args(argv)

    # La consola de Windows suele venir en cp1252 y los resúmenes llevan
    # caracteres Unicode (§, →, ─). Sin esto, un `print` los rompería.
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s  %(levelname)-7s %(name)-22s %(message)s",
        stream=sys.stdout,
    )
    cfg = load_config(args.config, args.dataset)
    return COMMANDS[args.command](cfg, args)


if __name__ == "__main__":
    raise SystemExit(main())
