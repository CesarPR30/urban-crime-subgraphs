"""Calibración automática de σ y f_min a partir de los datos (§4.1, §4.3).

## El problema

Los valores de σ y `f_min` que usa el pipeline se fijaron contrastando contra
los agregados de §7.3, que vienen de una corrida anterior sobre Chicago
2024-2025. Eso **no generaliza**: con otro dataset —otra ciudad, otra ventana,
otra densidad de red— esos números de referencia no existen, y calibrar contra
ellos sería ajustar a una respuesta que no se tiene.

## El criterio que sí generaliza

Existe un trade-off estructural que no depende de conocer la respuesta:

* aflojar los parámetros (σ grande, `f_min` bajo) **sube la cobertura** y
  **hunde la densidad**: las regiones se derraman y dejan de señalar nada;
* apretarlos hace lo contrario: regiones diminutas, densísimas, que cubren una
  fracción irrelevante del crimen.

Entre esos dos extremos hay una **rodilla**: el punto a partir del cual cada
punto extra de cobertura cuesta mucha más densidad de la que costaba antes. Esa
rodilla es una propiedad del dataset, no de una respuesta conocida, y es lo que
esta calibración localiza.

El método es el clásico de máxima distancia a la cuerda (*kneedle*):

1. Se calcula la **frontera de Pareto** en (cobertura ↑, densidad ↑): las
   configuraciones donde no se puede ganar una sin perder la otra.
2. Se normalizan ambos ejes a [0, 1] sobre el rango de la frontera.
3. Se traza la cuerda entre sus dos extremos y se toma el punto con **máxima
   distancia perpendicular** a esa cuerda, por el lado bueno.

.. math::
    d(p) = \\frac{(y_1-y_0)(x_p-x_0) - (x_1-x_0)(y_p-y_0)}
                 {\\sqrt{(x_1-x_0)^2 + (y_1-y_0)^2}}

Una frontera perfectamente recta no tiene rodilla y el método lo dice
(`curvature` ≈ 0) en vez de inventar un óptimo.

`K` **no se calibra**: no es un parámetro estadístico sino una decisión de
alcance —cuántas zonas se quieren estudiar al mes—, y además no degrada la
densidad al subir, así que su «óptimo» sería siempre el máximo del rango.
"""

from __future__ import annotations

import itertools
import logging
import math
import time
from dataclasses import asdict, dataclass, field

import numpy as np

from .density import KernelCache, density_field, to_csr
from .hotspots import extract_month

logger = logging.getLogger(__name__)

#: Malla por defecto. Se cubre desde «regiones diminutas» hasta «media ciudad»
#: para que la frontera tenga los dos extremos y la rodilla quede dentro.
DEFAULT_SIGMAS = [60.0, 90.0, 120.0, 150.0, 200.0, 250.0]
DEFAULT_ALPHAS = [0.1, 0.2, 0.3, 0.4, 0.5]
DEFAULT_FMINS = [0.05, 0.075, 0.10, 0.125, 0.15, 0.20]
DEFAULT_KS = [5, 10, 20, 30, 50, 100]


@dataclass
class SweepPoint:
    """Una combinación evaluada con las cuatro métricas de §7.1."""

    sigma: float
    alpha: float
    f_min: float
    k: int
    captured: int
    coverage: float
    nodes: int
    nodes_per_month: float
    density: float
    lift: float
    hotspots: int
    pareto: bool = False

    def config(self) -> str:
        return f"σ={self.sigma:g} · α={self.alpha:g} · f_min={self.f_min:g}"


@dataclass
class Calibration:
    """Resultado de la calibración automática."""

    sigma_m: float
    f_min_ratio: float
    alpha: float
    curvature: float
    knee: SweepPoint
    front: list[SweepPoint] = field(default_factory=list)
    grid: list[SweepPoint] = field(default_factory=list)
    k_sweep: list[SweepPoint] = field(default_factory=list)
    alpha_sensitivity: float = 0.0
    meta: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "recommended": {
                "sigma_m": self.sigma_m,
                "f_min_ratio": self.f_min_ratio,
                "alpha": self.alpha,
            },
            "curvature": round(self.curvature, 4),
            "knee": asdict(self.knee),
            "alpha_sensitivity": round(self.alpha_sensitivity, 4),
            "front": [asdict(p) for p in self.front],
            "grid": [asdict(p) for p in self.grid],
            "k_sweep": [asdict(p) for p in self.k_sweep],
            "meta": self.meta,
        }


# --------------------------------------------------------------------------- #
# Barrido
# --------------------------------------------------------------------------- #


def evaluate_combo(months, fields, g, counts, cat_counts, alpha, f_min, k):
    """Agrega las métricas de §7.1 para una combinación, sobre los 24 meses."""
    cap = nodes = n_hs = 0
    for m in months:
        hs, _ = extract_month(m, fields[m], g, counts[m], cat_counts[m],
                           alpha=alpha, f_min_ratio=f_min, top_k=k)
        cap += sum(h.crimes for h in hs)
        nodes += sum(h.n_nodes for h in hs)
        n_hs += len(hs)
    return cap, nodes, n_hs


def run_sweep(
    g, counts, cat_counts, months, *,
    sigmas=None, alphas=None, fmins=None, ks=None,
    truncate_sigmas: float = 3.0, base=None, progress=None,
) -> tuple[list[SweepPoint], list[SweepPoint], dict]:
    """Recorre la malla σ × α × f_min y, aparte, un barrido de K.

    Los kernels gaussianos dependen **solo de σ**, así que se construyen una vez
    por valor de σ y se reutilizan en todas las combinaciones de α y f_min que
    cuelgan de él. Sin eso el barrido sería inviable.
    """
    sigmas = sigmas or DEFAULT_SIGMAS
    alphas = alphas or DEFAULT_ALPHAS
    fmins = fmins or DEFAULT_FMINS
    ks = ks or DEFAULT_KS
    t0 = time.perf_counter()

    total = sum(int(counts[m].sum()) for m in months)
    city_mean = total / len(months) / g.n if months else 0.0
    src_all = np.unique(np.concatenate([np.nonzero(counts[m])[0] for m in months]))

    grid: list[SweepPoint] = []
    fields_by_sigma: dict[float, dict] = {}

    for sigma in sigmas:
        radius = sigma * truncate_sigmas
        logger.info("σ = %.0f m (r = %.0f m): construyendo kernels…", sigma, radius)
        kern = KernelCache(g, sigma, radius).build(src_all)
        fields = {}
        for m in months:
            src = np.nonzero(counts[m])[0]
            fields[m] = density_field(kern, src, counts[m][src], g.n)
        fields_by_sigma[sigma] = fields

        for alpha, f_min in itertools.product(alphas, fmins):
            cap, nodes, n_hs = evaluate_combo(
                months, fields, g, counts, cat_counts, alpha, f_min,
                (base or {}).get("k", 20))
            dens = cap / nodes if nodes else 0.0
            grid.append(SweepPoint(
                sigma=sigma, alpha=alpha, f_min=f_min, k=(base or {}).get("k", 20),
                captured=cap, coverage=round(cap / total, 5) if total else 0.0,
                nodes=nodes, nodes_per_month=round(nodes / len(months), 1),
                density=round(dens, 4),
                lift=round(dens / city_mean, 3) if city_mean else 0.0,
                hotspots=n_hs,
            ))
        if progress:
            progress(sigma, time.perf_counter() - t0)

    # Barrido de K sobre σ y f_min base (o los primeros de la malla).
    k_sigma = (base or {}).get("sigma", sigmas[len(sigmas) // 2])
    k_alpha = (base or {}).get("alpha", alphas[len(alphas) // 2])
    k_fmin = (base or {}).get("f_min", fmins[len(fmins) // 2])
    k_sweep: list[SweepPoint] = []
    fields = fields_by_sigma.get(k_sigma)
    if fields is not None:
        for k in ks:
            cap, nodes, n_hs = evaluate_combo(
                months, fields, g, counts, cat_counts, k_alpha, k_fmin, k)
            dens = cap / nodes if nodes else 0.0
            k_sweep.append(SweepPoint(
                sigma=k_sigma, alpha=k_alpha, f_min=k_fmin, k=k,
                captured=cap, coverage=round(cap / total, 5) if total else 0.0,
                nodes=nodes, nodes_per_month=round(nodes / len(months), 1),
                density=round(dens, 4),
                lift=round(dens / city_mean, 3) if city_mean else 0.0,
                hotspots=n_hs,
            ))

    meta = {
        "crimes_total": total,
        "network_nodes": g.n,
        "months": len(months),
        "city_mean_density": round(city_mean, 4),
        "truncate_sigmas": truncate_sigmas,
        "axes": {"sigma": sigmas, "alpha": alphas, "f_min": fmins, "k": ks},
        "elapsed_s": round(time.perf_counter() - t0, 1),
    }
    return grid, k_sweep, meta


# --------------------------------------------------------------------------- #
# Frontera y rodilla
# --------------------------------------------------------------------------- #


def pareto_front(points: list[SweepPoint]) -> list[SweepPoint]:
    """Puntos no dominados en (cobertura ↑, densidad ↑).

    `a` domina a `b` si es al menos igual en ambas métricas y estrictamente
    mejor en una. La frontera es donde no se puede ganar cobertura sin perder
    densidad — el conjunto donde la elección es genuinamente un trade-off y no
    una mejora gratis.
    """
    front = []
    for a in points:
        dominated = any(
            b.coverage >= a.coverage and b.density >= a.density
            and (b.coverage > a.coverage or b.density > a.density)
            for b in points if b is not a
        )
        a.pareto = not dominated
        if not dominated:
            front.append(a)
    return sorted(front, key=lambda p: p.coverage)


def knee_point(front: list[SweepPoint]) -> tuple[SweepPoint, float]:
    """Rodilla de la frontera por máxima distancia a la cuerda (*kneedle*).

    Devuelve `(punto, curvatura)`. La curvatura es la distancia normalizada
    máxima: 0 significa frontera recta —sin rodilla, la elección es puramente
    de preferencia— y valores altos, un codo marcado.
    """
    if len(front) < 3:
        return (front[-1] if front else None), 0.0

    xs = np.array([p.coverage for p in front], dtype=float)
    ys = np.array([p.density for p in front], dtype=float)
    # Normalizar: los dos ejes tienen unidades incomparables (fracción vs
    # crímenes por nodo), así que sin esto la «distancia» no significaría nada.
    xr, yr = xs.max() - xs.min(), ys.max() - ys.min()
    if xr <= 0 or yr <= 0:
        return front[len(front) // 2], 0.0
    xn = (xs - xs.min()) / xr
    yn = (ys - ys.min()) / yr

    x0, y0, x1, y1 = xn[0], yn[0], xn[-1], yn[-1]
    denom = math.hypot(x1 - x0, y1 - y0)
    if denom <= 0:
        return front[len(front) // 2], 0.0
    # Distancia con signo: positiva por encima de la cuerda, que es el lado
    # donde ambas métricas son mejores de lo que la interpolación lineal daría.
    d = ((y1 - y0) * (xn - x0) - (x1 - x0) * (yn - y0)) / denom
    i = int(np.argmax(-d)) if (-d).max() > d.max() else int(np.argmax(d))
    return front[i], float(abs(d[i]))


def alpha_sensitivity(grid: list[SweepPoint], sigma: float, f_min: float) -> float:
    """Rango relativo de cobertura al variar α con σ y f_min fijos.

    Cerca de 0 significa que α no está haciendo nada: con el piso puesto alto,
    el barrido solo recorre nodos densos donde apenas quedan máximos espurios
    que simplificar.
    """
    rows = [p for p in grid if p.sigma == sigma and p.f_min == f_min]
    if len(rows) < 2:
        return 0.0
    covs = [p.coverage for p in rows]
    lo, hi = min(covs), max(covs)
    return (hi - lo) / hi if hi else 0.0


def calibrate(
    g, counts, cat_counts, months, *,
    sigmas=None, alphas=None, fmins=None, ks=None,
    truncate_sigmas: float = 3.0, base=None, progress=None,
) -> Calibration:
    """Ejecuta el barrido y devuelve los parámetros recomendados."""
    grid, k_sweep, meta = run_sweep(
        g, counts, cat_counts, months,
        sigmas=sigmas, alphas=alphas, fmins=fmins, ks=ks,
        truncate_sigmas=truncate_sigmas, base=base, progress=progress,
    )
    front = pareto_front(grid)
    knee, curvature = knee_point(front)
    if knee is None:
        raise ValueError("El barrido no produjo ninguna configuración viable.")

    sens = alpha_sensitivity(grid, knee.sigma, knee.f_min)
    meta["pareto_size"] = len(front)
    logger.info(
        "Rodilla: %s -> %.1f %% cobertura, %.2f cr/nodo, %.0f nodos/mes "
        "(curvatura %.3f, α mueve la cobertura un %.1f %%)",
        knee.config(), 100 * knee.coverage, knee.density,
        knee.nodes_per_month, curvature, 100 * sens,
    )
    return Calibration(
        sigma_m=knee.sigma, f_min_ratio=knee.f_min, alpha=knee.alpha,
        curvature=curvature, knee=knee, front=front, grid=grid,
        k_sweep=k_sweep, alpha_sensitivity=sens, meta=meta,
    )
