"""Capa numérica: búsqueda del programa de pasos óptimo a horizonte fijo n.

El problema minₕ τ(h), con τ(h) el peor caso certificado por el PEP, es no
convexo en h. Aquí se ataca con el método de entropía cruzada (una política
gaussiana sobre h que se reestima con la élite: la versión más simple de
RL de política continua) seguido de un refinamiento local con Nelder–Mead.
Sirve para reproducir los óptimos numéricos conocidos (n ≤ 10) y como
línea base contra la que medir lo que proponga la capa simbólica.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import minimize

from .pep import gd_worst_case

warnings.filterwarnings("ignore", category=UserWarning)


@dataclass
class SearchResult:
    n: int
    h: np.ndarray
    value: float
    history: list[float] = field(default_factory=list)  # mejor valor por generación
    evaluations: int = 0


def _tau(h, objective="fval") -> float:
    if np.any(h <= 0) or np.any(h > 20):
        return 1.0
    try:
        return gd_worst_case(h, objective=objective).value
    except Exception:
        return 1.0


def cross_entropy(
    n: int,
    generations: int = 25,
    population: int = 40,
    elite_frac: float = 0.25,
    init_mean: float | np.ndarray = 1.5,
    init_std: float = 0.6,
    seed: int = 0,
    objective: str = "fval",
    refine: bool = True,
    verbose: bool = False,
) -> SearchResult:
    rng = np.random.default_rng(seed)
    mean = np.full(n, init_mean, dtype=float) if np.isscalar(init_mean) else np.asarray(init_mean, dtype=float).copy()
    std = np.full(n, init_std, dtype=float)
    n_elite = max(2, int(elite_frac * population))
    best_h, best_v = mean.copy(), _tau(mean, objective)
    history = []
    evals = 1
    for g in range(generations):
        pop = rng.normal(mean, std, size=(population, n))
        pop = np.clip(pop, 0.05, 20.0)
        vals = np.array([_tau(h, objective) for h in pop])
        evals += population
        order = np.argsort(vals)
        elite = pop[order[:n_elite]]
        if vals[order[0]] < best_v:
            best_v, best_h = float(vals[order[0]]), pop[order[0]].copy()
        mean = elite.mean(axis=0)
        std = np.maximum(elite.std(axis=0), 1e-3)
        history.append(best_v)
        if verbose:
            print(f"gen {g:3d}: mejor {best_v:.6f}  media élite {vals[order[:n_elite]].mean():.6f}  std {std.mean():.3f}")
        if std.max() < 1e-4:
            break
    if refine:
        r = minimize(lambda h: _tau(h, objective), best_h, method="Nelder-Mead", options={"xatol": 1e-6, "fatol": 1e-9, "maxiter": 400 * n})
        evals += r.nfev
        if r.fun < best_v:
            best_v, best_h = float(r.fun), np.asarray(r.x)
        history.append(best_v)
    return SearchResult(n=n, h=best_h, value=best_v, history=history, evaluations=evals)
