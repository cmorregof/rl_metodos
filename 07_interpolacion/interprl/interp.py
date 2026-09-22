"""Interpolación polinómica en [−1, 1]: nodos, fórmula baricéntrica y
constante de Lebesgue. Es la parte "numérica pura" del proyecto y también
el verificador de la fase 3.

* `nodes(n, gamma)`: n + 1 nodos que interpolan entre equiespaciados
  (γ = 0) y Chebyshev–Lobatto (γ = 1), ambos con los extremos ±1.
* `barycentric_weights`, `interpolate`: fórmula baricéntrica (Berrut y
  Trefethen, 2004), estable incluso con muchos nodos.
* `lebesgue_constant(x)`: Λ = máx Σ|ℓᵢ(x)|. Es el verificador independiente
  de la fase 3: dado un conjunto de nodos, dice cuánto puede amplificar la
  interpolación el error de los datos, sin saber nada de cómo se eligieron.
"""

from __future__ import annotations

import math
from typing import Sequence


def equispaced(n: int) -> list[float]:
    return [-1.0 + 2.0 * i / n for i in range(n + 1)]


def chebyshev_lobatto(n: int) -> list[float]:
    """Extremos de T_n: xᵢ = −cos(iπ/n). Incluyen ±1."""
    return [-math.cos(math.pi * i / n) for i in range(n + 1)]


def chebyshev_first_kind(n: int) -> list[float]:
    """Ceros de T_{n+1} (los del teorema de minimalidad de máx|w|)."""
    return [-math.cos(math.pi * (2 * i + 1) / (2 * n + 2)) for i in range(n + 1)]


def extended_chebyshev(n: int) -> list[float]:
    """Ceros de T_{n+1} estirados para que los extremos caigan en ±1."""
    z = chebyshev_first_kind(n)
    s = 1.0 / math.cos(math.pi / (2 * n + 2))
    return [x * s for x in z]


def nodes(n: int, gamma: float) -> list[float]:
    """Mezcla convexa entre equiespaciados (γ = 0) y Chebyshev–Lobatto (γ = 1)."""
    u, c = equispaced(n), chebyshev_lobatto(n)
    return [(1 - gamma) * a + gamma * b for a, b in zip(u, c)]


def barycentric_weights(x: Sequence[float]) -> list[float]:
    w = []
    for i, xi in enumerate(x):
        p = 1.0
        for j, xj in enumerate(x):
            if j != i:
                p *= xi - xj
        w.append(1.0 / p)
    return w


def interpolate(x: Sequence[float], y: Sequence[float], w: Sequence[float], t: float) -> float:
    num = den = 0.0
    for xi, yi, wi in zip(x, y, w):
        d = t - xi
        if d == 0.0:
            return yi
        q = wi / d
        num += q * yi
        den += q
    return num / den


def lebesgue_function(x: Sequence[float], w: Sequence[float], t: float) -> float:
    """λ(t) = Σ|ℓᵢ(t)| con ℓᵢ en forma baricéntrica."""
    den = 0.0
    terms = []
    for xi, wi in zip(x, w):
        d = t - xi
        if d == 0.0:
            return 1.0
        q = wi / d
        terms.append(q)
        den += q
    return sum(abs(q) for q in terms) / abs(den)


def _golden_max(f, a: float, b: float, iters: int = 60) -> tuple[float, float]:
    g = (math.sqrt(5) - 1) / 2
    c, d = b - g * (b - a), a + g * (b - a)
    fc, fd = f(c), f(d)
    for _ in range(iters):
        if fc < fd:
            a, c, fc = c, d, fd
            d = a + g * (b - a)
            fd = f(d)
        else:
            b, d, fd = d, c, fc
            c = b - g * (b - a)
            fc = f(c)
    t = 0.5 * (a + b)
    return t, f(t)


def lebesgue_constant(x: Sequence[float], samples: int = 64) -> tuple[float, float]:
    """(Λ, t*) con Λ = máx_{t∈[−1,1]} λ(t).

    En cada subintervalo entre nodos consecutivos λ es suave y tiene un único
    máximo local (Luttmann–Rivlin); se localiza con una malla y se refina con
    sección áurea. El resultado es reproducible a ~1e-12 con dos mallas
    distintas, que es lo que comprueba `verify`.
    """
    xs = sorted(x)
    w = barycentric_weights(xs)
    f = lambda t: lebesgue_function(xs, w, t)
    best_t, best = -1.0, 1.0
    cuts = sorted({-1.0, 1.0, *xs})
    for a, b in zip(cuts[:-1], cuts[1:]):
        grid = [a + (b - a) * (k + 0.5) / samples for k in range(samples)]
        vals = [f(t) for t in grid]
        k = max(range(samples), key=vals.__getitem__)
        lo = grid[k - 1] if k > 0 else a
        hi = grid[k + 1] if k < samples - 1 else b
        t, v = _golden_max(f, lo, hi)
        if v > best:
            best_t, best = t, v
    return best, best_t


def max_abs_nodal_poly(x: Sequence[float], samples: int = 2001) -> float:
    """máx |w(t)| con w(t) = ∏(t − xᵢ), por muestreo fino (para el informe)."""
    best = 0.0
    for k in range(samples):
        t = -1.0 + 2.0 * k / (samples - 1)
        p = 1.0
        for xi in x:
            p *= t - xi
        best = max(best, abs(p))
    return best
