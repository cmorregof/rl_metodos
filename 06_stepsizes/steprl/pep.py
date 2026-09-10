"""Verificador: el *performance estimation problem* (PEP) del descenso de gradiente.

Dada una sucesión de pasos h = (h₁, …, hₙ) (normalizados por L), el descenso
de gradiente xₖ₊₁ = xₖ − (hₖ/L)·∇f(xₖ) sobre una f convexa y L-suave arbitraria
con ‖x₀ − x*‖ ≤ R tiene un **peor caso exacto** para f(xₙ) − f* (o ‖∇f(xₙ)‖²),
que es el valor de un programa semidefinido (Drori–Teboulle 2014; Taylor,
Hendrickx y Glineur 2017): las funciones L-suaves convexas se caracterizan
por las desigualdades de interpolación

    fᵢ ≥ fⱼ + ⟨gⱼ, xᵢ − xⱼ⟩ + (1/2L)·‖gᵢ − gⱼ‖²   para todo par i ≠ j,

y basta trabajar con la matriz de Gram de (x₀, g₀, …, gₙ) y el vector de
valores (f₀, …, fₙ) con x* = 0, g* = 0, f* = 0.

El valor del SDP es un **certificado**: el dual (multiplicadores λᵢⱼ ≥ 0 y
τ ≥ 0) da una demostración de que f(xₙ) − f* ≤ τ·L·R² combinando las
desigualdades de interpolación; el primal da la función que alcanza el peor
caso. Aquí el verificador es la recompensa del agente y el dual, la prueba.
"""

from __future__ import annotations

from dataclasses import dataclass

import cvxpy as cp
import numpy as np

SILVER_RATIO = 1 + np.sqrt(2)


@dataclass
class PEPResult:
    value: float  # peor caso de la métrica (con L = R = 1 es la tasa τ)
    h: tuple[float, ...]
    objective: str
    dual: np.ndarray | None = None  # multiplicadores λᵢⱼ de las desigualdades de interpolación
    tau_dual: float | None = None  # multiplicador de la condición inicial
    gram: np.ndarray | None = None  # matriz de Gram del peor caso (primal)
    fvals: np.ndarray | None = None  # valores f₀…fₙ del peor caso
    status: str = ""


def _vectors(h: np.ndarray, L: float) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Coordenadas de x₀…xₙ y g₀…gₙ en la base (x₀, g₀, …, gₙ); x* = g* = 0."""
    n = len(h)
    dim = n + 2
    xs, gs = [], []
    x = np.zeros(dim)
    x[0] = 1.0
    for i in range(n + 1):
        g = np.zeros(dim)
        g[i + 1] = 1.0
        xs.append(x.copy())
        gs.append(g)
        if i < n:
            x = x - (h[i] / L) * g
    # el punto estrella
    xs.append(np.zeros(dim))
    gs.append(np.zeros(dim))
    return xs, gs


def gd_worst_case(
    h,
    L: float = 1.0,
    R: float = 1.0,
    objective: str = "fval",
    solver: str = "CLARABEL",
    want_certificate: bool = False,
) -> PEPResult:
    """Peor caso de f(xₙ) − f* (objective="fval") o ‖∇f(xₙ)‖² ("gradnorm")
    del descenso de gradiente con pasos h sobre F₀,L con ‖x₀ − x*‖ ≤ R."""
    h = np.asarray(h, dtype=float)
    n = len(h)
    dim = n + 2
    xs, gs = _vectors(h, L)
    star = n + 1
    G = cp.Variable((dim, dim), PSD=True)
    F = cp.Variable(n + 1)  # f₀…fₙ; f* = 0

    def fv(i):
        return 0.0 if i == star else F[i]

    def inner(u, v):
        return cp.sum(cp.multiply(np.outer(u, v), G))

    constraints = []
    pairs = []
    for i in range(dim):
        for j in range(dim):
            if i == j:
                continue
            d = gs[i] - gs[j]
            constraints.append(fv(i) >= fv(j) + inner(gs[j], xs[i] - xs[j]) + inner(d, d) / (2 * L))
            pairs.append((i, j))
    init = inner(xs[0], xs[0]) <= R * R
    constraints.append(init)
    if objective == "fval":
        obj = F[n]
    elif objective == "gradnorm":
        obj = inner(gs[n], gs[n])
    else:
        raise ValueError(objective)
    prob = cp.Problem(cp.Maximize(obj), constraints)
    prob.solve(solver=solver)
    res = PEPResult(value=float(prob.value), h=tuple(h.tolist()), objective=objective, status=prob.status)
    if want_certificate:
        lam = np.zeros((dim, dim))
        for (i, j), c in zip(pairs, constraints[:-1]):
            lam[i, j] = float(c.dual_value)
        res.dual = lam
        res.tau_dual = float(init.dual_value)
        res.gram = np.asarray(G.value)
        res.fvals = np.asarray(F.value)
    return res


def prefix_worst_cases(h, objective: str = "fval", **kw) -> list[float]:
    """Peor caso de cada prefijo (h₁…hₜ), t = 1…n: lo que hace falta para el régimen *anytime*."""
    h = list(h)
    return [gd_worst_case(h[:t], objective=objective, **kw).value for t in range(1, len(h) + 1)]


# ---------------------------------------------------------------------------
# Programas de pasos de referencia
# ---------------------------------------------------------------------------
def constant_bound(n: int, h: float = 1.0) -> float:
    """Cota exacta de Drori–Teboulle para paso constante h ∈ (0, 1]: L R² / (4nh + 2)."""
    return 1.0 / (4 * n * h + 2)


def silver_schedule(k: int) -> list[float]:
    """Silver stepsize schedule de Altschuler–Parrilo de longitud n = 2ᵏ − 1:
    el paso t-ésimo es 1 + ρ^{ν(t)−1} con ν(t) la valuación 2-ádica de t y
    ρ = 1 + √2; equivalentemente h₁ = [√2] y h₂ₙ₊₁ = [hₙ, 1 + ρ^{k−2}, hₙ]
    (n = 3: [√2, 2, √2]; n = 7: [√2, 2, √2, 1 + ρ, √2, 2, √2]). El texto de
    Altschuler–Parrilo (2023) imprime «1 + √2» en el cuarto paso; es una errata:
    tanto la fórmula 2-ádica como la leyenda de su figura 1 (picos 1 + ρ^{k−1})
    dan 1 + ρ = 2 + √2 ≈ 3.414, y con ese valor se cumple la garantía del teorema."""
    if k <= 0:
        return []
    h = [float(np.sqrt(2))]
    for j in range(2, k + 1):
        h = h + [1 + SILVER_RATIO ** (j - 2)] + h
    return h


def silver_bound(k: int) -> float:
    """Garantía del teorema principal: f(xₙ) − f* ≤ L R² / (1 + √(4ρ²ᵏ − 3))."""
    return 1.0 / (1 + np.sqrt(4 * SILVER_RATIO ** (2 * k) - 3))


def _phi(x: float, y: float) -> float:
    """Paso de unión de Zhang et al. (2024): φ(x, y) = [−(x+y) + √((x+y+2)² + 4(x+1)(y+1))]/2."""
    return (-(x + y) + np.sqrt((x + y + 2) ** 2 + 4 * (x + 1) * (y + 1))) / 2


def _concat(s: list[float], r: list[float]) -> list[float]:
    """concat(s, r) = [s, φ(1ᵀs, 1ᵀr), r] (Zhang et al. 2024, ec. de concatenación)."""
    return s + [float(_phi(sum(s), sum(r)))] + r


def zhang_silver_block(i: int) -> list[float]:
    """s̄₀ = [], s̄ᵢ = concat(s̄ᵢ₋₁, s̄ᵢ₋₁): coincide con el silver de longitud 2ⁱ − 1 y suma ρⁱ − 1."""
    s: list[float] = []
    for _ in range(i):
        s = _concat(s, s)
    return s


def zhang_schedule(n: int, c: float | None = None) -> list[float]:
    """Schedule anytime de Zhang et al., «Anytime acceleration of gradient descent» (arXiv:2411.17668):
    el bloque s̄ⱼ se repite kⱼ = ⌊2·2^{c j}⌋ veces (c = log₂ρ en el teorema, es decir kⱼ = ⌊2ρʲ⌋) y los
    bloques se encadenan en orden con ŝᵢ = concat(ŝᵢ₋₁, sᵢ). Garantía demostrada: O(T^{−ϑ}) para todo T,
    ϑ = 2 log₂ρ / (1 + log₂ρ) ≈ 1.119."""
    c = float(np.log2(SILVER_RATIO)) if c is None else c
    s: list[float] = []
    j = 1
    while len(s) < n:
        block = zhang_silver_block(j)
        for _ in range(int(np.floor(2 * 2 ** (c * j)))):
            s = _concat(s, block)
            if len(s) >= n:
                break
        j += 1
    return s[:n]


ZHANG_EXPONENT = 2 * np.log2(SILVER_RATIO) / (1 + np.log2(SILVER_RATIO))


def fit_exponent(ns, values) -> float:
    """Exponente p tal que valor ≈ C·n⁻ᵖ (ajuste lineal en log-log)."""
    x = np.log(np.asarray(ns, dtype=float))
    y = np.log(np.asarray(values, dtype=float))
    A = np.vstack([x, np.ones_like(x)]).T
    slope, _ = np.linalg.lstsq(A, y, rcond=None)[0]
    return -float(slope)
