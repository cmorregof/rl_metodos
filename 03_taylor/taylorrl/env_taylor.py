"""Entorno de RL para *aprender a usar la serie de Taylor* como método numérico.

Objetivo: calcular f(x*) con la serie de Taylor centrada en 0 usando el menor
número de términos, con tolerancia `tol`. El agente **no conoce f ni su radio
de convergencia**. Tras cada término solo observa:

* el tamaño del último término añadido respecto a la tolerancia,
  log₁₀(|tₙ| / tol), en cubos;
* el cociente entre los dos últimos términos no nulos, |tₙ / tₙ₋₁|, en cubos
  (esto es lo que mira el criterio del cociente de d'Alembert);
* si lleva pocos (n ≤ 4) o muchos (n > 4) términos.

Y decide una de tres acciones:

* SUMAR el siguiente término (coste −1 + log₁₀(error anterior / error nuevo):
  shaping basado en potencial con Φ = −log₁₀ del error real, que el entorno
  conoce pero el agente no ve);
* PARAR y entregar la suma parcial: +10 si |f(x*) − Sₙ| < tol, −10 si no;
* ABANDONAR declarando que la serie no converge en x*: +10 si en efecto
  |x*| supera el radio de convergencia, −10 si no.

Lo que emerge es el criterio del cociente (si los términos crecen de forma
sostenida la serie no converge) y el criterio del último término (parar cuando
el último término es mucho menor que la tolerancia y el cociente es pequeño).
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Callable

ADD, STOP, ABORT = 0, 1, 2
ACTION_NAMES = ("sumar término", "parar y entregar", "abandonar: no converge")
ACTION_SHORT = ("+", "■", "✗")
N_ACTIONS = 3

TERM_EDGES = (-2.0, 0.0, 2.0)  # log10(|t|/tol): <−2, −2..0, 0..2, ≥2
TERM_LABELS = ("|t|<tol/100", "tol/100–tol", "tol–100tol", "|t|≥100tol")
RATIO_EDGES = (0.5, 1.0)  # |tₙ/tₙ₋₁|: <0.5, 0.5–1, ≥1
RATIO_LABELS = ("q<0.5", "0.5≤q<1", "q≥1")
N_LATE = 4

REWARD_TERM = -1.0
REWARD_RIGHT = 10.0
REWARD_WRONG = -10.0
REWARD_TIMEOUT = -5.0


def term_bucket(t: float, tol: float) -> int:
    v = math.log10(max(t, 1e-300) / tol)
    for i, e in enumerate(TERM_EDGES):
        if v < e:
            return i
    return len(TERM_EDGES)


def ratio_bucket(q: float) -> int:
    for i, e in enumerate(RATIO_EDGES):
        if q < e:
            return i
    return len(RATIO_EDGES)


def theoretical_action(tb: int, rb: int, late: int) -> tuple[int, ...]:
    """Acciones aceptables según la teoría.

    * cociente ≥ 1 sostenido (late): la serie no converge → abandonar; al principio
      los términos pueden crecer aunque converja (eˣ con x = 2.5) → seguir sumando.
    * último término < tol/100 con cociente < 1: la cola ≈ |t|·q/(1−q) < tol → parar.
    * tol/100 ≤ |t| < tol: si q < 0.5 la cola es < |t| < tol → parar; si 0.5 ≤ q < 1 es ambiguo.
    * |t| ≥ tol: seguir sumando.
    """
    if rb == 2:
        return (ABORT,) if late else (ADD,)
    if tb == 0:
        return (STOP,)
    if tb == 1:
        return (STOP,) if rb == 0 else (STOP, ADD)
    return (ADD,)


@dataclass
class Series:
    name: str
    f: Callable[[float], float]
    coef: Callable[[int], float]  # coeficiente c_k de la serie en 0
    radius: float
    domain: tuple[float, float]  # dominio de f para elegir x*


def _binom_half(k: int) -> float:
    c = 1.0
    for j in range(k):
        c *= (0.5 - j) / (j + 1)
    return c


SERIES: list[Series] = [
    Series("eˣ", math.exp, lambda k: 1.0 / math.factorial(k), math.inf, (-2.5, 2.5)),
    Series("sin x", math.sin, lambda k: 0.0 if k % 2 == 0 else (-1) ** (k // 2) / math.factorial(k), math.inf, (-2.5, 2.5)),
    Series("cos x", math.cos, lambda k: 0.0 if k % 2 else (-1) ** (k // 2) / math.factorial(k), math.inf, (-2.5, 2.5)),
    Series("ln(1+x)", lambda x: math.log1p(x), lambda k: 0.0 if k == 0 else (-1) ** (k + 1) / k, 1.0, (-0.6, 2.5)),
    Series("1/(1−x)", lambda x: 1 / (1 - x), lambda k: 1.0, 1.0, (-2.5, 0.6)),
    Series("arctan x", math.atan, lambda k: 0.0 if k % 2 == 0 else (-1) ** (k // 2) / k, 1.0, (-2.5, 2.5)),
    Series("√(1+x)", lambda x: math.sqrt(1 + x), _binom_half, 1.0, (-0.6, 2.5)),
    Series("1/(1+x²)", lambda x: 1 / (1 + x * x), lambda k: 0.0 if k % 2 else (-1) ** (k // 2), 1.0, (-2.5, 2.5)),
]


@dataclass
class Problem:
    series: Series
    x: float

    @property
    def diverges(self) -> bool:
        return abs(self.x) > self.series.radius

    @property
    def name(self) -> str:
        return f"{self.series.name} en x* = {self.x:.2f}"


def make_problem(rng: random.Random) -> Problem:
    s = rng.choice(SERIES)
    lo, hi = s.domain
    while True:
        x = rng.uniform(lo, hi)
        if abs(x) < 0.15:
            continue
        # evitar la zona ambigua alrededor del radio (convergencia lentísima)
        if s.radius < math.inf and 0.6 < abs(x) < 1.3:
            continue
        return Problem(s, x)


@dataclass
class TaylorEnv:
    tol: float = 1e-6
    max_terms: int = 60
    seed: int = 0
    rng: random.Random = field(init=False)
    problem: Problem = field(init=False)
    n: int = 0  # grado alcanzado (último índice sumado)
    partial: float = 0.0
    terms: list[float] = field(default_factory=list)  # términos no nulos añadidos
    partials: list[float] = field(default_factory=list)
    last_ratio: float = float("nan")

    def __post_init__(self) -> None:
        self.rng = random.Random(self.seed)
        self.reset()

    def reset(self) -> tuple:
        self.problem = make_problem(self.rng)
        self.n = -1
        self.partial = 0.0
        self.terms = []
        self.partials = []
        self.last_ratio = float("nan")
        self._add_term()  # el término de orden 0 siempre se suma (sin decisión)
        return self.state()

    @property
    def true_value(self) -> float:
        return self.problem.series.f(self.problem.x)

    @property
    def error(self) -> float:
        return abs(self.true_value - self.partial)

    def _add_term(self) -> None:
        """Suma el siguiente término no nulo (los nulos no cuestan)."""
        while True:
            self.n += 1
            c = self.problem.series.coef(self.n)
            if c != 0.0 or self.n > self.max_terms + 5:
                break
        t = c * self.problem.x ** self.n
        if self.terms and abs(self.terms[-1]) > 0:
            self.last_ratio = abs(t) / abs(self.terms[-1])
        self.terms.append(t)
        self.partial += t
        self.partials.append(self.partial)

    @property
    def n_terms(self) -> int:
        return len(self.terms)

    def state(self) -> tuple:
        tb = term_bucket(abs(self.terms[-1]), self.tol)
        rb = 1 if math.isnan(self.last_ratio) else ratio_bucket(self.last_ratio)
        late = 1 if self.n_terms > N_LATE else 0
        return (tb, rb, late)

    def step(self, action: int) -> tuple[tuple, float, bool, str]:
        if action == STOP:
            ok = self.error < self.tol
            return self.state(), (REWARD_RIGHT if ok else REWARD_WRONG), True, ("stop_ok" if ok else "stop_bad")
        if action == ABORT:
            ok = self.problem.diverges
            return self.state(), (REWARD_RIGHT if ok else REWARD_WRONG), True, ("abort_ok" if ok else "abort_bad")
        old = self.error
        self._add_term()
        new = self.error
        progress = math.log10(max(old, 1e-300) / max(new, 1e-300))
        progress = max(-3.0, min(3.0, progress))
        if not math.isfinite(self.partial) or abs(self.partial) > 1e12:
            return self.state(), REWARD_TERM + progress + REWARD_TIMEOUT, True, "overflow"
        if self.n_terms >= self.max_terms:
            return self.state(), REWARD_TERM + progress + REWARD_TIMEOUT, True, "timeout"
        return self.state(), REWARD_TERM + progress, False, "add"

    def optimal_terms(self) -> int | None:
        """Mínimo número de términos no nulos con error < tol (None si diverge)."""
        if self.problem.diverges:
            return None
        s, n, k = 0.0, 0, 0
        for k in range(self.max_terms * 3):
            c = self.problem.series.coef(k)
            if c == 0.0:
                continue
            s += c * self.problem.x ** k
            n += 1
            if abs(self.true_value - s) < self.tol:
                return n
        return None
