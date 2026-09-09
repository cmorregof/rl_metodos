"""Entorno de RL para *aprender la secante con salvaguardas*.

El paso de la secante se conoce: la recta por los dos últimos puntos
(xₙ₋₁, f(xₙ₋₁)) y (xₙ, f(xₙ)) corta el eje en

    xₙ₊₁ = xₙ − f(xₙ) · (xₙ − xₙ₋₁) / (f(xₙ) − f(xₙ₋₁)).

Es Newton con la derivada sustituida por un cociente incremental (proyecto 04)
y, como Newton, lejos de la raíz puede salirse del corchete, oscilar (∛x) o
caer fuera del dominio (ln x). Lo que el agente no sabe es **cuándo fiarse de
ese paso y cuándo refugiarse en el corchete** [lo, hi] con cambio de signo que
siempre mantiene (proyecto 01). Observa:

* ρ = |f(xₙ)| / |f(xₙ₋₁)|, la reducción del residuo en el último paso (cubos:
  < 0.05 régimen superlineal, 0.05–0.5, 0.5–1, ≥ 1 el paso empeoró);
* si el punto que propone la secante cae DENTRO del corchete o fuera de él.

Y elige cómo generar el siguiente punto: SECANTE (los dos últimos puntos),
REGULA FALSI (la secante por los dos extremos del corchete: siempre cae
dentro, pero un extremo se estanca y el orden es 1) o BISECCIÓN (el punto
medio). Recompensa: −1 por evaluación + log₁₀ de la reducción del **error
real** |x − r| (shaping basado en potencial; el entorno conoce la raíz, el
agente no), −10 si diverge o la secante es indefinida, fin cuando |x − r| < tol.

Lo que emerge es el método de Dekker/Brent: secante mientras el punto cae en
el corchete y el residuo baja (y ahí el orden observado es φ ≈ 1.618), corchete
cuando la secante se sale o deja de mejorar.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Callable

SEC, FALSI, BISECT = 0, 1, 2
ACTION_NAMES = ("secante por los dos últimos puntos", "regula falsi por los extremos del corchete", "bisección del corchete")
ACTION_SHORT = ("sec", "rf", "bis")
N_ACTIONS = 3

RATIO_EDGES = (0.05, 0.5, 1.0)
RATIO_LABELS = ("ρ<0.05", "0.05–0.5", "0.5–1", "ρ≥1")
REWARD_STEP = -1.0
REWARD_FAIL = -10.0
PHI = (1 + math.sqrt(5)) / 2


def ratio_bucket(rho: float) -> int:
    for i, e in enumerate(RATIO_EDGES):
        if rho < e:
            return i
    return len(RATIO_EDGES)


def theoretical_action(rb: int, inside: int) -> tuple[int, ...]:
    """Regla de referencia: método de Dekker (1969), la base de Brent.

    * el punto de la secante cae dentro del corchete y el último paso redujo
      |f| al menos a la mitad → se acepta la secante (régimen superlineal);
    * en otro caso (se sale del corchete, o apenas mejora, o empeoró) → paso
      con el corchete: bisección o regula falsi.
    El agente puede descubrir matices (p. ej. que la regula falsi es un
    refugio peor que la bisección porque un extremo se estanca).
    """
    if inside and rb <= 1:
        return (SEC,)
    return (BISECT, FALSI)


@dataclass
class Problem:
    name: str
    f: Callable[[float], float]
    root: float
    lo: float
    hi: float
    family: str = ""

    def __post_init__(self) -> None:
        if not self.family:
            self.family = self.name


def _cbrt(x: float) -> float:
    return math.copysign(abs(x) ** (1 / 3), x)


def make_problem(rng: random.Random) -> Problem:
    kind = rng.choice(["atan", "cbrt", "exp", "cubic", "gauss", "log", "poly3", "cycle", "flat"])
    if kind == "atan":  # corchete muy asimétrico: la secante por dos puntos lejanos se sale del corchete
        return Problem("arctan x", math.atan, 0.0, -rng.uniform(0.2, 1.0), rng.uniform(3.0, 8.0))
    if kind == "cbrt":  # la secante pura oscila sin converger; la regula falsi se estanca
        return Problem("∛x", _cbrt, 0.0, -rng.uniform(0.3, 1.5), rng.uniform(1.5, 4.0))
    if kind == "exp":  # convexa: la secante por la derecha se pasa de largo por la izquierda
        c = rng.uniform(1.5, 6.0)
        r = math.log(c)
        return Problem(f"eˣ − {c:.2f}", lambda x, c=c: math.exp(x) - c, r, r - rng.uniform(1.0, 4.0), r + rng.uniform(1.5, 4.0), "eˣ − c")
    if kind == "cubic":  # convergencia limpia: para ver el orden φ
        return Problem("x³ − x − 1", lambda x: x**3 - x - 1, 1.3247179572447460, rng.uniform(0.9, 1.25), rng.uniform(1.6, 3.0))
    if kind == "gauss":  # casi plana lejos de la raíz: la secante dispara puntos muy lejos
        return Problem("x·e^{−x²}", lambda x: x * math.exp(-x * x), 0.0, -rng.uniform(0.3, 2.5), rng.uniform(0.3, 2.5))
    if kind == "log":  # la secante desde la derecha cae en x ≤ 0, fuera del dominio
        return Problem("ln x − 1", lambda x: math.log(x) - 1 if x > 0 else float("nan"), math.e, rng.uniform(0.2, 1.0), rng.uniform(6.0, 20.0))
    if kind == "cycle":
        return Problem("x³ − 2x + 2", lambda x: x**3 - 2 * x + 2, -1.7692923542386314, rng.uniform(-3.5, -2.2), rng.uniform(-1.5, 0.5))
    if kind == "flat":  # raíz con f' pequeña a un lado: la regula falsi se estanca durante muchos pasos
        k = rng.choice([5, 7])
        return Problem(f"x^{k} + x/8", lambda x, k=k: x**k + x / 8, 0.0, -rng.uniform(0.6, 1.2), rng.uniform(0.6, 1.2), "xᵏ + x/8")
    r = rng.uniform(-1, 1)
    p, q = rng.uniform(-2, 2), rng.uniform(0.2, 1.5)
    return Problem(
        f"(x−{r:.2f})((x−{p:.2f})²+{q:.2f})",
        lambda x, r=r, p=p, q=q: (x - r) * ((x - p) ** 2 + q),
        r,
        r - rng.uniform(0.3, 3.0),
        r + rng.uniform(0.3, 3.0),
        "cúbica con un par complejo",
    )


@dataclass
class SecantEnv:
    tol: float = 1e-9
    max_iter: int = 40
    seed: int = 0
    rng: random.Random = field(init=False)
    problem: Problem = field(init=False)
    x: float = 0.0
    fx: float = 1.0
    x_prev: float = 0.0
    fx_prev: float = 1.0
    lo: float = 0.0
    hi: float = 1.0
    flo: float = -1.0
    fhi: float = 1.0
    n_iter: int = 0
    n_evals: int = 0
    history: list[float] = field(default_factory=list)
    action_history: list[int] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.rng = random.Random(self.seed)
        self.reset()

    def _eval(self, x: float) -> float:
        try:
            v = self.problem.f(x)
        except (OverflowError, ValueError, ZeroDivisionError):
            return float("inf")
        return v if math.isfinite(v) else float("inf")

    def reset(self) -> tuple:
        self.problem = make_problem(self.rng)
        p = self.problem
        self.lo, self.hi = p.lo, p.hi
        self.flo, self.fhi = self._eval(self.lo), self._eval(self.hi)
        # los dos primeros iterados son los extremos del corchete; el más reciente, el de menor |f|
        if abs(self.flo) <= abs(self.fhi):
            self.x_prev, self.fx_prev, self.x, self.fx = self.hi, self.fhi, self.lo, self.flo
        else:
            self.x_prev, self.fx_prev, self.x, self.fx = self.lo, self.flo, self.hi, self.fhi
        self.n_iter = 0
        self.n_evals = 2
        self.history = [self.x_prev, self.x]
        self.action_history = []
        return self.state()

    # ---- observables -------------------------------------------------------
    @property
    def secant_point(self) -> float:
        if not (math.isfinite(self.fx) and math.isfinite(self.fx_prev)):
            return float("nan")
        den = self.fx - self.fx_prev
        if abs(den) < 1e-300 or self.x == self.x_prev:
            return float("nan")
        return self.x - self.fx * (self.x - self.x_prev) / den

    @property
    def falsi_point(self) -> float:
        return self.hi - self.fhi * (self.hi - self.lo) / (self.fhi - self.flo)

    @property
    def rho(self) -> float:
        if self.n_iter == 0 or not math.isfinite(self.fx_prev) or abs(self.fx_prev) < 1e-300:
            return float("nan")
        return abs(self.fx) / abs(self.fx_prev) if math.isfinite(self.fx) else float("inf")

    @property
    def secant_inside(self) -> int:
        s = self.secant_point
        return 1 if math.isfinite(s) and self.lo <= s <= self.hi else 0

    def state(self) -> tuple:
        if self.n_iter == 0:
            return ("inicio",)
        rho = self.rho
        return (3 if not math.isfinite(rho) else ratio_bucket(rho), self.secant_inside)

    @property
    def err(self) -> float:
        return abs(self.x - self.problem.root) if math.isfinite(self.x) else float("inf")

    @property
    def width(self) -> float:
        return self.hi - self.lo

    # ---- dinámica ----------------------------------------------------------
    def _update_bracket(self, x: float, fx: float) -> None:
        if not (self.lo < x < self.hi) or not math.isfinite(fx):
            return
        if fx == 0.0:
            self.lo = self.hi = x
            self.flo = self.fhi = 0.0
        elif (fx < 0) == (self.flo < 0):
            self.lo, self.flo = x, fx
        else:
            self.hi, self.fhi = x, fx

    def step(self, action: int) -> tuple[tuple, float, bool, str]:
        old = self.err
        if action == SEC:
            new_x = self.secant_point
            if not math.isfinite(new_x):
                self.n_iter += 1
                self.action_history.append(action)
                return self.state(), REWARD_FAIL, True, "undefined"
        elif action == FALSI:
            new_x = self.falsi_point
        else:
            new_x = 0.5 * (self.lo + self.hi)
        fx = self._eval(new_x)
        self.x_prev, self.fx_prev = self.x, self.fx
        self.x, self.fx = new_x, fx
        self._update_bracket(new_x, fx)
        self.n_iter += 1
        self.n_evals += 1
        self.history.append(self.x)
        self.action_history.append(action)
        new = self.err
        if not math.isfinite(self.x) or not math.isfinite(fx) or new > 1e3 * (1 + abs(self.problem.hi - self.problem.lo)):
            return self.state(), REWARD_FAIL, True, "diverged"
        progress = max(-3.0, min(6.0, math.log10(max(old, 1e-300) / max(new, 1e-300))))
        r = REWARD_STEP + progress
        if new < self.tol:
            return self.state(), r, True, "converged"
        if self.n_iter >= self.max_iter:
            return self.state(), r, True, "timeout"
        return self.state(), r, False, "step"

    def observed_order(self) -> float:
        """Orden de convergencia estimado con los tres últimos pasos, solo si fueron de secante."""
        if len(self.action_history) < 3 or any(a != SEC for a in self.action_history[-3:]):
            return float("nan")
        es = [abs(x - self.problem.root) for x in self.history[-4:] if math.isfinite(x)]
        es = [e for e in es if e > 1e-15]
        if len(es) < 3:
            return float("nan")
        try:
            return math.log(es[-1] / es[-2]) / math.log(es[-2] / es[-3])
        except (ValueError, ZeroDivisionError):
            return float("nan")
