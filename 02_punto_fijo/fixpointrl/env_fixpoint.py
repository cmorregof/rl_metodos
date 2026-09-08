"""Entorno de RL para *aprender a controlar la iteración de punto fijo*.

Para resolver f(x) = 0 se usa el mapa de relajación g_α(x) = x − α·f(x): la
raíz r es punto fijo de g_α para cualquier α, pero la iteración solo converge
si g_α es contractiva cerca de r, es decir |g_α'(r)| = |1 − α·f'(r)| < 1.
El agente **no conoce f'** ni el teorema. Solo observa, tras cada iteración,
dos cosas que cualquiera puede medir:

* la razón ρₙ = |f(xₙ₊₁)| / |f(xₙ)| entre residuos consecutivos (cerca de la
  raíz, ρ ≈ |1 − α·f'(r)| = |g_α'(r)|: la constante de contracción empírica);
* si el residuo cambia de signo (la iteración oscila alrededor de la raíz) o
  no (se acerca monótonamente).

Y decide cómo ajustar α: mantenerlo, doblarlo, reducirlo a la mitad o cambiar
de signo. Recompensa por iteración: −1 + log₁₀(|f(xₙ)| / |f(xₙ₊₁)|) (shaping
basado en potencial con Φ = −log₁₀|f(x)|, que no depende de α y por tanto no
se puede "engañar" cambiando el paso), −10 si la iteración diverge, y el
episodio termina cuando |f(x)| < tol.

Lo que emerge es la ley de control que dicta el teorema de Banach: si ρ ≥ 1
la iteración no es contractiva y hay que cambiar el mapa (reducir α o
invertir su signo); si ρ < 1 pero oscila, α es demasiado grande; si ρ < 1 y es
monótona pero lenta, α es demasiado pequeño. El óptimo α ≈ 1/f'(r) (ρ → 0)
es, de hecho, el paso de Newton.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Callable

KEEP, DOUBLE, HALVE, FLIP = 0, 1, 2, 3
ACTION_NAMES = ("mantener α", "doblar α", "reducir α a la mitad", "invertir signo de α")
ACTION_SHORT = ("=", "×2", "÷2", "±")
N_ACTIONS = 4

RATIO_EDGES = (0.25, 0.5, 0.8, 1.0, 1.5)  # 6 cubos: [0,.25) [.25,.5) [.5,.8) [.8,1) [1,1.5) [1.5,∞)
RATIO_LABELS = ("<0.25", "0.25–0.5", "0.5–0.8", "0.8–1", "1–1.5", "≥1.5")

REWARD_STEP = -1.0
REWARD_DIVERGED = -10.0
ALPHA_MAX, ALPHA_MIN = 8.0, 1 / 64


def ratio_bucket(rho: float) -> int:
    for i, e in enumerate(RATIO_EDGES):
        if rho < e:
            return i
    return len(RATIO_EDGES)


def theoretical_action(bucket: int, osc: int) -> tuple[int, ...]:
    """Acciones óptimas según la teoría (ρ = |1 − α f'| para f casi lineal).

    monótona (0 < αf' < 1, ρ = 1 − αf'): doblar α da ρ' = |2ρ − 1|, mejor si ρ > 1/3.
    oscilante (αf' > 1, ρ = αf' − 1):   reducir a la mitad da ρ' = |1 − ρ|/2, mejor si ρ > 1/3.
    monótona con ρ > 1 (αf' < 0): el signo de α es incorrecto → invertir.
    oscilante con ρ > 1 (αf' > 2): α demasiado grande → reducir.
    Devuelve el conjunto de acciones aceptables (los cubos ambiguos admiten varias).
    """
    if osc == 0:  # monótona
        return {0: (KEEP,), 1: (KEEP, DOUBLE), 2: (DOUBLE,), 3: (DOUBLE,), 4: (FLIP,), 5: (FLIP, HALVE)}[bucket]
    return {0: (KEEP,), 1: (KEEP, HALVE), 2: (HALVE,), 3: (HALVE,), 4: (HALVE,), 5: (HALVE,)}[bucket]


@dataclass
class Problem:
    name: str
    f: Callable[[float], float]
    root: float
    x0: float
    dfr: float  # f'(r), solo para diagnóstico (el agente no lo ve)


def make_problem(rng: random.Random) -> Problem:
    kind = rng.choice(["lin_cubic", "exp", "sin", "poly", "cos_mix"])
    if kind == "lin_cubic":
        r, m = rng.uniform(-1, 1), rng.choice([-1, 1]) * rng.uniform(0.4, 4.0)
        f = lambda x, r=r, m=m: m * (x - r) + 0.3 * (x - r) ** 3
        name, dfr = f"{m:.2f}(x−{r:.2f}) + 0.3(x−{r:.2f})³", m
    elif kind == "exp":
        c = rng.uniform(1.5, 5.0)
        f = lambda x, c=c: math.exp(x) - c
        r, dfr = math.log(c), c
        name = f"eˣ − {c:.2f}"
    elif kind == "sin":
        k, c = rng.uniform(0.8, 2.0), rng.uniform(-0.6, 0.6)
        s = rng.choice([-1, 1])
        f = lambda x, k=k, c=c, s=s: s * (math.sin(k * x) - c)
        r = math.asin(c) / k
        dfr = s * k * math.cos(k * r)
        name = f"{'−' if s < 0 else ''}(sin({k:.2f}x) − {c:.2f})"
    elif kind == "poly":
        r = rng.uniform(-1.5, 1.5)
        p, q = rng.uniform(-2, 2), rng.uniform(0.3, 2)
        s = rng.choice([-1, 1])
        f = lambda x, r=r, p=p, q=q, s=s: s * (x - r) * ((x - p) ** 2 + q)
        dfr = s * ((r - p) ** 2 + q)
        name = f"{'−' if s < 0 else ''}(x−{r:.2f})((x−{p:.2f})²+{q:.2f})"
    else:
        f = lambda x: math.cos(x) - x
        r = 0.7390851332151607
        dfr = -math.sin(r) - 1
        name = "cos(x) − x"
    x0 = r + rng.choice([-1, 1]) * rng.uniform(0.3, 1.2)
    return Problem(name, f, r, x0, dfr)


@dataclass
class FixpointEnv:
    tol: float = 1e-6
    max_iter: int = 40
    seed: int = 0
    rng: random.Random = field(init=False)
    problem: Problem = field(init=False)
    alpha: float = 0.5
    x: float = 0.0
    x_prev: float = 0.0
    fx: float = 1.0
    fx_prev: float = 1.0
    n_iter: int = 0
    history: list[float] = field(default_factory=list)
    alpha_history: list[float] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.rng = random.Random(self.seed)
        self.reset()

    def reset(self) -> tuple:
        self.problem = make_problem(self.rng)
        self.alpha = self.rng.choice([-1.0, -0.5, -0.1, 0.1, 0.5, 1.0])
        self.x = self.problem.x0
        self.fx = self._f(self.x)
        self.fx_prev = self.fx
        self.n_iter = 0
        self.history = [self.x]
        self.alpha_history = []
        return self.state()

    def _f(self, x: float) -> float:
        try:
            v = self.problem.f(x)
        except (OverflowError, ValueError):
            return float("inf")
        return v if math.isfinite(v) else float("inf")

    def g(self, x: float) -> float:
        fx = self._f(x)
        return x - self.alpha * fx if math.isfinite(fx) else float("inf")

    def _iterate(self) -> None:
        self.x_prev, self.fx_prev = self.x, self.fx
        self.x = self.g(self.x)
        self.fx = self._f(self.x) if math.isfinite(self.x) else float("inf")
        self.history.append(self.x)
        self.alpha_history.append(self.alpha)

    @property
    def rho(self) -> float:
        """Razón de residuos consecutivos ≈ |g_α'(r)| cerca de la raíz."""
        if self.n_iter == 0 or abs(self.fx_prev) < 1e-300:
            return float("nan")
        return abs(self.fx) / abs(self.fx_prev)

    @property
    def oscillating(self) -> int:
        a, b = self.fx, self.fx_prev
        if not (math.isfinite(a) and math.isfinite(b)):
            return 0
        return 1 if a * b < 0 else 0

    def state(self) -> tuple:
        if self.n_iter == 0:
            return ("inicio",)
        rho = self.rho
        return (5 if not math.isfinite(rho) else ratio_bucket(rho), self.oscillating)

    def _apply(self, action: int) -> None:
        if action == DOUBLE:
            self.alpha = math.copysign(min(ALPHA_MAX, abs(self.alpha) * 2), self.alpha)
        elif action == HALVE:
            self.alpha = math.copysign(max(ALPHA_MIN, abs(self.alpha) / 2), self.alpha)
        elif action == FLIP:
            self.alpha = -self.alpha

    def step(self, action: int) -> tuple[tuple, float, bool, str]:
        self._apply(action)
        old = abs(self.fx)
        self._iterate()
        self.n_iter += 1
        new = abs(self.fx)
        if (
            not math.isfinite(self.x)
            or not math.isfinite(new)
            or abs(self.x - self.problem.root) > 1e3 * (1 + abs(self.problem.x0 - self.problem.root))
        ):
            return self.state(), REWARD_DIVERGED, True, "diverged"
        progress = math.log10(max(old, 1e-300) / max(new, 1e-300))
        progress = max(-3.0, min(6.0, progress))
        if new < self.tol:
            return self.state(), REWARD_STEP + progress, True, "converged"
        if self.n_iter >= self.max_iter:
            return self.state(), REWARD_STEP + progress, True, "timeout"
        return self.state(), REWARD_STEP + progress, False, "step"

    def newton_alpha(self) -> float:
        return 1.0 / self.problem.dfr
