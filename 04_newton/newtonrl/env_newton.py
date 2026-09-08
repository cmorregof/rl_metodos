"""Entorno de RL para *aprender Newton con salvaguardas*.

El paso de Newton d = −f(x)/f'(x) se conoce (es el paso de punto fijo óptimo
del proyecto 02, α = 1/f'). Lo que el agente no sabe es **cuándo es seguro
darlo entero**: lejos de la raíz Newton puede sobrepasarse, entrar en ciclos
(x³ − 2x + 2 desde 0) o divergir (arctan desde |x₀| > 1.39). El agente observa:

* ρ = |f(xₙ)| / |f(xₙ₋₁)|, la reducción del residuo en el último paso (cubos:
  < 0.05 régimen cuadrático, 0.05–0.5, 0.5–1, ≥ 1 el paso empeoró);
* si el tamaño del paso de Newton propuesto crece o decrece respecto al anterior.

Y elige el factor de amortiguación del siguiente paso: λ = 1 (Newton puro),
λ = 1/2, λ = 1/4, o RETROCEDER (volver al punto anterior y dar la mitad del
último paso: búsqueda lineal). Recompensa: −1 por evaluación + log₁₀ de la
reducción del **error real** |x − r| (shaping basado en potencial; el entorno
conoce la raíz, el agente no la ve: así no se premia "converger" a una falsa
raíz en el infinito donde f → 0), −10 si diverge o f' se anula, fin cuando
|x − r| < tol.

Lo que emerge es Newton amortiguado con backtracking: paso completo cuando el
residuo cae rápido (cuenca de convergencia cuadrática), amortiguar cuando
apenas mejora, retroceder cuando empeora.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Callable

FULL, HALF, QUARTER, BACK = 0, 1, 2, 3
ACTION_NAMES = ("paso completo λ=1", "paso λ=1/2", "paso λ=1/4", "retroceder y dar medio paso")
ACTION_SHORT = ("λ1", "λ½", "λ¼", "↩")
N_ACTIONS = 4

RATIO_EDGES = (0.05, 0.5, 1.0)
RATIO_LABELS = ("ρ<0.05", "0.05–0.5", "0.5–1", "ρ≥1")
ACTION_COST_NOTE = "cada evaluación de f cuesta 1; la búsqueda lineal puede costar varias"
REWARD_STEP = -1.0
REWARD_FAIL = -10.0


def ratio_bucket(rho: float) -> int:
    for i, e in enumerate(RATIO_EDGES):
        if rho < e:
            return i
    return len(RATIO_EDGES)


def theoretical_action(rb: int, growing: int) -> tuple[int, ...]:
    """Regla de referencia: Newton amortiguado con condición de Armijo (c = ½).

    * ρ < 0.5: el último paso redujo |f| al menos a la mitad → se acepta y se da
      el siguiente paso completo (cuenca de convergencia cuadrática si ρ ≪ 1).
    * ρ ≥ 0.5: descenso insuficiente o empeoramiento → Armijo rechaza el paso:
      retroceder con búsqueda lineal o, al menos, amortiguar el siguiente.
    El agente puede descubrir matices que esta regla no contempla (p. ej. que si
    el paso de Newton propuesto ya decrece, el paso completo es más barato).
    """
    if rb <= 1:
        return (FULL,)
    return (BACK, HALF, QUARTER)


@dataclass
class Problem:
    name: str
    f: Callable[[float], float]
    df: Callable[[float], float]
    root: float
    x0: float


def _cbrt(x: float) -> float:
    return math.copysign(abs(x) ** (1 / 3), x)


def make_problem(rng: random.Random) -> Problem:
    kind = rng.choice(["atan", "cbrt", "exp", "cubic", "gauss", "log", "poly3", "cycle"])
    sgn = rng.choice([-1, 1])
    if kind == "atan":  # Newton puro diverge si |x₀| > 1.39
        return Problem("arctan x", math.atan, lambda x: 1 / (1 + x * x), 0.0, sgn * rng.uniform(0.5, 3.0))
    if kind == "cbrt":  # Newton puro diverge desde CUALQUIER x₀ (xₙ₊₁ = −2xₙ)
        return Problem("∛x", _cbrt, lambda x: (1 / 3) * abs(x) ** (-2 / 3) if x != 0 else float("inf"), 0.0, sgn * rng.uniform(0.5, 3.0))
    if kind == "exp":  # desde la izquierda el primer paso se pasa de largo enormemente
        c = rng.uniform(1.5, 6.0)
        r = math.log(c)
        return Problem(f"eˣ − {c:.2f}", lambda x, c=c: math.exp(x) - c, math.exp, r, r + sgn * rng.uniform(0.5, 3.0))
    if kind == "cubic":  # convexa y creciente a la derecha: convergencia monótona y cuadrática
        return Problem("x³ − x − 1", lambda x: x**3 - x - 1, lambda x: 3 * x * x - 1, 1.3247179572447460, rng.uniform(1.6, 3.5))
    if kind == "gauss":  # con |x₀| ≈ 0.5 Newton puro cicla o se pasa al otro lado
        return Problem("x·e^{−x²}", lambda x: x * math.exp(-x * x), lambda x: (1 - 2 * x * x) * math.exp(-x * x), 0.0, sgn * rng.uniform(0.3, 0.6))
    if kind == "log":  # desde x₀ > e² el paso cae en x ≤ 0, fuera del dominio
        return Problem("ln x − 1", lambda x: math.log(x) - 1 if x > 0 else float("nan"), lambda x: 1 / x if x > 0 else float("nan"), math.e, rng.uniform(0.3, 8.0))
    if kind == "cycle":  # x³ − 2x + 2: desde la izquierda de la raíz, convergencia limpia
        return Problem("x³ − 2x + 2", lambda x: x**3 - 2 * x + 2, lambda x: 3 * x * x - 2, -1.7692923542386314, rng.uniform(-3.5, -2.2))
    r = rng.uniform(-1, 1)
    p, q = rng.uniform(-2, 2), rng.uniform(0.2, 1.5)
    return Problem(
        f"(x−{r:.2f})((x−{p:.2f})²+{q:.2f})",
        lambda x, r=r, p=p, q=q: (x - r) * ((x - p) ** 2 + q),
        lambda x, r=r, p=p, q=q: ((x - p) ** 2 + q) + 2 * (x - r) * (x - p),
        r,
        r + sgn * rng.uniform(0.5, 3.0),
    )


@dataclass
class NewtonEnv:
    tol: float = 1e-9
    max_iter: int = 50
    seed: int = 0
    rng: random.Random = field(init=False)
    problem: Problem = field(init=False)
    x: float = 0.0
    fx: float = 1.0
    dfx: float = 1.0
    x_prev: float = 0.0
    fx_prev: float = 1.0
    last_step: float = 0.0  # último desplazamiento efectivo
    last_dir: float = float("inf")  # última dirección de Newton (desde x_prev)
    last_lam: float = 1.0
    n_iter: int = 0
    n_evals: int = 0
    history: list[float] = field(default_factory=list)
    lam_history: list[float] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.rng = random.Random(self.seed)
        self.reset()

    def _eval(self, x: float) -> tuple[float, float]:
        try:
            v, d = self.problem.f(x), self.problem.df(x)
        except (OverflowError, ValueError, ZeroDivisionError):
            return float("inf"), float("inf")
        return (v if math.isfinite(v) else float("inf")), (d if math.isfinite(d) else float("inf"))

    def reset(self) -> tuple:
        self.problem = make_problem(self.rng)
        self.x = self.problem.x0
        self.fx, self.dfx = self._eval(self.x)
        self.x_prev, self.fx_prev = self.x, self.fx
        self.last_step = 0.0
        self.last_dir = float("inf")
        self.last_lam = 1.0
        self.n_iter = 0
        self.n_evals = 1
        self.history = [self.x]
        self.lam_history = []
        return self.state()

    @property
    def newton_step(self) -> float:
        if not math.isfinite(self.dfx) or abs(self.dfx) < 1e-300 or not math.isfinite(self.fx):
            return float("inf")
        return -self.fx / self.dfx

    @property
    def rho(self) -> float:
        if self.n_iter == 0 or not math.isfinite(self.fx_prev) or abs(self.fx_prev) < 1e-300:
            return float("nan")
        return abs(self.fx) / abs(self.fx_prev) if math.isfinite(self.fx) else float("inf")

    @property
    def step_growing(self) -> int:
        d = abs(self.newton_step)
        if self.n_iter == 0 or not math.isfinite(d):
            return 1 if not math.isfinite(d) else 0
        return 1 if d > abs(self.last_step) else 0

    def state(self) -> tuple:
        if self.n_iter == 0:
            return ("inicio",)
        rho = self.rho
        return (3 if not math.isfinite(rho) else ratio_bucket(rho), self.step_growing)  # f no finita ⇒ "empeoró"

    @property
    def err(self) -> float:
        return abs(self.x - self.problem.root)

    def step(self, action: int) -> tuple[tuple, float, bool, str]:
        old = self.err
        n_evals = 1
        if action == BACK and self.n_iter > 0 and math.isfinite(self.last_dir):
            # búsqueda lineal (Armijo simplificado): desde el punto anterior, con la
            # dirección de Newton anterior, reducir λ hasta que |f| disminuya.
            base_x, base_f = self.x_prev, self.fx_prev
            lam = self.last_lam / 2
            n_evals = 0
            while True:
                new_x = base_x + lam * self.last_dir
                fx, dfx = self._eval(new_x)
                n_evals += 1
                if (math.isfinite(fx) and abs(fx) < abs(base_f)) or lam <= 1 / 64:
                    break
                lam /= 2
            self.last_lam = lam
            self.lam_history.append(-lam)
        else:
            lam = {FULL: 1.0, HALF: 0.5, QUARTER: 0.25, BACK: 1.0}[action]
            d = self.newton_step
            if not math.isfinite(d):
                self.n_iter += 1
                return self.state(), REWARD_FAIL, True, "derivative_zero"
            base_x, base_f = self.x, self.fx
            new_x = self.x + lam * d
            fx, dfx = self._eval(new_x)
            self.last_dir, self.last_lam = d, lam
            self.lam_history.append(lam)
        self.x_prev, self.fx_prev = base_x, base_f
        self.x, self.fx, self.dfx = new_x, fx, dfx
        self.last_step = self.x - self.x_prev
        self.n_iter += 1
        self.n_evals += n_evals
        self.history.append(self.x)
        new = self.err
        if not math.isfinite(self.x) or new > 1e3 * (1 + abs(self.problem.x0 - self.problem.root)):
            return self.state(), REWARD_FAIL, True, "diverged"
        progress = max(-3.0, min(6.0, math.log10(max(old, 1e-300) / max(new, 1e-300))))
        r = REWARD_STEP * n_evals + progress
        if new < self.tol:
            return self.state(), r, True, "converged"
        if self.n_iter >= self.max_iter:
            return self.state(), r, True, "timeout"
        return self.state(), r, False, "step"

    def observed_order(self) -> float:
        """Estimación del orden de convergencia con los tres últimos errores."""
        es = [abs(x - self.problem.root) for x in self.history[-4:] if math.isfinite(x)]
        es = [e for e in es if e > 1e-15]
        if len(es) < 3:
            return float("nan")
        try:
            return math.log(es[-1] / es[-2]) / math.log(es[-2] / es[-3])
        except (ValueError, ZeroDivisionError):
            return float("nan")
