"""Entorno de RL para *aprender el método de bisección*.

El agente no conoce el algoritmo. En cada iteración decide dos cosas:

1. **Dónde cortar** el intervalo [a, b]: elige una fracción λ ∈ Λ y se evalúa
   f en x = a + λ (b − a).
2. **Qué subintervalo conservar**: [a, x] o [x, b].

Recompensas:
* −1 por iteración (cada evaluación de f cuesta);
* + log₂(longitud anterior / longitud nueva): *shaping basado en potencial*
  (Ng, Harada y Russell, 1999) con Φ = −log₂(longitud); no altera la política
  óptima, solo hace visible el progreso paso a paso;
* −10 y fin del episodio si el subintervalo conservado **pierde el cambio de
  signo** (viola la hipótesis de Bolzano, "perdió la raíz");
* el episodio termina (sin bonificación extra) cuando la longitud baja de la
  tolerancia: la recompensa acumulada es log₂(w₀/wₙ) − n, máxima cuando se
  llega con el menor número n de evaluaciones.

Lo que el agente termina aprendiendo es exactamente el método de bisección:
λ = 1/2 (el corte minimax) y conservar siempre el subintervalo donde
f cambia de signo (el invariante f(aₙ)·f(bₙ) ≤ 0).
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Callable

LAMBDAS: tuple[float, ...] = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
KEEP_LEFT, KEEP_RIGHT = 0, 1

REWARD_STEP = -1.0
REWARD_LOST_ROOT = -10.0
REWARD_CONVERGED = 0.0


@dataclass
class Problem:
    name: str
    f: Callable[[float], float]
    a: float
    b: float
    root: float


def _bisect_reference(f: Callable[[float], float], a: float, b: float, tol: float = 1e-12) -> float:
    fa = f(a)
    for _ in range(200):
        m = 0.5 * (a + b)
        fm = f(m)
        if fa * fm <= 0:
            b = m
        else:
            a, fa = m, fm
        if b - a < tol:
            break
    return 0.5 * (a + b)


def make_problem(rng: random.Random) -> Problem:
    """Genera una función continua con cambio de signo en [a, b]."""
    kind = rng.choice(["poly3", "sin", "exp", "cos_mix", "cubic_root"])
    if kind == "poly3":
        r = rng.uniform(-1.5, 1.5)
        p, q = rng.uniform(-2, 2), rng.uniform(0.2, 2)
        f = lambda x, r=r, p=p, q=q: (x - r) * ((x - p) ** 2 + q)
        a, b = r - rng.uniform(0.5, 2.5), r + rng.uniform(0.5, 2.5)
        name = f"(x−{r:.2f})·((x−{p:.2f})²+{q:.2f})"
    elif kind == "sin":
        k = rng.uniform(0.7, 1.6)
        c = rng.uniform(-0.8, 0.8)
        f = lambda x, k=k, c=c: math.sin(k * x) - c
        root = math.asin(c) / k
        a, b = root - rng.uniform(0.3, 1.0), root + rng.uniform(0.3, 1.0)
        name = f"sin({k:.2f}x) − {c:.2f}"
    elif kind == "exp":
        c = rng.uniform(1.5, 6.0)
        f = lambda x, c=c: math.exp(x) - c
        root = math.log(c)
        a, b = root - rng.uniform(0.5, 2.0), root + rng.uniform(0.5, 2.0)
        name = f"eˣ − {c:.2f}"
    elif kind == "cos_mix":
        f = lambda x: math.cos(x) - x
        a, b = rng.uniform(-1.0, 0.3), rng.uniform(1.2, 2.5)
        name = "cos(x) − x"
    else:
        r = rng.uniform(-1, 1)
        f = lambda x, r=r: (x - r) ** 3 + 0.3 * (x - r)
        a, b = r - rng.uniform(0.4, 2.0), r + rng.uniform(0.4, 2.0)
        name = f"(x−{r:.2f})³ + 0.3(x−{r:.2f})"
    if f(a) * f(b) >= 0:  # por seguridad numérica
        return make_problem(rng)
    root = _bisect_reference(f, a, b)
    return Problem(name, f, a, b, root)


@dataclass
class BisectionEnv:
    tol: float = 1e-3
    max_iter: int = 60
    seed: int = 0
    rng: random.Random = field(init=False)
    problem: Problem = field(init=False)
    a: float = field(init=False, default=0.0)
    b: float = field(init=False, default=1.0)
    fa: float = field(init=False, default=0.0)
    fb: float = field(init=False, default=0.0)
    x: float = field(init=False, default=0.5)
    fx: float = field(init=False, default=0.0)
    lam_index: int = field(init=False, default=4)
    n_iter: int = field(init=False, default=0)
    history: list[tuple[float, float]] = field(init=False, default_factory=list)

    def __post_init__(self) -> None:
        self.rng = random.Random(self.seed)
        self.reset()

    # ---- ciclo -----------------------------------------------------------
    def reset(self) -> tuple:
        self.problem = make_problem(self.rng)
        self.a, self.b = self.problem.a, self.problem.b
        self.fa, self.fb = self.problem.f(self.a), self.problem.f(self.b)
        self.n_iter = 0
        self.history = [(self.a, self.b)]
        return self.state_split()

    @property
    def width0(self) -> float:
        return self.problem.b - self.problem.a

    def width_bucket(self) -> int:
        """Cuántas 'mitades' faltan aproximadamente para la tolerancia."""
        ratio = (self.b - self.a) / self.tol
        return min(20, max(0, int(math.log2(ratio)))) if ratio > 1 else 0

    def state_split(self) -> tuple:
        # Un único estado de corte: la fracción óptima no depende de la anchura,
        # y así todas las muestras refuerzan la misma fila de la tabla Q.
        return ("split",)

    def state_keep(self) -> tuple:
        # Solo los signos: son lo único que decide qué lado conserva la raíz.
        sa = 1 if self.fa > 0 else -1
        sx = 1 if self.fx > 0 else (-1 if self.fx < 0 else 0)
        sb = 1 if self.fb > 0 else -1
        return ("keep", sa, sx, sb)

    def step_split(self, lam_index: int) -> tuple:
        self.lam_index = lam_index
        lam = LAMBDAS[lam_index]
        self.x = self.a + lam * (self.b - self.a)
        self.fx = self.problem.f(self.x)
        return self.state_keep()

    def step_keep(self, keep: int) -> tuple[tuple, float, bool, str]:
        self.n_iter += 1
        old_width = self.b - self.a
        if keep == KEEP_LEFT:
            new_a, new_fa, new_b, new_fb = self.a, self.fa, self.x, self.fx
        else:
            new_a, new_fa, new_b, new_fb = self.x, self.fx, self.b, self.fb
        self.a, self.fa, self.b, self.fb = new_a, new_fa, new_b, new_fb
        self.history.append((self.a, self.b))
        if self.fa * self.fb > 0:
            return self.state_split(), REWARD_LOST_ROOT, True, "lost"
        progress = math.log2(old_width / (self.b - self.a))
        if self.b - self.a < self.tol:
            return self.state_split(), REWARD_STEP + progress + REWARD_CONVERGED, True, "converged"
        if self.n_iter >= self.max_iter:
            return self.state_split(), REWARD_STEP + progress, True, "timeout"
        return self.state_split(), REWARD_STEP + progress, False, "step"

    def optimal_steps(self) -> int:
        """Iteraciones que necesita la bisección exacta (λ = 1/2)."""
        return max(1, math.ceil(math.log2(self.width0 / self.tol)))
