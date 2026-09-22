"""Entorno de RL para *aprender a interpolar*: dónde poner los nodos y cuántos.

Objetivo: aproximar f en [−1, 1] con un polinomio interpolante de modo que
máx|f − pₙ| < tol usando el menor número de nodos. Hay dos decisiones:

1. **Familia de nodos** (una vez por episodio, cabeza "bandido"): γ ∈ Γ mezcla
   nodos equiespaciados (γ = 0) con nodos de Chebyshev–Lobatto (γ = 1). El
   agente no sabe qué es Chebyshev; solo ve el retorno del episodio.
2. **Cuántos nodos** (cabeza Q, un paso por nodo): tras cada nodo añadido el
   agente observa solo cosas medibles sin conocer f:
   * la diferencia entre los dos últimos interpolantes, máx|pₙ − pₙ₋₁|, frente
     a la tolerancia (4 cubos): es el estimador de error que usa cualquiera
     que no conoce f;
   * si esa diferencia bajó o subió respecto al paso anterior (tendencia);
   * si lleva pocos (n ≤ 6) o muchos nodos.
   Y decide: AÑADIR un nodo (coste −1 + log₁₀(error anterior / error nuevo),
   shaping por potencial con el error real, que el entorno conoce y el agente
   no ve) o PARAR (+10 si el error real es < tol, −10 si no).

Lo que emerge: la cabeza de nodos prefiere γ = 1 porque con equiespaciados
las funciones tipo Runge **empeoran** al añadir nodos (progreso negativo hasta
agotar el presupuesto); y la cabeza de parada aprende que cuando dos
interpolantes consecutivos difieren en mucho menos que la tolerancia y la
diferencia va bajando, el error real ya está por debajo.

Simplificación deliberada: al añadir un nodo se recalculan todos (la familia
cambia con n), y eso cuesta 1, no n. El coste mide "grado del polinomio",
no evaluaciones de f. Se dice en el README.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Callable

from .interp import barycentric_weights, interpolate, nodes

GAMMAS: tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.0)
ADD, STOP = 0, 1
ACTION_NAMES = ("añadir nodo", "parar y entregar")
ACTION_SHORT = ("+", "■")
N_ACTIONS = 2

EST_EDGES = (-2.0, 0.0, 2.0)  # log10(est/tol): <−2, −2..0, 0..2, ≥2
EST_LABELS = ("est<tol/100", "tol/100–tol", "tol–100tol", "est≥100tol")
TREND_LABELS = ("baja", "sube")
N_LATE = 6
N_MIN = 2
N_MAX = 40
GRID = 201

REWARD_NODE = -1.0
REWARD_RIGHT = 10.0
REWARD_WRONG = -10.0
REWARD_TIMEOUT = -5.0


def est_bucket(est: float, tol: float) -> int:
    v = math.log10(max(est, 1e-300) / tol)
    for i, e in enumerate(EST_EDGES):
        if v < e:
            return i
    return len(EST_EDGES)


def theoretical_action(eb: int, trend: int, late: int) -> tuple[int, ...]:
    """Regla del libro: parar cuando el estimador |pₙ − pₙ₋₁| está muy por
    debajo de la tolerancia y sigue bajando; seguir en otro caso. En la franja
    tol/100–tol con tendencia a la baja, cualquiera de las dos vale."""
    if eb == 0:
        return (STOP,) if trend == 0 else (STOP, ADD)
    if eb == 1:
        return (STOP, ADD) if trend == 0 else (ADD,)
    return (ADD,)


@dataclass
class Problem:
    name: str
    f: Callable[[float], float]
    analytic: bool  # True si es entera o su singularidad está lejos (equiespaciados también convergen)


def make_problem(rng: random.Random) -> Problem:
    kind = rng.choice(["runge", "runge", "pole", "exp", "sin", "sqrt"])
    if kind == "runge":
        a = rng.uniform(4.0, 8.0)  # polo en ±i/√a dentro de la región de Runge: equiespaciados divergen
        return Problem(f"1/(1+{a:.1f}x²)", lambda x, a=a: 1.0 / (1.0 + a * x * x), False)
    if kind == "pole":
        c = rng.choice([-1, 1]) * rng.uniform(1.1, 1.5)
        return Problem(f"1/(x−{c:+.2f})", lambda x, c=c: 1.0 / (x - c), False)
    if kind == "exp":
        k = rng.uniform(0.5, 3.0)
        return Problem(f"e^({k:.1f}x)", lambda x, k=k: math.exp(k * x), True)
    if kind == "sin":
        k, c = rng.uniform(1.0, 6.0), rng.uniform(0, math.pi)
        return Problem(f"sin({k:.1f}x+{c:.1f})", lambda x, k=k, c=c: math.sin(k * x + c), True)
    c = rng.uniform(1.1, 1.4)
    return Problem(f"√(x+{c:.2f})", lambda x, c=c: math.sqrt(x + c), False)


@dataclass
class InterpEnv:
    tol: float = 1e-5
    seed: int = 0
    rng: random.Random = field(init=False)
    problem: Problem = field(init=False)
    gamma: float = 1.0
    n: int = 0
    x: list[float] = field(default_factory=list)
    y: list[float] = field(default_factory=list)
    w: list[float] = field(default_factory=list)
    grid: list[float] = field(default_factory=list)
    fgrid: list[float] = field(default_factory=list)
    pgrid: list[float] = field(default_factory=list)
    est: float = math.inf
    est_prev: float = math.inf
    err: float = math.inf
    err_hist: list[float] = field(default_factory=list)
    est_hist: list[float] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.rng = random.Random(self.seed)
        self.grid = [-1.0 + 2.0 * k / (GRID - 1) for k in range(GRID)]
        self.reset()

    def reset(self) -> None:
        self.problem = make_problem(self.rng)
        self.fgrid = [self.problem.f(t) for t in self.grid]
        self.est = self.est_prev = math.inf
        self.err_hist, self.est_hist = [], []
        self.pgrid = []
        self.n = 0

    def set_gamma(self, gamma: float) -> tuple:
        """Fija la familia de nodos y construye el interpolante inicial (n = N_MIN)."""
        self.gamma = gamma
        self.n = N_MIN - 1
        self._build()
        self._build()  # dos interpolantes: hace falta un par para el estimador
        return self.state()

    def _build(self) -> None:
        self.n += 1
        self.x = nodes(self.n, self.gamma)
        self.y = [self.problem.f(t) for t in self.x]
        self.w = barycentric_weights(self.x)
        old = self.pgrid
        self.pgrid = [interpolate(self.x, self.y, self.w, t) for t in self.grid]
        self.err = max(abs(a - b) for a, b in zip(self.fgrid, self.pgrid))
        self.est_prev = self.est
        self.est = max(abs(a - b) for a, b in zip(old, self.pgrid)) if old else math.inf
        self.err_hist.append(self.err)
        self.est_hist.append(self.est)

    @property
    def scale(self) -> float:
        return max(1.0, max(abs(v) for v in self.fgrid))

    def state(self) -> tuple:
        eb = est_bucket(self.est / self.scale, self.tol)
        trend = 0 if self.est <= self.est_prev else 1
        late = 1 if self.n > N_LATE else 0
        return (eb, trend, late)

    def step(self, action: int) -> tuple[tuple, float, bool, str]:
        if action == STOP:
            ok = self.err / self.scale < self.tol
            return self.state(), (REWARD_RIGHT if ok else REWARD_WRONG), True, ("stop_ok" if ok else "stop_bad")
        old = self.err
        self._build()
        progress = math.log10(max(old, 1e-300) / max(self.err, 1e-300))
        progress = max(-3.0, min(3.0, progress))
        if not math.isfinite(self.err) or self.err > 1e12:
            return self.state(), REWARD_NODE + progress + REWARD_TIMEOUT, True, "overflow"
        if self.n >= N_MAX:
            return self.state(), REWARD_NODE + progress + REWARD_TIMEOUT, True, "timeout"
        return self.state(), REWARD_NODE + progress, False, "add"

    def optimal_nodes(self, gamma: float = 1.0) -> int | None:
        """Menor n con máx|f − pₙ|/escala < tol para la familia γ (None si no lo logra antes de N_MAX)."""
        for n in range(N_MIN, N_MAX + 1):
            x = nodes(n, gamma)
            y = [self.problem.f(t) for t in x]
            w = barycentric_weights(x)
            err = max(abs(fv - interpolate(x, y, w, t)) for t, fv in zip(self.grid, self.fgrid))
            if err / self.scale < self.tol:
                return n
        return None
