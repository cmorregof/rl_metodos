"""Base de conocimiento: convergencia cuadrática local del método de Newton.

Teorema. Sea f ∈ C² en un entorno de r con f(r) = 0 y f'(r) ≠ 0. Existe δ > 0
tal que, para todo x₀ ∈ [r − δ, r + δ], la sucesión xₙ₊₁ = xₙ − f(xₙ)/f'(xₙ)
está bien definida, permanece en el intervalo, converge a r y

    |xₙ₊₁ − r| ≤ (M / 2m) · |xₙ − r|²,   m = min |f'|, M = max |f''|,

es decir, converge con orden al menos 2. La pieza clave es el teorema de Taylor
con resto de Lagrange (proyecto 03) con n = 1.

Distractores: "Newton converge desde cualquier x₀" (falso: arctan desde 2),
"si f'(r) = 0 sigue siendo cuadrático" (falso: raíces múltiples dan orden 1),
"por Bolzano existe raíz, luego Newton converge" (non sequitur), etc.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Step:
    key: str
    title: str
    text: str
    deps: tuple[str, ...] = field(default_factory=tuple)
    distractor: bool = False
    is_qed: bool = False


STEPS: list[Step] = [
    Step("H1", "Hipótesis", "f ∈ C² en un entorno de r, f(r) = 0 y f'(r) ≠ 0 (raíz simple)."),
    Step(
        "NBHD",
        "Entorno seguro",
        "Por continuidad de f' y f'' existe δ₀ > 0 con |f'(x)| ≥ m > 0 y |f''(x)| ≤ M en I = [r − δ₀, r + δ₀].",
        ("H1",),
    ),
    Step(
        "DEF",
        "Definición",
        "Para xₙ ∈ I el paso xₙ₊₁ = xₙ − f(xₙ)/f'(xₙ) está bien definido porque f'(xₙ) ≠ 0.",
        ("NBHD",),
    ),
    Step(
        "TAYLOR",
        "Taylor (n = 1)",
        "Por el teorema de Taylor con resto de Lagrange en xₙ: 0 = f(r) = f(xₙ) + f'(xₙ)(r − xₙ) + ½f''(ξₙ)(r − xₙ)², con ξₙ entre xₙ y r.",
        ("H1",),
    ),
    Step(
        "ERRID",
        "Identidad del error",
        "Dividiendo por f'(xₙ) y usando la definición: r − xₙ₊₁ = −f''(ξₙ)/(2f'(xₙ)) · (r − xₙ)².",
        ("TAYLOR", "DEF"),
    ),
    Step(
        "QUAD",
        "Cota cuadrática",
        "Con eₙ = xₙ − r: |eₙ₊₁| ≤ (M / 2m)·|eₙ|² =: C·|eₙ|² mientras xₙ ∈ I.",
        ("ERRID", "NBHD"),
    ),
    Step(
        "DELTA",
        "Elección de δ",
        "Tómese δ ≤ δ₀ con C·δ < 1. Si |e₀| ≤ δ entonces |e₁| ≤ C·δ·|e₀| < |e₀| ≤ δ.",
        ("QUAD",),
    ),
    Step(
        "INV",
        "Invariante",
        "Por inducción xₙ ∈ [r − δ, r + δ] para todo n y |eₙ₊₁| ≤ (C·δ)·|eₙ|.",
        ("DELTA",),
    ),
    Step(
        "CONV",
        "Convergencia",
        "|eₙ| ≤ (C·δ)ⁿ·|e₀| → 0 porque C·δ < 1, luego xₙ → r.",
        ("INV",),
    ),
    Step(
        "ORDER",
        "Orden 2",
        "De la cota cuadrática, |eₙ₊₁|/|eₙ|² ≤ C: la convergencia es de orden al menos 2 (los dígitos correctos se duplican).",
        ("QUAD", "CONV"),
    ),
    Step(
        "QED",
        "Conclusión",
        "Newton converge a r desde todo x₀ con |x₀ − r| ≤ δ, con orden cuadrático. ∎",
        ("CONV", "ORDER"),
        is_qed=True,
    ),
    # --- Distractores ----------------------------------------------------
    Step("D_GLOBAL", "Error clásico", "Newton converge desde cualquier x₀.  (¡falso: arctan desde x₀ = 2 diverge!)", distractor=True),
    Step("D_MULT", "Falso", "Si f'(r) = 0 (raíz múltiple) la convergencia sigue siendo cuadrática.", distractor=True),
    Step("D_BOLZ", "Non sequitur", "Por Bolzano existe una raíz r; luego Newton converge a ella.", distractor=True),
    Step("D_LINEAR", "Insuficiente", "g(x) = x − f(x)/f'(x) tiene |g'(r)| < 1, luego converge linealmente y basta.", distractor=True),
    Step("D_MONO", "Insuficiente", "f es monótona, luego la sucesión de Newton es monótona y converge.", distractor=True),
    Step("D_FP0", "Absurdo", "Si f'(xₙ) = 0 se toma xₙ₊₁ = xₙ y se continúa.", distractor=True),
    Step("D_BISECT", "Otro método", "Se conserva el subintervalo con cambio de signo (bisección).", distractor=True),
]

KEY_TO_INDEX: dict[str, int] = {s.key: i for i, s in enumerate(STEPS)}
N_STEPS = len(STEPS)
CORE_KEYS: tuple[str, ...] = tuple(s.key for s in STEPS if not s.distractor)


def step_by_key(key: str) -> Step:
    return STEPS[KEY_TO_INDEX[key]]


def deps_mask(step: Step) -> int:
    m = 0
    for d in step.deps:
        m |= 1 << KEY_TO_INDEX[d]
    return m


DEPS_MASK: list[int] = [deps_mask(s) for s in STEPS]
QED_INDEX = KEY_TO_INDEX["QED"]


def _closure(key: str, acc: set[str]) -> set[str]:
    acc.add(key)
    for d in step_by_key(key).deps:
        _closure(d, acc)
    return acc


REQUIRED_KEYS: frozenset[str] = frozenset(_closure("QED", set()))
MIN_PROOF_LENGTH = len(REQUIRED_KEYS)
