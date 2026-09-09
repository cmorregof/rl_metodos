"""Base de conocimiento: convergencia local de orden φ del método de la secante.

Teorema. Sea f ∈ C² en un entorno de r con f(r) = 0 y f'(r) ≠ 0. Existe δ > 0
tal que, para todo par x₀, x₁ ∈ [r − δ, r + δ] con x₀ ≠ x₁, la sucesión

    xₙ₊₁ = xₙ − f(xₙ) · (xₙ − xₙ₋₁) / (f(xₙ) − f(xₙ₋₁))

está bien definida, permanece en el intervalo, converge a r y

    |xₙ₊₁ − r| ≤ (M / 2m) · |xₙ − r| · |xₙ₋₁ − r|,   m = min |f'|, M = max |f''|,

de donde el orden de convergencia es φ = (1 + √5)/2 ≈ 1.618. La pieza clave es
la forma de Newton del interpolante con diferencias divididas (que sustituye al
resto de Lagrange del proyecto 03/04) y la recurrencia de Fibonacci que
produce la cota producto.

Distractores: "es Newton con la derivada aproximada, luego es cuadrático"
(falso: φ < 2), "es un método de punto fijo xₙ₊₁ = g(xₙ), luego converge
linealmente" (falso: depende de dos puntos), "como los puntos encierran la
raíz el error es ≤ (b − a)/2ⁿ" (eso es bisección; la regula falsi se
estanca), etc.
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
        "Para xₙ ≠ xₙ₋₁ en I, f(xₙ) ≠ f(xₙ₋₁) (f es inyectiva en I porque f' ≠ 0), luego el paso de la secante xₙ₊₁ = xₙ − f(xₙ)(xₙ − xₙ₋₁)/(f(xₙ) − f(xₙ₋₁)) está bien definido.",
        ("NBHD",),
    ),
    Step(
        "INTERP",
        "Forma de Newton",
        "Con diferencias divididas, para todo x: f(x) = f(xₙ) + f[xₙ₋₁, xₙ](x − xₙ) + f[xₙ₋₁, xₙ, x](x − xₙ)(x − xₙ₋₁). En x = r el lado izquierdo es 0.",
        ("H1",),
    ),
    Step(
        "ERRID",
        "Identidad del error",
        "Dividiendo por f[xₙ₋₁, xₙ] y usando la definición: r − xₙ₊₁ = −(f[xₙ₋₁, xₙ, r] / f[xₙ₋₁, xₙ]) · (r − xₙ)(r − xₙ₋₁).",
        ("INTERP", "DEF"),
    ),
    Step(
        "MVT",
        "Valor medio",
        "Si xₙ₋₁, xₙ ∈ I: f[xₙ₋₁, xₙ] = f'(ξₙ) con |f'(ξₙ)| ≥ m, y f[xₙ₋₁, xₙ, r] = ½f''(ηₙ) con |f''(ηₙ)| ≤ M (Rolle iterado, como en el resto de Lagrange).",
        ("NBHD",),
    ),
    Step(
        "PROD",
        "Cota producto",
        "Con eₙ = xₙ − r: |eₙ₊₁| ≤ (M / 2m)·|eₙ|·|eₙ₋₁| =: C·|eₙ|·|eₙ₋₁| mientras xₙ₋₁, xₙ ∈ I.",
        ("ERRID", "MVT"),
    ),
    Step(
        "DELTA",
        "Elección de δ",
        "Tómese δ ≤ δ₀ con C·δ < 1. Si |e₀|, |e₁| ≤ δ entonces |e₂| ≤ C·|e₁|·|e₀| ≤ (C·δ)·|e₁| < δ.",
        ("PROD",),
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
        "|eₙ| ≤ (C·δ)ⁿ⁻¹·|e₁| → 0 porque C·δ < 1, luego xₙ → r.",
        ("INV",),
    ),
    Step(
        "FIB",
        "Fibonacci",
        "Sea dₙ = C·|eₙ| ≤ d := C·δ < 1. De la cota producto, dₙ₊₁ ≤ dₙ·dₙ₋₁, luego dₙ ≤ d^{Fₙ} con Fₙ los números de Fibonacci (F₀ = F₁ = 1).",
        ("PROD", "DELTA"),
    ),
    Step(
        "ORDER",
        "Orden φ",
        "Si |eₙ₊₁| ~ K·|eₙ|ᵖ, la cota producto da p = 1 + 1/p, es decir p² = p + 1: p = φ = (1 + √5)/2 ≈ 1.618 (= lím Fₙ₊₁/Fₙ). Superlineal, pero no cuadrático.",
        ("FIB", "CONV"),
    ),
    Step(
        "QED",
        "Conclusión",
        "La secante converge a r desde todo par x₀, x₁ con |xᵢ − r| ≤ δ, con orden φ ≈ 1.618. ∎",
        ("CONV", "ORDER"),
        is_qed=True,
    ),
    # --- Distractores ----------------------------------------------------
    Step("D_QUAD", "Error clásico", "La secante aproxima f'(xₙ) por un cociente incremental, luego es Newton y hereda el orden 2.  (¡falso: el orden es φ < 2!)", distractor=True),
    Step("D_FIXPT", "Falso", "La secante es un método de punto fijo xₙ₊₁ = g(xₙ), luego converge linealmente con razón |g'(r)|.", distractor=True),
    Step("D_BRACKET", "Otro método", "Como xₙ y xₙ₋₁ encierran la raíz, |eₙ| ≤ (b − a)/2ⁿ.  (eso es bisección; la regula falsi se estanca)", distractor=True),
    Step("D_GLOBAL", "Error clásico", "La secante converge desde cualquier par x₀, x₁.  (¡falso: en ∛x oscila sin converger!)", distractor=True),
    Step("D_BOLZ", "Non sequitur", "Por Bolzano existe una raíz r; luego la secante converge a ella.", distractor=True),
    Step("D_EQUAL", "Absurdo", "Si f(xₙ) = f(xₙ₋₁) se toma xₙ₊₁ = xₙ y se continúa.", distractor=True),
    Step("D_BOUNDED", "Insuficiente", "Los iterados están acotados, luego convergen.", distractor=True),
    Step("D_MULT", "Falso", "Si f'(r) = 0 (raíz múltiple) el orden sigue siendo φ.", distractor=True),
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
