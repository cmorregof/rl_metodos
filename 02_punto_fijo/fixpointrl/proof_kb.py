"""Base de conocimiento: demostración del teorema del punto fijo (Banach en
[a, b]) y de la convergencia de la iteración xₙ₊₁ = g(xₙ).

Teorema. Sea g continua en [a, b] con g([a, b]) ⊆ [a, b] y contractiva:
|g(x) − g(y)| ≤ k |x − y| con 0 ≤ k < 1. Entonces g tiene un único punto fijo
p, la iteración xₙ₊₁ = g(xₙ) converge a p para todo x₀ ∈ [a, b], y

    |xₙ − p| ≤ kⁿ · max(x₀ − a, b − x₀)        (cota a priori)
    |xₙ − p| ≤ kⁿ / (1 − k) · |x₁ − x₀|        (cota a posteriori)

Los distractores incluyen errores clásicos: creer que |g'(p)| < 1 basta para
cualquier x₀ (solo garantiza convergencia local), que k = 1 es suficiente, que
una sucesión acotada converge, o el *non sequitur* "existe punto fijo, luego
la iteración converge a él".
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
    Step("H1", "Hipótesis 1", "g es continua en [a, b] y g([a, b]) ⊆ [a, b]."),
    Step(
        "H2",
        "Hipótesis 2",
        "g es contracción: existe k ∈ [0, 1) con |g(x) − g(y)| ≤ k·|x − y| para todo x, y ∈ [a, b].",
    ),
    Step(
        "AUX",
        "Función auxiliar",
        "h(x) = g(x) − x es continua, h(a) = g(a) − a ≥ 0 y h(b) = g(b) − b ≤ 0.",
        ("H1",),
    ),
    Step(
        "EXIST",
        "Existencia",
        "Por Bolzano aplicado a h existe p ∈ [a, b] con h(p) = 0, es decir g(p) = p.",
        ("AUX",),
    ),
    Step(
        "UNIQ",
        "Unicidad",
        "Si p y q son puntos fijos, |p − q| = |g(p) − g(q)| ≤ k·|p − q| con k < 1, luego p = q.",
        ("EXIST", "H2"),
    ),
    Step(
        "DEF",
        "Definición",
        "Sea x₀ ∈ [a, b] y xₙ₊₁ = g(xₙ). Por H1 e inducción, xₙ ∈ [a, b] para todo n.",
        ("H1",),
    ),
    Step(
        "ERR",
        "Contracción del error",
        "|xₙ₊₁ − p| = |g(xₙ) − g(p)| ≤ k·|xₙ − p|.",
        ("DEF", "EXIST", "H2"),
    ),
    Step(
        "GEO",
        "Decaimiento geométrico",
        "Por inducción sobre n: |xₙ − p| ≤ kⁿ·|x₀ − p|.",
        ("ERR",),
    ),
    Step(
        "LIM",
        "Convergencia",
        "Como 0 ≤ k < 1, kⁿ → 0, así que xₙ → p.",
        ("GEO",),
    ),
    Step(
        "PRIORI",
        "Cota a priori",
        "p ∈ [a, b] ⇒ |x₀ − p| ≤ max(x₀ − a, b − x₀), luego |xₙ − p| ≤ kⁿ·max(x₀ − a, b − x₀).",
        ("GEO",),
    ),
    Step(
        "STEP",
        "Pasos consecutivos",
        "|xₙ₊₁ − xₙ| = |g(xₙ) − g(xₙ₋₁)| ≤ k·|xₙ − xₙ₋₁| ≤ kⁿ·|x₁ − x₀|.",
        ("DEF", "H2"),
    ),
    Step(
        "POST",
        "Cota a posteriori",
        "|xₙ − p| ≤ Σᵢ₌ₙ^∞ |xᵢ₊₁ − xᵢ| ≤ kⁿ·|x₁ − x₀|·Σⱼ kʲ = kⁿ/(1 − k)·|x₁ − x₀| (serie geométrica y xₙ → p).",
        ("STEP", "LIM"),
    ),
    Step(
        "QED",
        "Conclusión",
        "Existe un único punto fijo p, xₙ → p para todo x₀ ∈ [a, b] y valen las cotas a priori y a posteriori. ∎",
        ("UNIQ", "LIM", "PRIORI", "POST"),
        is_qed=True,
    ),
    # --- Distractores: nunca son válidos ---------------------------------
    Step("D_LOCAL", "Error clásico", "|g'(p)| < 1, luego la iteración converge desde cualquier x₀.  (¡solo garantiza convergencia local!)", distractor=True),
    Step("D_BOLZ", "Non sequitur", "Por Bolzano existe un punto fijo p; luego xₙ → p.", distractor=True),
    Step("D_BOUND", "Falso", "La sucesión (xₙ) está acotada en [a, b], luego converge.", distractor=True),
    Step("D_MONO", "Insuficiente", "g es creciente, luego (xₙ) es monótona y converge a p.", distractor=True),
    Step("D_K1", "Falso", "Basta con k = 1: si |g(x) − g(y)| ≤ |x − y| la iteración converge.", distractor=True),
    Step("D_NEWTON", "Otro método", "Iteramos xₙ₊₁ = xₙ − f(xₙ)/f'(xₙ) (Newton).", distractor=True),
    Step("D_DERIV", "Irrelevante", "Supongamos g derivable con g'(p) = 0.", distractor=True),
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
