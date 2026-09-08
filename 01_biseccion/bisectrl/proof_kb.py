"""Base de conocimiento de la demostración de convergencia de la bisección.

Cada paso es un lema/afirmación con dependencias. Un paso es *válido* en un
estado si todas sus dependencias ya fueron establecidas y el paso no se ha
usado. Los pasos marcados como `distractor` nunca son válidos: son lemas
falsos, irrelevantes o de otro método (Newton, derivadas...) que el agente
tiene que aprender a descartar. El más traicionero es el *non sequitur*
"por Bolzano existe una raíz, luego el método converge a ella": la existencia
de la raíz no dice nada sobre la convergencia de la sucesión.

Teorema (convergencia de la bisección). Sea f continua en [a, b] con
f(a)·f(b) < 0. Las sucesiones a_n, b_n, m_n = (a_n + b_n)/2 generadas por la
bisección satisfacen m_n → c con f(c) = 0 y |m_n − c| ≤ (b − a) / 2^{n+1}.
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
    Step("H1", "Hipótesis 1", "f es continua en [a, b]."),
    Step("H2", "Hipótesis 2", "f(a) · f(b) < 0 (cambio de signo)."),
    Step(
        "DEF",
        "Definición",
        "Sea a₀ = a, b₀ = b, mₙ = (aₙ + bₙ)/2. Si f(aₙ)·f(mₙ) ≤ 0 tomar "
        "[aₙ₊₁, bₙ₊₁] = [aₙ, mₙ]; si no, [mₙ, bₙ].",
        ("H2",),
    ),
    Step(
        "INV",
        "Invariante",
        "Por inducción sobre n: f(aₙ) · f(bₙ) ≤ 0 para todo n (el cambio de "
        "signo nunca se pierde).",
        ("DEF", "H2"),
    ),
    Step(
        "WIDTH",
        "Longitud",
        "Cada paso divide la longitud a la mitad: bₙ − aₙ = (b − a) / 2ⁿ.",
        ("DEF",),
    ),
    Step(
        "MONO_A",
        "Monotonía de aₙ",
        "(aₙ) es no decreciente y acotada superiormente por b.",
        ("DEF",),
    ),
    Step(
        "MONO_B",
        "Monotonía de bₙ",
        "(bₙ) es no creciente y acotada inferiormente por a.",
        ("DEF",),
    ),
    Step(
        "CONV_A",
        "Convergencia de aₙ",
        "Toda sucesión monótona y acotada converge: aₙ → α.",
        ("MONO_A",),
    ),
    Step(
        "CONV_B",
        "Convergencia de bₙ",
        "Análogamente bₙ → β.",
        ("MONO_B",),
    ),
    Step(
        "SAME",
        "Mismo límite",
        "β − α = lim (bₙ − aₙ) = lim (b − a)/2ⁿ = 0, luego α = β =: c.",
        ("CONV_A", "CONV_B", "WIDTH"),
    ),
    Step(
        "CONT",
        "Continuidad",
        "Por continuidad de f: f(aₙ) · f(bₙ) → f(c) · f(c) = f(c)².",
        ("H1", "SAME"),
    ),
    Step(
        "ROOT",
        "Raíz",
        "El límite de una sucesión ≤ 0 es ≤ 0: f(c)² ≤ 0, así que f(c) = 0.",
        ("INV", "CONT"),
    ),
    Step(
        "MID",
        "Cota de error",
        "c ∈ [aₙ, bₙ] para todo n, luego |mₙ − c| ≤ (bₙ − aₙ)/2 = (b − a)/2ⁿ⁺¹.",
        ("WIDTH", "SAME"),
    ),
    Step(
        "QED",
        "Conclusión",
        "Por tanto mₙ → c, f(c) = 0 y el error decrece geométricamente. ∎",
        ("ROOT", "MID"),
        is_qed=True,
    ),
    # --- Distractores: nunca son válidos ---------------------------------
    Step(
        "D_BOLZ",
        "Non sequitur",
        "Por Bolzano existe c con f(c) = 0; luego mₙ → c.  (¡existencia no implica convergencia!)",
        distractor=True,
    ),
    Step("D_DERIV", "Distractor", "Supongamos que f es derivable en (a, b).", distractor=True),
    Step("D_NEWTON", "Distractor", "Iteramos xₙ₊₁ = xₙ − f(xₙ)/f'(xₙ) (Newton).", distractor=True),
    Step("D_TVM", "Distractor", "Por el teorema del valor medio existe ξ con f'(ξ) = 0.", distractor=True),
    Step("D_LIP", "Distractor", "f es Lipschitz, luego la sucesión es de Cauchy.", distractor=True),
    Step("D_DIV", "Distractor", "La sucesión (aₙ) diverge a +∞.", distractor=True),
    Step("D_UNIQ", "Distractor", "La raíz c es única en [a, b].", distractor=True),
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
# Pasos estrictamente necesarios para llegar a QED (cierre transitivo de deps).
def _closure(key: str, acc: set[str]) -> set[str]:
    acc.add(key)
    for d in step_by_key(key).deps:
        _closure(d, acc)
    return acc


REQUIRED_KEYS: frozenset[str] = frozenset(_closure("QED", set()))
MIN_PROOF_LENGTH = len(REQUIRED_KEYS)
