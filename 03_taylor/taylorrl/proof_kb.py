"""Base de conocimiento: teorema de Taylor con resto de Lagrange y convergencia
de la serie de Taylor.

Teorema. Sea f de clase Cⁿ⁺¹ en un intervalo I que contiene a y x, y
Pₙ(x) = Σₖ₌₀ⁿ f⁽ᵏ⁾(a)(x − a)ᵏ/k!. Entonces existe ξ entre a y x con

    f(x) = Pₙ(x) + f⁽ⁿ⁺¹⁾(ξ)(x − a)ⁿ⁺¹/(n + 1)!      (resto de Lagrange).

Si además |f⁽ᵏ⁾| ≤ M en I para todo k, entonces |Rₙ(x)| ≤ M|x − a|ⁿ⁺¹/(n+1)! → 0
y la serie de Taylor converge a f(x).

Distractores: errores clásicos como "f ∈ C^∞ ⇒ su serie converge a f" (falso:
e^{−1/x²}), "los términos tienden a 0 ⇒ la serie converge" (falso: armónica),
"por el teorema del valor medio Rₙ → 0" (non sequitur: el TVM es el caso n = 0).
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
    Step("H1", "Hipótesis", "f es de clase Cⁿ⁺¹ en un intervalo I que contiene a a y a x (x ≠ a)."),
    Step(
        "DEF",
        "Polinomio y resto",
        "Pₙ(t) = Σₖ₌₀ⁿ f⁽ᵏ⁾(a)(t − a)ᵏ/k!  y  Rₙ(t) = f(t) − Pₙ(t).",
        ("H1",),
    ),
    Step(
        "MATCH",
        "Derivadas coinciden en a",
        "Pₙ⁽ᵏ⁾(a) = f⁽ᵏ⁾(a) para k = 0, …, n, luego Rₙ⁽ᵏ⁾(a) = 0 para k = 0, …, n.",
        ("DEF",),
    ),
    Step(
        "AUX",
        "Función auxiliar",
        "Sea K la constante con Rₙ(x) = K(x − a)ⁿ⁺¹/(n+1)! y g(t) = Rₙ(t) − K(t − a)ⁿ⁺¹/(n+1)!.",
        ("DEF",),
    ),
    Step(
        "AUXZ",
        "Ceros de g",
        "g(x) = 0 por la elección de K, y g⁽ᵏ⁾(a) = 0 para k = 0, …, n.",
        ("AUX", "MATCH"),
    ),
    Step(
        "ROLLE1",
        "Rolle",
        "g(a) = g(x) = 0 y g ∈ C¹ ⇒ por Rolle existe ξ₁ entre a y x con g'(ξ₁) = 0.",
        ("AUXZ", "H1"),
    ),
    Step(
        "ROLLEN",
        "Rolle iterado",
        "g'(a) = g'(ξ₁) = 0 ⇒ existe ξ₂ con g''(ξ₂) = 0; repitiendo n+1 veces existe ξ con g⁽ⁿ⁺¹⁾(ξ) = 0.",
        ("ROLLE1",),
    ),
    Step(
        "DERIV",
        "Derivada (n+1)-ésima",
        "Pₙ⁽ⁿ⁺¹⁾ ≡ 0 (grado n), luego g⁽ⁿ⁺¹⁾(t) = f⁽ⁿ⁺¹⁾(t) − K.",
        ("AUX", "DEF"),
    ),
    Step(
        "LAGR",
        "Resto de Lagrange",
        "0 = g⁽ⁿ⁺¹⁾(ξ) = f⁽ⁿ⁺¹⁾(ξ) − K ⇒ K = f⁽ⁿ⁺¹⁾(ξ), es decir Rₙ(x) = f⁽ⁿ⁺¹⁾(ξ)(x − a)ⁿ⁺¹/(n+1)!.",
        ("ROLLEN", "DERIV"),
    ),
    Step(
        "BOUND",
        "Cota del resto",
        "Si |f⁽ⁿ⁺¹⁾| ≤ M en I: |Rₙ(x)| ≤ M|x − a|ⁿ⁺¹/(n+1)!.",
        ("LAGR",),
    ),
    Step(
        "FACT",
        "Lema del factorial",
        "Para todo r > 0: rⁿ⁺¹/(n+1)! → 0 cuando n → ∞ (el cociente entre términos consecutivos r/(n+2) → 0).",
    ),
    Step(
        "CONV",
        "Convergencia de la serie",
        "Si |f⁽ᵏ⁾| ≤ M en I para todo k, entonces |Rₙ(x)| ≤ M|x − a|ⁿ⁺¹/(n+1)! → 0 y Σ f⁽ᵏ⁾(a)(x − a)ᵏ/k! = f(x).",
        ("BOUND", "FACT"),
    ),
    Step(
        "QED",
        "Conclusión",
        "f(x) = Pₙ(x) + f⁽ⁿ⁺¹⁾(ξ)(x − a)ⁿ⁺¹/(n+1)!, y con derivadas uniformemente acotadas la serie de Taylor converge a f. ∎",
        ("LAGR", "CONV"),
        is_qed=True,
    ),
    # --- Distractores ----------------------------------------------------
    Step("D_CINF", "Error clásico", "f ∈ C^∞, luego su serie de Taylor converge a f.  (¡falso: e^{−1/x²} en 0!)", distractor=True),
    Step("D_TERM", "Falso", "Los términos f⁽ᵏ⁾(a)(x − a)ᵏ/k! tienden a 0, luego la serie converge.", distractor=True),
    Step("D_TVM", "Non sequitur", "Por el teorema del valor medio f(x) − f(a) = f'(ξ)(x − a), luego Rₙ(x) → 0.", distractor=True),
    Step("D_RADIUS", "Falso", "Toda serie de Taylor converge en todo ℝ.", distractor=True),
    Step("D_BOLZ", "Irrelevante", "Por Bolzano existe c con f(c) = 0.", distractor=True),
    Step("D_NEWTON", "Otro método", "Iteramos xₙ₊₁ = xₙ − f(xₙ)/f'(xₙ) (Newton).", distractor=True),
    Step("D_LHOP", "Irrelevante", "Por L'Hôpital, lim Rₙ(x)/(x − a)ⁿ = 0.", distractor=True),
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
