"""Base de conocimiento: teorema del error de interpolación polinómica.

Teorema. Sea f ∈ Cⁿ⁺¹[a, b], x₀ < … < xₙ nodos distintos en [a, b] y pₙ el
único polinomio de grado ≤ n con pₙ(xᵢ) = f(xᵢ). Para cada x ∈ [a, b] existe
ξ = ξ(x) ∈ (a, b) con

    f(x) − pₙ(x) = f⁽ⁿ⁺¹⁾(ξ) · w(x) / (n + 1)!,   w(t) = ∏ᵢ (t − xᵢ).

En consecuencia máx|f − pₙ| ≤ M·máx|w|/(n+1)! con M = máx|f⁽ⁿ⁺¹⁾|, y con
los nodos de Chebyshev (ceros de Tₙ₊₁ en [−1, 1]) máx|w| = 2⁻ⁿ, el mínimo
posible.

Distractores: los errores clásicos de interpolación. «Más nodos ⇒ menor
error» (falso: Runge con equiespaciados), «f continua ⇒ pₙ → f» (falso:
Faber), «por Weierstrass hay un polinomio cercano, luego el interpolante
converge» (non sequitur: el interpolante no es ese polinomio), la confusión
con el resto de Taylor, y «el error es 0 en los nodos, luego es pequeño».
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
    Step("H1", "Hipótesis 1", "f ∈ Cⁿ⁺¹[a, b]."),
    Step("H2", "Hipótesis 2", "x₀ < x₁ < … < xₙ son n + 1 nodos distintos en [a, b]."),
    Step(
        "EXIST",
        "Existencia y unicidad",
        "Existe un único pₙ de grado ≤ n con pₙ(xᵢ) = f(xᵢ): la base de Lagrange ℓᵢ lo construye y "
        "dos soluciones difieren en un polinomio de grado ≤ n con n + 1 ceros, luego nulo.",
        ("H2",),
    ),
    Step(
        "DEFW",
        "Polinomio nodal",
        "w(t) = ∏ᵢ₌₀ⁿ (t − xᵢ) tiene grado n + 1, coeficiente principal 1 y w⁽ⁿ⁺¹⁾ ≡ (n + 1)!.",
        ("H2",),
    ),
    Step(
        "FIX",
        "Fijar x",
        "Sea x ∈ [a, b] distinto de todos los nodos (si x = xᵢ ambos lados de la fórmula son 0). Entonces w(x) ≠ 0.",
        ("DEFW",),
    ),
    Step(
        "AUX",
        "Función auxiliar",
        "Sea g(t) = f(t) − pₙ(t) − [f(x) − pₙ(x)]·w(t)/w(x). Por H1, g ∈ Cⁿ⁺¹[a, b].",
        ("H1", "EXIST", "FIX"),
    ),
    Step(
        "ZEROS",
        "n + 2 ceros",
        "g se anula en los n + 1 nodos (allí f = pₙ y w = 0) y en x (por construcción): n + 2 ceros distintos.",
        ("AUX",),
    ),
    Step(
        "ROLLE",
        "Rolle iterado",
        "Entre dos ceros consecutivos de g hay un cero de g′ (Rolle): g′ tiene ≥ n + 1 ceros, g″ ≥ n, …, "
        "g⁽ⁿ⁺¹⁾ tiene al menos un cero ξ ∈ (a, b).",
        ("ZEROS",),
    ),
    Step(
        "DERIV",
        "Derivada n + 1",
        "pₙ⁽ⁿ⁺¹⁾ ≡ 0 (grado ≤ n) y w⁽ⁿ⁺¹⁾ ≡ (n + 1)!, luego g⁽ⁿ⁺¹⁾(t) = f⁽ⁿ⁺¹⁾(t) − [f(x) − pₙ(x)]·(n + 1)!/w(x).",
        ("AUX", "DEFW"),
    ),
    Step(
        "FORMULA",
        "Fórmula del error",
        "Evaluando en ξ: 0 = f⁽ⁿ⁺¹⁾(ξ) − [f(x) − pₙ(x)]·(n + 1)!/w(x), es decir f(x) − pₙ(x) = f⁽ⁿ⁺¹⁾(ξ)·w(x)/(n + 1)!.",
        ("ROLLE", "DERIV"),
    ),
    Step(
        "BOUND",
        "Cota uniforme",
        "Con M = máx|f⁽ⁿ⁺¹⁾|: máx_{[a,b]} |f − pₙ| ≤ M · máx|w| / (n + 1)!. El único factor que depende de los nodos es máx|w|.",
        ("FORMULA",),
    ),
    Step(
        "CHEB",
        "Minimalidad de Chebyshev",
        "En [−1, 1], entre todos los polinomios mónicos de grado n + 1, Tₙ₊₁/2ⁿ tiene la menor norma uniforme, 2⁻ⁿ; "
        "sus ceros son los nodos de Chebyshev y con ellos máx|w| = 2⁻ⁿ.",
        ("DEFW",),
    ),
    Step(
        "QED",
        "Conclusión",
        "Con nodos de Chebyshev, máx|f − pₙ| ≤ M / (2ⁿ (n + 1)!), y ninguna otra elección de nodos mejora el factor máx|w|. ∎",
        ("BOUND", "CHEB"),
        is_qed=True,
    ),
    # --- Distractores ------------------------------------------------------
    Step("D_MORE", "Distractor", "Al añadir nodos el error de interpolación siempre disminuye.  (falso: Runge con nodos equiespaciados)", distractor=True),
    Step("D_CONT", "Distractor", "Como f es continua, pₙ → f uniformemente cuando n → ∞.  (falso: teorema de Faber)", distractor=True),
    Step("D_WEIER", "Non sequitur", "Por Weierstrass existe un polinomio a distancia < ε de f; luego el interpolante pₙ está a distancia < ε.", distractor=True),
    Step("D_NODES", "Distractor", "El error es 0 en los nodos, así que entre nodos es pequeño.", distractor=True),
    Step("D_EQUI", "Distractor", "Los nodos equiespaciados minimizan máx|w| porque reparten el intervalo uniformemente.", distractor=True),
    Step("D_TAYLOR", "Distractor", "Por Taylor, f(x) − pₙ(x) = f⁽ⁿ⁺¹⁾(ξ)(x − x₀)ⁿ⁺¹/(n + 1)!.  (confunde el interpolante con el polinomio de Taylor)", distractor=True),
    Step("D_BEST", "Distractor", "Como pₙ es único, es el polinomio de grado ≤ n más cercano a f.  (falso: no es la mejor aproximación uniforme)", distractor=True),
    Step("D_NEWTON", "Distractor", "Iteramos xₖ₊₁ = xₖ − f(xₖ)/f′(xₖ).", distractor=True),
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
