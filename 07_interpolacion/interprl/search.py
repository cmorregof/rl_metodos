"""Fase 3 · frontera: buscar nodos con constante de Lebesgue mínima.

Aquí no hay grafo de lemas. Hay un **verificador independiente**
(`interp.lebesgue_constant`) que, dado cualquier conjunto de nodos, calcula
cuánto amplifica la interpolación el error en los datos, y una **búsqueda**
que propone nodos. La "demostración" de que unos nodos son mejores que los de
Chebyshev es el número que devuelve el verificador, reproducible por cualquiera.

Es el mismo patrón que el proyecto 06 (PEP como juez) en pequeño:

* propuesta: n + 1 nodos simétricos en [−1, 1] con los extremos fijos en ±1
  (se sabe que los nodos óptimos de Lebesgue son simétricos e incluyen ±1);
* referencia: Chebyshev–Lobatto (extremos incluidos) y Chebyshev extendido;
* búsqueda: entropía cruzada sobre los nodos interiores + refinamiento por
  coordenadas (Nelder–Mead sería otra opción; aquí todo está escrito a mano
  para que un estudiante lo lea entero);
* resultado: nodos, Λ, y el cociente Λ/Λ_ref; si es < 1 la propuesta bate la
  referencia y el certificado es reproducir Λ con `verify`.

Lo que se sabe (Brutman 1978, 1997): Λ_óptimo(n) ≈ (2/π) ln(n + 1) + 0.52; el
Chebyshev extendido está a menos de 0.02 del óptimo; los óptimos exactos solo
se conocen numéricamente. Batir al extendido en la 3.ª cifra ya es un
resultado no trivial que un estudiante puede reproducir y defender.
"""

from __future__ import annotations

import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path

from .interp import chebyshev_lobatto, extended_chebyshev, lebesgue_constant


@dataclass
class Candidate:
    nodes: list[float]
    lebesgue: float
    argmax: float


def symmetric_nodes(interior: list[float]) -> list[float]:
    """Nodos completos a partir de los interiores positivos (0 < t₁ < … < 1)."""
    pos = sorted(t for t in interior if 0 <= t < 1)
    return sorted({-1.0, 1.0, *pos, *(-t for t in pos if t > 0)})


def evaluate(interior: list[float], n: int) -> Candidate | None:
    x = symmetric_nodes(interior)
    if len(x) != n + 1:
        return None
    lam, t = lebesgue_constant(x)
    return Candidate(x, lam, t)


def _n_interior(n: int) -> tuple[int, bool]:
    """Cuántos interiores positivos hacen falta y si el 0 es nodo (n par ⇒ n+1 impar ⇒ sí)."""
    has_zero = (n + 1) % 2 == 1
    return (n + 1 - 2 - (1 if has_zero else 0)) // 2, has_zero


def references(n: int) -> dict[str, Candidate]:
    out = {}
    for name, x in (("chebyshev_lobatto", chebyshev_lobatto(n)), ("chebyshev_extendido", extended_chebyshev(n))):
        lam, t = lebesgue_constant(x)
        out[name] = Candidate(x, lam, t)
    return out


def cross_entropy(n: int, seed: int = 0, iters: int = 40, pop: int = 60, elite: int = 8) -> Candidate:
    """Entropía cruzada sobre los interiores positivos, arrancando del Chebyshev extendido."""
    rng = random.Random(seed)
    k, has_zero = _n_interior(n)
    start = [t for t in extended_chebyshev(n) if t > 0 and t < 1]
    mu = list(start[:k]) if len(start) >= k else [(i + 1) / (k + 1) for i in range(k)]
    sigma = [0.05] * k
    best: Candidate | None = None

    def build(inter: list[float]) -> list[float]:
        return inter + ([0.0] if has_zero else [])

    for it in range(iters):
        samples = []
        for _ in range(pop):
            cand = sorted(min(0.999, max(1e-3, rng.gauss(m, s))) for m, s in zip(mu, sigma))
            if len(set(cand)) < k:
                continue
            c = evaluate(build(cand), n)
            if c is not None:
                samples.append((c.lebesgue, cand, c))
        samples.sort(key=lambda s: s[0])
        top = samples[:elite]
        if not top:
            break
        if best is None or top[0][0] < best.lebesgue:
            best = top[0][2]
        mu = [sum(s[1][i] for s in top) / len(top) for i in range(k)]
        sigma = [max(1e-6, math.sqrt(sum((s[1][i] - mu[i]) ** 2 for s in top) / len(top))) for i in range(k)]
    # refinamiento por coordenadas
    inter = [t for t in best.nodes if 0 < t < 1]
    step = 1e-3
    while step > 1e-9:
        improved = False
        for i in range(len(inter)):
            for d in (step, -step):
                trial = list(inter)
                trial[i] = min(0.999, max(1e-3, trial[i] + d))
                c = evaluate(build(sorted(trial)), n)
                if c is not None and c.lebesgue < best.lebesgue - 1e-13:
                    best, inter, improved = c, sorted(trial), True
        if not improved:
            step /= 2
    return best


def run_search(n: int, seed: int = 0, iters: int = 40, out: Path | None = None) -> dict:
    refs = references(n)
    best = cross_entropy(n, seed=seed, iters=iters)
    ref = min(refs.values(), key=lambda c: c.lebesgue)
    result = {
        "n": n,
        "mejor": asdict(best),
        "referencias": {k: asdict(v) for k, v in refs.items()},
        "ratio_vs_mejor_referencia": best.lebesgue / ref.lebesgue,
        "bate_referencia": best.lebesgue < ref.lebesgue - 1e-12,
    }
    if out is not None:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, indent=1, ensure_ascii=False), encoding="utf-8")
    return result


def verify_file(path: Path) -> tuple[bool, float, float]:
    """Recalcula Λ de los nodos guardados con otra malla y compara con lo afirmado."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    nodes = data["mejor"]["nodes"]
    claimed = data["mejor"]["lebesgue"]
    lam, _ = lebesgue_constant(nodes, samples=97)
    return abs(lam - claimed) < 1e-9, lam, claimed
