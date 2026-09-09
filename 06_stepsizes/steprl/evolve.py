"""Capa simbólica: un LLM propone *programas* que generan programas de pasos,
el PEP los certifica, y los mejores sobreviven (búsqueda evolutiva guiada por
LLM, al estilo FunSearch/AlphaEvolve, con el SDP como único juez).

Un candidato es código Python que define `schedule(n) -> list[float]`. Se
puntúa en dos regímenes:

* horizonte fijo: τ(n) = peor caso certificado de f(xₙ) − f* para varios n,
  comparado con la mejor referencia conocida (silver / óptimos numéricos);
* anytime: para el programa largo s = schedule(N), el peor caso de cada
  prefijo τ₁…τ_N y el mayor p tal que τₜ ≤ ½·t⁻ᵖ para todo t ≤ N (½ es la
  cota trivial f(x₀) − f* ≤ LR²/2). El mejor exponente anytime publicado es
  1.119 (Zhang et al. 2024); la cota inferior, 1.334 (2026).
"""

from __future__ import annotations

import json
import math
import re
import time
import traceback
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from .pep import fit_exponent, gd_worst_case, prefix_worst_cases, silver_schedule

# Óptimos numéricos a horizonte fijo (Das Gupta et al.; tabla de Grimmer–Shu–Wang, /2 por convención) y
# los que reproduce nuestra búsqueda por entropía cruzada (n = 6 sale ligeramente mejor: por contrastar).
REFERENCE_FIXED: dict[int, float] = {
    1: 0.125000, 2: 0.065945, 3: 0.042895, 4: 0.031170, 5: 0.024070,
    6: 0.020049, 7: 0.016330, 8: 0.014055, 9: 0.012280, 10: 0.010620,
}
ANYTIME_BEST_KNOWN = 1.119
ANYTIME_LOWER_BOUND = 1.334

SYSTEM_PROMPT = """Eres un investigador en optimización convexa. Buscamos programas de pasos (stepsize schedules)
para el descenso de gradiente x_{k+1} = x_k − (h_k/L)·∇f(x_k) sobre funciones convexas L-suaves con ‖x_0 − x*‖ ≤ R.
El peor caso exacto de f(x_n) − f* (con L = R = 1) lo certifica un programa semidefinido (PEP); esa certificación
es la única recompensa. Tu tarea: proponer código Python que defina `schedule(n: int) -> list[float]` (n pasos
positivos). Piensa en estructura, no en números sueltos: composiciones recursivas (silver: h = [h', 1+ρ^{k-2}, h']),
pasos largos finales, patrones fractales, valuaciones 2-ádicas, etc. Devuelve UN solo bloque ```python``` con el
programa completo y autocontenido (solo math y numpy), sin explicación fuera del bloque salvo un comentario breve."""


def extract_code(text: str) -> str:
    m = re.search(r"```(?:python)?\s*(.*?)```", text, re.S)
    return (m.group(1) if m else text).strip()


def load_schedule_fn(code: str):
    ns: dict = {"__builtins__": __builtins__, "math": math, "np": np, "numpy": np}
    exec(compile(code, "<candidate>", "exec"), ns, ns)  # un solo espacio de nombres: las constantes de módulo se ven desde schedule()
    fn = ns.get("schedule")
    if fn is None:
        raise ValueError("el programa no define schedule(n)")
    return fn


def _valid(h, n: int) -> np.ndarray:
    h = np.asarray(list(h), dtype=float)
    if h.shape != (n,) or not np.all(np.isfinite(h)) or np.any(h <= 0) or np.any(h > 50):
        raise ValueError(f"schedule({n}) inválido: {h}")
    return h


ANYTIME_C = 0.5  # f(x₀) − f* ≤ L R²/2 siempre: anclamos la garantía anytime en esa constante


def anytime_exponent(taus: list[float], C: float = ANYTIME_C) -> tuple[float, float]:
    """Mayor p tal que τₜ ≤ C·t⁻ᵖ para TODO t = 2…N (garantía anytime certificada con constante C);
    devuelve (p, pendiente LS en log-log). La pendiente LS es informativa pero esconde los picos
    (el silver truncado tiene τ₄ = τ₈ = 0.0858), por eso la puntuación es p, no la pendiente."""
    ts = np.arange(1, len(taus) + 1)
    taus = np.asarray(taus, dtype=float)
    mask = ts >= 2
    p = float(np.min(np.log(C / taus[mask]) / np.log(ts[mask])))
    slope = fit_exponent(ts[mask], taus[mask]) if mask.sum() >= 2 else float("nan")
    return p, float(slope)


@dataclass
class Candidate:
    code: str
    generation: int
    fixed: dict[int, float] = field(default_factory=dict)  # n → τ(n)
    fixed_ratio: float = float("nan")  # media geométrica de τ(n)/referencia (1 = igual que lo conocido)
    anytime_p: float = float("nan")
    anytime_slope: float = float("nan")
    anytime_taus: list[float] = field(default_factory=list)
    error: str = ""
    seconds: float = 0.0

    @property
    def ok(self) -> bool:
        return not self.error

    def summary(self) -> str:
        if not self.ok:
            return f"ERROR: {self.error}"
        fx = ", ".join(f"n={n}: {v:.5f} ({v / REFERENCE_FIXED[n]:.3f}×ref)" for n, v in self.fixed.items() if n in REFERENCE_FIXED)
        return (f"horizonte fijo [{fx}] ratio medio {self.fixed_ratio:.4f}; anytime: τ_t ≤ ½·t^(−{self.anytime_p:.3f}) para todo t ≤ {len(self.anytime_taus)}"
                f" (pendiente LS {self.anytime_slope:.3f})")


def evaluate(code: str, generation: int, fixed_ns=(1, 2, 3, 4, 5, 6, 7, 8, 10), anytime_N: int = 31) -> Candidate:
    c = Candidate(code=code, generation=generation)
    t0 = time.perf_counter()
    try:
        fn = load_schedule_fn(code)
        ratios = []
        for n in fixed_ns:
            h = _valid(fn(n), n)
            tau = gd_worst_case(h).value
            c.fixed[n] = tau
            if n in REFERENCE_FIXED:
                ratios.append(tau / REFERENCE_FIXED[n])
        c.fixed_ratio = float(np.exp(np.mean(np.log(ratios)))) if ratios else float("nan")
        if anytime_N:
            h = _valid(fn(anytime_N), anytime_N)
            c.anytime_taus = prefix_worst_cases(h)
            c.anytime_p, c.anytime_slope = anytime_exponent(c.anytime_taus)
    except Exception as e:  # el candidato falló: se conserva el motivo para el LLM
        c.error = f"{type(e).__name__}: {e}"[:400]
    c.seconds = time.perf_counter() - t0
    return c


def baselines(anytime_N: int = 31) -> dict[str, Candidate]:
    """Referencias: silver (longitud 2ᵏ − 1 truncada / extendida por la fórmula 2-ádica) y paso constante 1."""
    silver_code = (
        "import math\nRHO = 1 + math.sqrt(2)\n"
        "def schedule(n):\n    def v2(t):\n        k = 0\n        while t % 2 == 0:\n            t //= 2; k += 1\n        return k\n"
        "    return [1 + RHO ** (v2(t) - 1) for t in range(1, n + 1)]\n"
    )
    const_code = "def schedule(n):\n    return [1.0] * n\n"
    return {"silver": evaluate(silver_code, 0, anytime_N=anytime_N), "constante": evaluate(const_code, 0, anytime_N=anytime_N)}


def build_prompt(pool: list[Candidate], objective: str, anytime_N: int) -> str:
    lines = [
        f"Objetivo actual: {'ANYTIME (maximizar p tal que τ_t ≤ ½·t^(−p) para todo prefijo t; el mejor exponente anytime publicado es 1.119 y la cota inferior 1.334; ojo: un solo prefijo malo hunde p)' if objective == 'anytime' else 'HORIZONTE FIJO (minimizar τ(n)/referencia; referencia = mejores óptimos numéricos conocidos)'}.",
        f"Referencias a horizonte fijo τ_ref(n): {json.dumps({k: round(v, 6) for k, v in REFERENCE_FIXED.items()})}",
        f"En el régimen anytime evaluamos schedule({anytime_N}) y el peor caso de cada prefijo t = 1…{anytime_N}.",
        "",
        "Programas evaluados hasta ahora (de peor a mejor), con su puntuación certificada:",
    ]
    for c in pool:
        lines.append("```python\n" + c.code + "\n```")
        lines.append("→ " + c.summary())
        if c.ok and c.anytime_taus:
            lines.append("   τ_t anytime: " + ", ".join(f"{v:.4f}" for v in c.anytime_taus))
        lines.append("")
    lines.append("Propón un programa nuevo, distinto de los anteriores, que mejore la puntuación del objetivo actual. "
                 "Explica en un comentario de una línea qué idea estructural pruebas.")
    return "\n".join(lines)


def score_key(c: Candidate, objective: str) -> float:
    if not c.ok:
        return float("inf")
    if objective == "anytime":
        return -c.anytime_p if c.anytime_p == c.anytime_p else float("inf")
    return c.fixed_ratio if c.fixed_ratio == c.fixed_ratio else float("inf")


@dataclass
class EvolveLog:
    objective: str
    provider: str
    model: str
    candidates: list[dict] = field(default_factory=list)

    def save(self, path: Path) -> None:
        path.write_text(json.dumps(asdict(self), ensure_ascii=False, indent=1, default=float), encoding="utf-8")


def run_evolution(
    provider,
    generations: int = 10,
    objective: str = "anytime",
    pool_size: int = 4,
    anytime_N: int = 31,
    out_dir: Path | None = None,
    verbose: bool = True,
    provider_name: str = "",
    model_name: str = "",
) -> list[Candidate]:
    base = baselines(anytime_N=anytime_N)
    pool: list[Candidate] = [base["silver"], base["constante"]]  # las dos referencias, a igual N, siempre visibles
    log = EvolveLog(objective=objective, provider=provider_name, model=model_name)
    log.candidates.append({**asdict(base["silver"]), "name": "silver"})
    log.candidates.append({**asdict(base["constante"]), "name": "constante"})
    if verbose:
        print(f"[base] silver     → {base['silver'].summary()}")
        print(f"[base] constante  → {base['constante'].summary()}")
    for g in range(1, generations + 1):
        pool.sort(key=lambda c: score_key(c, objective), reverse=True)  # de peor a mejor para el prompt
        prompt = build_prompt(pool[-pool_size:], objective, anytime_N)
        try:
            text = provider.complete(SYSTEM_PROMPT, prompt)
        except Exception as e:
            if verbose:
                print(f"[gen {g}] proveedor falló: {e}")
            log.candidates.append({"generation": g, "error": f"proveedor: {e}"})
            continue
        code = extract_code(text)
        c = evaluate(code, g, anytime_N=anytime_N)
        log.candidates.append(asdict(c))
        pool.append(c)
        pool.sort(key=lambda c: score_key(c, objective))
        pool = pool[: max(pool_size, 1) * 2]
        if verbose:
            best = pool[0]
            print(f"[gen {g}] {c.summary()}  ({c.seconds:.1f}s)\n        mejor hasta ahora: {best.summary()}")
        if out_dir:
            out_dir.mkdir(parents=True, exist_ok=True)
            log.save(out_dir / "evolucion.json")
            (out_dir / "mejor.py").write_text(pool[0].code, encoding="utf-8")
    pool.sort(key=lambda c: score_key(c, objective))
    return pool
