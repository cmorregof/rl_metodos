"""Capa simbólica: un LLM propone *programas* que generan programas de pasos,
el PEP los certifica, y los mejores sobreviven (búsqueda evolutiva guiada por
LLM, al estilo FunSearch/AlphaEvolve, con el SDP como único juez).

Un candidato es código Python que define `schedule(n) -> list[float]`. Se
puntúa en dos regímenes:

* horizonte fijo: τ(n) = peor caso certificado de f(xₙ) − f* para varios n,
  comparado con la mejor referencia conocida (silver / óptimos numéricos);
* anytime: para el programa largo s = schedule(N), el peor caso de cada
  prefijo τ₁…τ_N y la **constante de la garantía a exponente objetivo**
  C(p*) = máxₜ τₜ·t^{p*} (τₜ ≤ C·t⁻ᵖ* certificado para todo t; menor es
  mejor; máximo sobre horizontes). Es la puntuación: ni los picos ni los
  arranques lentos la mejoran. Se acompaña del exponente por duplicación y
  del de la última duplicación, solo informativos. El mejor exponente anytime
  publicado es 1.119 (Zhang et al. 2024); la cota inferior, 1.334 (2026); el
  paso constante tiene orden 1, así que su C crece con N.
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
TARGET_P_DEFAULT = 1.12  # exponente objetivo de la garantía anytime: justo por encima del mejor publicado


def guarantee_constant(taus, p: float) -> tuple[float, int]:
    """Menor C tal que τₜ ≤ C·t⁻ᵖ para todo t ≤ N, es decir C = máxₜ τₜ·tᵖ, y el t que manda.
    Es la puntuación anytime: no se puede forzar ni con picos (suben C) ni con arranques lentos
    (τ₁ grande sube C), y comparar C entre programas a igual p es comparar garantías certificadas."""
    best_C, best_t = -1.0, 0
    for t, tau in enumerate(taus, 1):
        v = tau * t**p
        if v > best_C:
            best_C, best_t = v, t
    return float(best_C), best_t


def last_window_exponent(taus) -> float:
    """Orden observado en la última duplicación: log₂(τ_{N/2}/τ_N). Solo informativo."""
    N = len(taus)
    return float(math.log(taus[N // 2 - 1] / taus[N - 1]) / math.log(2)) if N >= 8 else float("nan")

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
    devuelve (p, pendiente LS en log-log). Es una garantía a N finito: con C = ½ está inflada para N
    pequeño (el paso constante da 1.21 a N = 31 y tiende a 1), así que NO es comparable con los
    exponentes asintóticos publicados; para eso está `doubling_exponent`."""
    ts = np.arange(1, len(taus) + 1)
    taus = np.asarray(taus, dtype=float)
    mask = ts >= 2
    p = float(np.min(np.log(C / taus[mask]) / np.log(ts[mask])))
    slope = fit_exponent(ts[mask], taus[mask]) if mask.sum() >= 2 else float("nan")
    return p, float(slope)


def doubling_exponent(taus: list[float], t_min: int = 8) -> tuple[float, int]:
    """Exponente asintótico robusto: mín sobre t ≥ t_min de log(τ_{⌊t/2⌋}/τ_t)/log 2, el orden observado
    en la última duplicación del horizonte. Para τ_t ~ C·t⁻ᵖ tiende a p sea cual sea C; un pico
    (τ_t ≈ τ_{t/2}, como el silver truncado) lo hunde a 0. Devuelve (p, t que manda)."""
    taus = np.asarray(taus, dtype=float)
    N = len(taus)
    if N < t_min:
        return float("nan"), 0
    best_p, best_t = float("inf"), 0
    for t in range(t_min, N + 1):
        p = math.log(taus[t // 2 - 1] / taus[t - 1]) / math.log(2)
        if p < best_p:
            best_p, best_t = p, t
    return float(best_p), best_t


@dataclass
class Candidate:
    code: str
    generation: int
    fixed: dict[int, float] = field(default_factory=dict)  # n → τ(n)
    fixed_ratio: float = float("nan")  # media geométrica de τ(n)/referencia (1 = igual que lo conocido)
    anytime_p: float = float("nan")
    anytime_slope: float = float("nan")
    doubling_p: float = float("nan")  # mínimo sobre horizontes del exponente por duplicación (informativo)
    doubling_t: int = 0
    target_p: float = TARGET_P_DEFAULT
    C_target: float = float("nan")  # PUNTUACIÓN anytime: máx sobre horizontes de máxₜ τₜ·t^{target_p} (menor es mejor)
    C_t: int = 0
    anytime_taus: list[float] = field(default_factory=list)  # prefijos del horizonte mayor
    per_horizon: dict[int, dict] = field(default_factory=dict)  # N → {p, t, anchored, slope, tau_N}
    error: str = ""
    seconds: float = 0.0

    @property
    def ok(self) -> bool:
        return not self.error

    def summary(self) -> str:
        if not self.ok:
            return f"ERROR: {self.error}"
        fx = ", ".join(f"n={n}: {v:.5f} ({v / REFERENCE_FIXED[n]:.3f}×ref)" for n, v in self.fixed.items() if n in REFERENCE_FIXED)
        if not self.per_horizon:
            return f"horizonte fijo [{fx}] ratio medio {self.fixed_ratio:.4f}"
        hz = "; ".join(
            f"N={N}: C({self.target_p:g})={r['C']:.3f} (manda t={r['C_t']}), τ_N={r['tau_N']:.5f}, dup p={r['p']:.3f}, última duplicación p={r['last_p']:.3f}"
            for N, r in sorted(self.per_horizon.items())
        )
        return (f"horizonte fijo [{fx}] ratio medio {self.fixed_ratio:.4f}; anytime: PUNTUACIÓN C({self.target_p:g}) = {self.C_target:.3f} "
                f"(τ_t ≤ C·t^(−{self.target_p:g}) certificado para todo t en todos los horizontes; menor es mejor) [{hz}]")


def evaluate(code: str, generation: int, fixed_ns=(1, 2, 3, 4, 5, 6, 7, 8, 10), anytime_N=(31, 63), target_p: float = TARGET_P_DEFAULT) -> Candidate:
    """`anytime_N`: horizonte o tupla de horizontes; la puntuación anytime es el peor exponente por
    duplicación entre ellos (un programa que sobreajusta a un horizonte, p. ej. con `_HORIZON = 31`
    escrito a mano, se hunde en el otro)."""
    c = Candidate(code=code, generation=generation, target_p=target_p)
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
        horizons = [N for N in (anytime_N if isinstance(anytime_N, (tuple, list)) else (anytime_N,)) if N]
        for N in sorted(horizons):
            h = _valid(fn(N), N)
            taus = prefix_worst_cases(h)
            pa, sl = anytime_exponent(taus)
            pd, td = doubling_exponent(taus)
            Cg, Ct = guarantee_constant(taus, target_p)
            c.per_horizon[N] = {"p": pd, "t": td, "anchored": pa, "slope": sl, "tau_N": taus[-1], "C": Cg, "C_t": Ct, "last_p": last_window_exponent(taus)}
            c.anytime_taus, c.anytime_p, c.anytime_slope = taus, pa, sl
            if not (c.doubling_p == c.doubling_p) or pd < c.doubling_p:
                c.doubling_p, c.doubling_t = pd, td
            if not (c.C_target == c.C_target) or Cg > c.C_target:
                c.C_target, c.C_t = Cg, Ct
    except Exception as e:  # el candidato falló: se conserva el motivo para el LLM
        c.error = f"{type(e).__name__}: {e}"[:400]
    c.seconds = time.perf_counter() - t0
    return c


def baselines(anytime_N=(31, 63), target_p: float = TARGET_P_DEFAULT) -> dict[str, Candidate]:
    """Referencias: silver (longitud 2ᵏ − 1 truncada / extendida por la fórmula 2-ádica) y paso constante 1."""
    silver_code = (
        "import math\nRHO = 1 + math.sqrt(2)\n"
        "def schedule(n):\n    def v2(t):\n        k = 0\n        while t % 2 == 0:\n            t //= 2; k += 1\n        return k\n"
        "    return [1 + RHO ** (v2(t) - 1) for t in range(1, n + 1)]\n"
    )
    const_code = "def schedule(n):\n    return [1.0] * n\n"
    return {"silver": evaluate(silver_code, 0, anytime_N=anytime_N, target_p=target_p), "constante": evaluate(const_code, 0, anytime_N=anytime_N, target_p=target_p)}


def build_prompt(pool: list[Candidate], objective: str, anytime_N, target_p: float = TARGET_P_DEFAULT) -> str:
    horizons = list(anytime_N) if isinstance(anytime_N, (tuple, list)) else [anytime_N]
    lines = [
        f"Objetivo actual: {f'ANYTIME: minimizar C({target_p:g}) = máx_t τ_t·t^{target_p:g}, la menor constante con la que la garantía f(x_t) − f* ≤ C·L·R²·t^(−{target_p:g}) queda certificada para TODO prefijo t en todos los horizontes. Menor es mejor. Un pico sube C; un arranque con pasos pequeños (τ_1 grande) también sube C: no hay atajos. Contexto: el mejor exponente anytime publicado es 1.119, la cota inferior asintótica 1.334 y el paso constante tiene orden 1 (su C crece con N). Un programa con pasos acotados solo mejora la constante, no el orden: para que C se mantenga acotada al crecer N hacen falta pasos que crezcan con t sin que ningún prefijo se dispare. Cuando la población se estabilice subiremos el exponente objetivo.' if objective == 'anytime' else 'HORIZONTE FIJO (minimizar τ(n)/referencia; referencia = mejores óptimos numéricos conocidos)'}.",
        f"Referencias a horizonte fijo τ_ref(n): {json.dumps({k: round(v, 6) for k, v in REFERENCE_FIXED.items()})}",
        f"En el régimen anytime evaluamos schedule(N) para varios N ({', '.join(map(str, horizons))} en esta ronda, pero el horizonte puede cambiar) "
        f"y el peor caso de cada prefijo t = 1…N; la puntuación es el PEOR exponente entre horizontes. No escribas el horizonte a mano: "
        f"un programa que solo funciona hasta N=31 se hunde en N=63.",
        "",
        "Programas evaluados hasta ahora (de peor a mejor), con su puntuación certificada:",
    ]
    for c in pool:
        lines.append("```python\n" + c.code + "\n```")
        lines.append("→ " + c.summary())
        if c.ok and c.anytime_taus:
            lines.append(f"   τ_t anytime (N={len(c.anytime_taus)}): " + ", ".join(f"{v:.4f}" for v in c.anytime_taus))
        lines.append("")
    lines.append("Propón un programa nuevo, distinto de los anteriores, que mejore la puntuación del objetivo actual. "
                 "Explica en un comentario de una línea qué idea estructural pruebas.")
    return "\n".join(lines)


def score_key(c: Candidate, objective: str) -> float:
    if not c.ok:
        return float("inf")
    if objective == "anytime":
        return c.C_target if c.C_target == c.C_target else float("inf")
    return c.fixed_ratio if c.fixed_ratio == c.fixed_ratio else float("inf")


@dataclass
class EvolveLog:
    objective: str
    provider: str
    model: str
    target_p: float = TARGET_P_DEFAULT
    candidates: list[dict] = field(default_factory=list)

    def save(self, path: Path) -> None:
        path.write_text(json.dumps(asdict(self), ensure_ascii=False, indent=1, default=float), encoding="utf-8")


def run_evolution(
    provider,
    generations: int = 10,
    objective: str = "anytime",
    pool_size: int = 4,
    anytime_N=(31, 63),
    target_p: float = TARGET_P_DEFAULT,
    out_dir: Path | None = None,
    provider_retries: int = 8,  # reintentos con espera creciente ante errores transitorios (red, tasa)
    verbose: bool = True,
    provider_name: str = "",
    model_name: str = "",
) -> list[Candidate]:
    base = baselines(anytime_N=anytime_N, target_p=target_p)
    pool: list[Candidate] = [base["silver"], base["constante"]]  # las dos referencias, a igual N, siempre visibles
    log = EvolveLog(objective=objective, provider=provider_name, model=model_name, target_p=target_p)
    log.candidates.append({**asdict(base["silver"]), "name": "silver"})
    log.candidates.append({**asdict(base["constante"]), "name": "constante"})
    if verbose:
        print(f"[base] silver     → {base['silver'].summary()}")
        print(f"[base] constante  → {base['constante'].summary()}")
    for g in range(1, generations + 1):
        pool.sort(key=lambda c: score_key(c, objective), reverse=True)  # de peor a mejor para el prompt
        prompt = build_prompt(pool[-pool_size:], objective, anytime_N, target_p)
        text = None
        for attempt in range(provider_retries + 1):
            try:
                text = provider.complete(SYSTEM_PROMPT, prompt)
                break
            except Exception as e:
                msg = str(e)
                transient = any(k in msg.lower() for k in ("connection", "timeout", "timed out", "límite de tasa", "rate", "overloaded", "502", "503", "529"))
                if transient and attempt < provider_retries:
                    wait = min(600, 30 * 2**attempt)  # 30 s, 60, 120, … hasta 10 min: sin red (tapa cerrada) no se queman generaciones
                    if verbose:
                        print(f"[gen {g}] proveedor: {msg[:80]} → reintento {attempt + 1}/{provider_retries} en {wait} s")
                    time.sleep(wait)
                    continue
                if verbose:
                    print(f"[gen {g}] proveedor falló: {e}")
                log.candidates.append({"generation": g, "error": f"proveedor: {e}"})
                if out_dir:
                    out_dir.mkdir(parents=True, exist_ok=True)
                    log.save(out_dir / "evolucion.json")  # también los fallos del proveedor quedan en disco
                break
        if text is None:
            continue
        code = extract_code(text)
        c = evaluate(code, g, anytime_N=anytime_N, target_p=target_p)
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
