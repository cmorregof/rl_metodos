"""Síntesis final: informe en Markdown y volcado JSON de la historia."""

from __future__ import annotations

import json
import math
from datetime import datetime
from pathlib import Path

from .env_fixpoint import ACTION_NAMES, ACTION_SHORT, RATIO_LABELS, theoretical_action
from .proof_kb import MIN_PROOF_LENGTH, REQUIRED_KEYS, STEPS, step_by_key
from .trainer import ALL_STATES, Trainer


def fmt_time(t: float) -> str:
    m, s = divmod(t, 60)
    return f"{int(m)} min {s:04.1f} s" if m else f"{s:.1f} s"


def final_proof_markdown(tr: Trainer) -> str:
    lines, n = [], 0
    for a, ok in tr.greedy_proof:
        st = STEPS[a]
        if ok:
            n += 1
            lines.append(f"{n}. **{st.title}** ({st.key}). {st.text}")
        else:
            lines.append(f"   - ✗ *salto lógico:* {st.text}")
    return "\n".join(lines)


def _depth() -> dict[str, int]:
    memo: dict[str, int] = {}

    def d(k: str) -> int:
        if k not in memo:
            st = step_by_key(k)
            memo[k] = 0 if not st.deps else 1 + max(d(x) for x in st.deps)
        return memo[k]

    return {k: d(k) for k in REQUIRED_KEYS}


def _mean(xs: list[float]) -> float:
    xs = [x for x in xs if not math.isnan(x)]
    return sum(xs) / len(xs) if xs else float("nan")


def synthesis(tr: Trainer) -> str:
    def gen_of(prefix: str) -> int | None:
        for m in tr.milestones:
            if m.text.startswith(prefix):
                return m.episode
        return None

    first_100 = sum(1 for o in tr.success_curve[:1]) and int(round(tr.success_curve[min(99, len(tr.success_curve) - 1)] * 100))
    g_flip = gen_of("aprendió que si diverge")
    g_halve = gen_of("aprendió que si oscila")
    g_law = gen_of("su ley de control")
    g_95 = gen_of("≥95 %")
    g_qed = gen_of("¡primera demostración completa!")
    g_clean = gen_of("primera demostración sin ningún salto")
    g_min = gen_of("primera demostración mínima")
    g_greedy = gen_of("la política voraz")
    ok, n = tr.law_matches_theory()
    law = tr.control_law()
    law_words = "; ".join(
        f"{tr.law_text(s)} → {ACTION_NAMES[a]}" for s, a in law.items() if a is not None and s[0] in (0, 2, 4) and (s[1] == 0 or s[0] != 4)
    )
    order = sorted(tr.first_greedy.items(), key=lambda kv: (kv[1], kv[0]))
    order_txt = ", ".join(f"{k} (gen {g})" for k, g in order)
    slowest = order[-1] if order else ("—", 0)
    depth = _depth()
    invalid_total = sum(tr.invalid_usage.values())
    distractors = sum(v for k, v in tr.invalid_usage.items() if step_by_key(k).distractor)
    local = tr.invalid_usage.get("D_LOCAL", 0)
    bolz = tr.invalid_usage.get("D_BOLZ", 0)

    p = [
        f"**Fase 1 (punto fijo).** El agente empieza con un α arbitrario y solo observa la razón de residuos "
        f"ρ = |f(xₙ₊₁)|/|f(xₙ)| y si la iteración oscila. En sus primeras 100 generaciones convergió el {first_100} % "
        f"de las veces (un α fijo converge ~35 %). "
        + (f"En la generación {g_flip} aprendió que una divergencia monótona (ρ > 1 sin oscilar) significa que α tiene el "
           f"signo equivocado y hay que invertirlo. " if g_flip else "")
        + (f"En la {g_halve} aprendió que oscilar es señal de que α es demasiado grande (|1 − αf'| > 1 con signo negativo) "
           f"y hay que reducirlo. " if g_halve else "")
        + (f"En la {g_law} su ley de control coincidía con la teoría de Banach en los 12 estados observables y se mantuvo "
           f"estable: {law_words}. Es decir, el agente aprendió a mantener |g'| < 1 y a empujar ρ hacia 0, que es "
           f"acercar α a 1/f'(r): el paso de Newton. " if g_law else f"Al final su ley coincide con la teoría en {ok}/{n} estados. ")
        + (f"Desde la generación {g_95} converge en ≥ 95 % de los casos. " if g_95 else "")
        + f"Total: {tr.converged_total} convergencias, {tr.diverged_total} divergencias y {tr.timeout_total} episodios "
        f"agotados en {tr.ep1} generaciones ({fmt_time(tr.phase1_time)}); media final de "
        f"{_mean(tr.iter_curve[-100:]):.1f} iteraciones por convergencia.",
        f"**Fase 2 (demostración).** Los pasos se asentaron en la política en el orden en que sus dependencias quedaban "
        f"disponibles: {order_txt}. El último fue {slowest[0]} (gen {slowest[1]}); la conclusión tiene profundidad "
        f"{depth['QED']} en el grafo, así que su valor solo se propaga hacia atrás una vez recorrida la cadena completa. "
        + (f"Primera demostración completa en la generación {g_qed}, " if g_qed else "No logró una demostración completa, ")
        + (f"primera sin saltos lógicos en la {g_clean}, " if g_clean else "")
        + (f"primera mínima ({MIN_PROOF_LENGTH} pasos) en la {g_min}. " if g_min else "")
        + (f"La política voraz produce la demostración mínima desde la generación {g_greedy}. " if g_greedy else "")
        + f"Cometió {invalid_total} saltos lógicos, {distractors} con distractores: el error clásico «|g'(p)| < 1 basta para "
        f"cualquier x₀» fue intentado {local} veces y el non sequitur «existe punto fijo, luego la iteración converge» "
        f"{bolz} veces; ambos quedaron descartados. Total: {tr.qed_total} demostraciones completas y {tr.minimal_total} "
        f"mínimas en {tr.ep2} generaciones ({fmt_time(tr.phase2_time)}).",
        f"**Lectura.** Las dos fases cuentan la misma historia desde lados distintos. En la práctica, el agente descubre "
        f"que lo único que decide la convergencia es la constante de contracción empírica ρ ≈ |g'(r)| y aprende a "
        f"fabricar un mapa contractivo ajustando α. En la teoría, aprende que la hipótesis de contracción (k < 1) es la "
        f"que produce existencia y unicidad del punto fijo, la convergencia geométrica y las cotas de error, y que ni "
        f"Bolzano, ni la acotación, ni |g'(p)| < 1 a solas la sustituyen. Tiempo total: {fmt_time(tr.elapsed)}.",
    ]
    return "\n\n".join(p)


def build_report(tr: Trainer) -> str:
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    law = tr.control_law()
    law_rows = "\n".join(
        f"| ρ {RATIO_LABELS[b]} | {'oscila' if o else 'monótona'} | {ACTION_SHORT[a] if a is not None else '?'} | "
        f"{'/'.join(ACTION_SHORT[x] for x in theoretical_action(b, o))} | "
        f"{'✔' if a is not None and a in theoretical_action(b, o) else '✘'} |"
        for (b, o), a in law.items()
    )
    depth = _depth()
    order_rows = "\n".join(
        f"| {k} | {step_by_key(k).title} | {depth[k]} | {tr.first_valid.get(k, '—')} | {tr.first_greedy.get(k, '—')} | {tr.step_usage.get(k, 0)} |"
        for k in sorted(REQUIRED_KEYS, key=lambda k: (tr.first_greedy.get(k, 10**9), depth[k]))
    )
    inval_rows = "\n".join(f"| {k} | {step_by_key(k).title} | {v} |" for k, v in tr.invalid_usage.most_common(8))
    hitos = "\n".join(f"| {m.phase} | {m.episode} | {m.t:.1f} s | {m.text} |" for m in tr.milestones)
    ok, n = tr.law_matches_theory()
    return f"""# FixpointRL · informe de aprendizaje

Generado: {now} · semilla {tr.cfg.seed} · tolerancia {tr.cfg.tol:g} · tiempo total **{fmt_time(tr.elapsed)}**
(fase 1: {fmt_time(tr.phase1_time)}, fase 2: {fmt_time(tr.phase2_time)}).

## Síntesis: cómo aprendió

{synthesis(tr)}

## Demostración final (política voraz, sin exploración)

**Teorema (punto fijo de Banach en [a, b]).** Sea g continua en [a, b] con g([a, b]) ⊆ [a, b] y
|g(x) − g(y)| ≤ k·|x − y| con 0 ≤ k < 1. Entonces g tiene un único punto fijo p, la iteración xₙ₊₁ = g(xₙ)
converge a p para todo x₀ ∈ [a, b], |xₙ − p| ≤ kⁿ·max(x₀ − a, b − x₀) y |xₙ − p| ≤ kⁿ/(1 − k)·|x₁ − x₀|.

{final_proof_markdown(tr)}

## Fase 1 · ley de control aprendida ({ok}/{n} estados coinciden con la teoría)

Acciones: `=` mantener α, `×2` doblar, `÷2` reducir a la mitad, `±` invertir el signo.

| ρ = |f(xₙ₊₁)|/|f(xₙ)| | forma | acción aprendida | teoría | |
|---|---|---|---|---|
{law_rows}

Convergencias: {tr.converged_total} · divergencias: {tr.diverged_total} · agotadas: {tr.timeout_total}
· éxito final (100 últimas): {100 * tr.success_curve[-1]:.0f} % · iteraciones medias: {_mean(tr.iter_curve[-100:]):.1f}

## Fase 2 · orden en que se asentó cada paso

| paso | título | profundidad | 1ª vez válido (gen) | entra en política voraz (gen) | usos |
|---|---|---|---|---|---|
{order_rows}

Saltos lógicos más frecuentes:

| paso | tipo | intentos inválidos |
|---|---|---|
{inval_rows}

Demostraciones completas: {tr.qed_total} · mínimas: {tr.minimal_total} · mejor retorno: {tr.best_reward:+.2f} (gen {tr.best_episode}).

## Bitácora de hitos

| fase | generación | tiempo | hito |
|---|---|---|---|
{hitos}
"""


def save_run(tr: Trainer, out_dir: Path) -> Path:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = out_dir / stamp
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "informe.md").write_text(build_report(tr), encoding="utf-8")
    (run_dir / "demostracion.md").write_text(final_proof_markdown(tr), encoding="utf-8")
    history = {
        "config": tr.cfg.__dict__,
        "elapsed": tr.elapsed,
        "phase1_time": tr.phase1_time,
        "phase2_time": tr.phase2_time,
        "milestones": [m.__dict__ for m in tr.milestones],
        "phase1": {
            "success_curve": tr.success_curve,
            "iter_curve": tr.iter_curve,
            "control_law": {str(s): a for s, a in tr.control_law().items()},
            "q": {str(s): tr.agent1.q.get(s) for s in ALL_STATES},
            "converged": tr.converged_total,
            "diverged": tr.diverged_total,
            "timeout": tr.timeout_total,
        },
        "phase2": {
            "reward_curve": tr.reward_curve,
            "first_valid": tr.first_valid,
            "first_greedy": tr.first_greedy,
            "step_usage": dict(tr.step_usage),
            "invalid_usage": dict(tr.invalid_usage),
            "qed_total": tr.qed_total,
            "minimal_total": tr.minimal_total,
            "final_proof": [[STEPS[a].key, ok] for a, ok in tr.greedy_proof],
        },
    }
    (run_dir / "historia.json").write_text(json.dumps(history, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    return run_dir
