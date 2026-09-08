"""Síntesis final: informe en Markdown y volcado JSON de la historia."""

from __future__ import annotations

import json
import math
from datetime import datetime
from pathlib import Path

from .env_taylor import ACTION_NAMES, ACTION_SHORT, RATIO_LABELS, TERM_LABELS, theoretical_action
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

    first_100 = int(round(tr.success_curve[min(99, len(tr.success_curve) - 1)] * 100)) if tr.success_curve else 0
    g_ratio = gen_of("descubrió el criterio del cociente")
    g_term = gen_of("descubrió el criterio del último término")
    g_law = gen_of("su ley coincide")
    g_95 = gen_of("≥95 %")
    g_qed = gen_of("¡primera demostración completa!")
    g_clean = gen_of("primera demostración sin ningún salto")
    g_min = gen_of("primera demostración mínima")
    g_greedy = gen_of("la política voraz")
    ok, n = tr.law_matches_theory()
    order = sorted(tr.first_greedy.items(), key=lambda kv: (kv[1], kv[0]))
    order_txt = ", ".join(f"{k} (gen {g})" for k, g in order)
    slowest = order[-1] if order else ("—", 0)
    depth = _depth()
    invalid_total = sum(tr.invalid_usage.values())
    distractors = sum(v for k, v in tr.invalid_usage.items() if step_by_key(k).distractor)
    cinf = tr.invalid_usage.get("D_CINF", 0)
    tvm = tr.invalid_usage.get("D_TVM", 0)
    term = tr.invalid_usage.get("D_TERM", 0)
    bad = tr.stop_bad_total + tr.abort_bad_total

    p = [
        f"**Fase 1 (serie de Taylor).** El agente suma términos de la serie en 0 para calcular f(x*) sin conocer f ni su "
        f"radio de convergencia; solo ve el tamaño del último término frente a la tolerancia y el cociente entre términos "
        f"consecutivos. En sus primeras 100 generaciones acertó el {first_100} % de las veces: paraba demasiado pronto o "
        f"seguía sumando series que no convergen. "
        + (f"En la generación {g_ratio} descubrió el criterio del cociente: si |tₙ/tₙ₋₁| ≥ 1 de forma sostenida (no solo al "
           f"principio, donde eˣ con x = 2.5 también crece), la serie no converge y hay que abandonar. " if g_ratio else "")
        + (f"En la {g_term} fijó el criterio del último término: parar cuando |tₙ| ≪ tol y los términos decrecen, que es la "
           f"cota de la cola geométrica |tₙ|·q/(1 − q). " if g_term else "")
        + (f"Desde la {g_law} su ley coincide con la teoría en todos los estados bien visitados. " if g_law else f"Al final coincide con la teoría en {ok}/{n} estados. ")
        + (f"Desde la generación {g_95} acierta en ≥ 95 % de los casos. " if g_95 else "")
        + f"Total: {tr.stop_ok_total} aproximaciones correctas, {tr.abort_ok_total} divergencias detectadas, {bad} veredictos "
        f"erróneos en {tr.ep1} generaciones ({fmt_time(tr.phase1_time)}); media final de {_mean(tr.iter_curve[-100:]):.1f} "
        f"términos por aproximación.",
        f"**Fase 2 (demostración).** Los pasos se asentaron en la política en este orden: {order_txt}. El último fue "
        f"{slowest[0]} (gen {slowest[1]}); la conclusión tiene profundidad {depth['QED']} en el grafo (Rolle iterado es la "
        f"cadena larga), así que su valor solo se propaga hacia atrás una vez recorrida entera. "
        + (f"Primera demostración completa en la generación {g_qed}, " if g_qed else "No logró una demostración completa, ")
        + (f"primera sin saltos lógicos en la {g_clean}, " if g_clean else "")
        + (f"primera mínima ({MIN_PROOF_LENGTH} pasos) en la {g_min}. " if g_min else "")
        + (f"La política voraz produce la demostración mínima desde la generación {g_greedy}. " if g_greedy else "")
        + f"Cometió {invalid_total} saltos lógicos, {distractors} con distractores: «f ∈ C^∞ luego su serie converge a f» "
        f"fue intentado {cinf} veces, «los términos tienden a 0 luego la serie converge» {term} veces y el non sequitur del "
        f"teorema del valor medio {tvm} veces; todos quedaron descartados. Total: {tr.qed_total} demostraciones completas y "
        f"{tr.minimal_total} mínimas en {tr.ep2} generaciones ({fmt_time(tr.phase2_time)}).",
        f"**Lectura.** En la práctica, el agente descubrió que lo que decide si una serie sirve es el cociente entre términos "
        f"(d'Alembert) y que el último término, junto con ese cociente, acota la cola. En la teoría, aprendió que el resto de "
        f"Lagrange sale de aplicar Rolle n+1 veces a una función auxiliar, y que la serie converge a f solo cuando ese resto "
        f"tiende a 0, cosa que ni la suavidad C^∞ ni el hecho de que los términos se hagan pequeños garantizan. Este resto "
        f"es justo lo que necesitará Newton en el proyecto 04. Tiempo total: {fmt_time(tr.elapsed)}.",
    ]
    return "\n\n".join(p)


def build_report(tr: Trainer) -> str:
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    law = tr.control_law()
    law_rows = "\n".join(
        f"| {TERM_LABELS[tb]} | {RATIO_LABELS[rb]} | {'n > 4' if late else 'n ≤ 4'} | {ACTION_SHORT[a] if a is not None else '?'} | "
        f"{'/'.join(ACTION_SHORT[x] for x in theoretical_action(tb, rb, late))} | "
        f"{'✔' if a is not None and a in theoretical_action(tb, rb, late) else ('—' if a is None else '✘')} |"
        for (tb, rb, late), a in law.items()
    )
    depth = _depth()
    order_rows = "\n".join(
        f"| {k} | {step_by_key(k).title} | {depth[k]} | {tr.first_valid.get(k, '—')} | {tr.first_greedy.get(k, '—')} | {tr.step_usage.get(k, 0)} |"
        for k in sorted(REQUIRED_KEYS, key=lambda k: (tr.first_greedy.get(k, 10**9), depth[k]))
    )
    inval_rows = "\n".join(f"| {k} | {step_by_key(k).title} | {v} |" for k, v in tr.invalid_usage.most_common(8))
    hitos = "\n".join(f"| {m.phase} | {m.episode} | {m.t:.1f} s | {m.text} |" for m in tr.milestones)
    ok, n = tr.law_matches_theory()
    return f"""# TaylorRL · informe de aprendizaje

Generado: {now} · semilla {tr.cfg.seed} · tolerancia {tr.cfg.tol:g} · tiempo total **{fmt_time(tr.elapsed)}**
(fase 1: {fmt_time(tr.phase1_time)}, fase 2: {fmt_time(tr.phase2_time)}).

## Síntesis: cómo aprendió

{synthesis(tr)}

## Demostración final (política voraz, sin exploración)

**Teorema (Taylor con resto de Lagrange).** Sea f ∈ Cⁿ⁺¹ en un intervalo que contiene a a y a x. Existe ξ entre a y x con
f(x) = Σₖ₌₀ⁿ f⁽ᵏ⁾(a)(x − a)ᵏ/k! + f⁽ⁿ⁺¹⁾(ξ)(x − a)ⁿ⁺¹/(n+1)!. Si |f⁽ᵏ⁾| ≤ M para todo k, la serie de Taylor converge a f(x).

{final_proof_markdown(tr)}

## Fase 1 · ley aprendida ({ok}/{n} estados bien visitados coinciden con la teoría)

Acciones: `+` sumar el siguiente término, `■` parar y entregar, `✗` abandonar (la serie no converge).

| último término | cociente q | n | acción aprendida | teoría | |
|---|---|---|---|---|---|
{law_rows}

Aproximaciones correctas: {tr.stop_ok_total} · divergencias detectadas: {tr.abort_ok_total} · veredictos erróneos: {tr.stop_bad_total + tr.abort_bad_total}
· agotadas: {tr.timeout_total} · éxito final (100 últimas): {100 * tr.success_curve[-1]:.0f} % · términos medios: {_mean(tr.iter_curve[-100:]):.1f}

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
            "terms_curve": tr.iter_curve,
            "law": {str(s): a for s, a in tr.control_law().items()},
            "q": {str(s): tr.agent1.q.get(s) for s in ALL_STATES},
            "stop_ok": tr.stop_ok_total, "stop_bad": tr.stop_bad_total,
            "abort_ok": tr.abort_ok_total, "abort_bad": tr.abort_bad_total, "timeout": tr.timeout_total,
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
