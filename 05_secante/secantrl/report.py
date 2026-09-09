"""Síntesis final: informe en Markdown y volcado JSON de la historia."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from .env_secant import ACTION_NAMES, ACTION_SHORT, FALSI, PHI, RATIO_LABELS, SEC, theoretical_action
from .proof_kb import MIN_PROOF_LENGTH, REQUIRED_KEYS, STEPS, step_by_key
from .trainer import ALL_STATES, Trainer, order_summary


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


def _mean(xs) -> float:
    xs = [x for x in xs if x == x]
    return sum(xs) / len(xs) if xs else float("nan")


def _fmt(x: float, nd: int = 2) -> str:
    return f"{x:.{nd}f}" if x == x else "—"


def synthesis(tr: Trainer) -> str:
    def gen_of(prefix: str) -> int | None:
        for m in tr.milestones:
            if m.text.startswith(prefix):
                return m.episode
        return None

    first_100 = int(round(tr.success_curve[min(99, len(tr.success_curve) - 1)] * 100)) if tr.success_curve else 0
    g_sec = gen_of("aprendió que cuando la secante")
    g_safe = gen_of("aprendió a no fiarse")
    g_phi = gen_of("orden de convergencia")
    g_law = gen_of("ley estable")
    g_dev = gen_of("matiz")
    g_95 = gen_of("≥95 %")
    g_qed = gen_of("¡primera demostración completa!")
    g_clean = gen_of("primera demostración sin ningún salto")
    g_min = gen_of("primera demostración mínima")
    g_greedy = gen_of("la política voraz")
    ok, n = tr.law_matches_theory()
    law = tr.control_law()
    law_words = "; ".join(f"{tr.law_text(s, pad=False)} → {ACTION_NAMES[a]}" for s, a in law.items() if a is not None and tr.visits(s) >= 20)
    devs = [tr.law_text(s, pad=False) for s, a in law.items() if a is not None and tr.visits(s) >= 100 and a not in theoretical_action(*s)]
    order = sorted(tr.first_greedy.items(), key=lambda kv: (kv[1], kv[0]))
    order_txt = ", ".join(f"{k} (gen {g})" for k, g in order)
    slowest = order[-1] if order else ("—", 0)
    depth = _depth()
    invalid_total = sum(tr.invalid_usage.values())
    distractors = sum(v for k, v in tr.invalid_usage.items() if step_by_key(k).distractor)
    quad = tr.invalid_usage.get("D_QUAD", 0)
    fixpt = tr.invalid_usage.get("D_FIXPT", 0)
    brack = tr.invalid_usage.get("D_BRACKET", 0)
    bolz = tr.invalid_usage.get("D_BOLZ", 0)
    med = tr.order_median()
    fam = {k: order_summary(v) for k, v in tr.orders_by_family.items() if len(v) >= 10}
    fam_txt = ""
    if len(fam) >= 2:
        lo_k = min(fam, key=fam.get)
        hi_k = max(fam, key=fam.get)
        fam_txt = f"Por familias, la mediana del orden va de {_fmt(fam[lo_k])} en {lo_k} a {_fmt(fam[hi_k])} en {hi_k}. "
    chained = tr.rf_chain_fraction()
    rf_share = tr.action_usage[FALSI] / max(1, sum(tr.action_usage.values()))
    if chained == chained:
        rf_txt = (
            f"La regula falsi la reserva para el refugio ({100 * rf_share:.0f} % de sus acciones) y "
            + (f"casi nunca la encadena: solo el {100 * chained:.0f} % de esos pasos sigue a otro. "
               if chained < 0.15 else
               f"la encadena a rachas cortas cuando el residuo empeora dentro del corchete ({100 * chained:.0f} % de esos pasos sigue a otro), "
               f"pero nunca la adopta como método: en las últimas 100 generaciones converge con {_mean(tr.iter_curve[-100:]):.1f} evaluaciones, "
               f"lejos de las ~22 de la regula falsi pura. ")
        )
    else:
        rf_txt = ""

    p = [
        f"**Fase 1 (secante con salvaguardas).** El agente conoce el paso de la secante pero no cuándo fiarse de él. Solo observa la "
        f"reducción del residuo ρ = |f(xₙ)|/|f(xₙ₋₁)| y si el punto que propone la secante cae dentro del corchete con cambio de signo, "
        f"y elige entre secante (dos últimos puntos), regula falsi (extremos del corchete) o bisección. En las primeras 100 generaciones "
        f"convergió el {first_100} % de las veces (en este banco la secante pura converge ~69 %: ∛x oscila, x·e^(−x²) dispara puntos "
        f"lejos, ln x cae fuera del dominio; la regula falsi pura ~63 % porque un extremo se estanca; la bisección siempre, pero con ~32 evaluaciones). "
        + (f"En la generación {g_sec} fijó que, cuando la secante cae dentro del corchete y el residuo baja, el paso de secante es el mejor: "
           f"es el régimen superlineal. " if g_sec else "")
        + (f"En la {g_safe} aprendió a no fiarse de la secante cuando se sale del corchete y volver a él ({tr.refuge_text()}): la salvaguarda "
           f"de Dekker que hereda de la bisección del proyecto 01. " if g_safe else "")
        + (f"Desde la {g_phi} la mediana del orden de convergencia observado en sus rachas de secante es ≈ {_fmt(med)}: φ = (1+√5)/2 ≈ "
           f"{PHI:.3f}, superlineal pero no cuadrático, tal como predice el teorema. " if g_phi else f"La mediana del orden observado es {_fmt(med)}. ")
        + fam_txt
        + (f"En la {g_law} su ley quedó estable: coincide con Dekker en {ok}/{n} estados bien visitados ({law_words}). " if g_law else f"Su ley final coincide con Dekker en {ok}/{n} estados bien visitados ({law_words}). ")
        + (f"Y descubrió un matiz que la regla no contempla (gen {g_dev}): si la secante cae dentro del corchete, seguirla aunque el residuo apenas "
           f"baje sale más barato que bisecar, porque bisecar tira el punto actual ({', '.join(devs)}). " if g_dev else "")
        + rf_txt
        + (f"Desde la generación {g_95} converge en ≥ 95 % de los casos. " if g_95 else "")
        + f"Total: {tr.converged_total} convergencias, {tr.diverged_total} divergencias y {tr.timeout_total} episodios agotados en "
        f"{tr.ep1} generaciones ({fmt_time(tr.phase1_time)}); media final de {_mean(tr.iter_curve[-100:]):.1f} evaluaciones por convergencia.",
        f"**Fase 2 (demostración).** Los pasos se asentaron en la política en este orden: {order_txt}. El último fue {slowest[0]} "
        f"(gen {slowest[1]}); la conclusión tiene profundidad {depth['QED']}. "
        + (f"Primera demostración completa en la generación {g_qed}, " if g_qed else "No logró una demostración completa, ")
        + (f"primera sin saltos lógicos en la {g_clean}, " if g_clean else "")
        + (f"primera mínima ({MIN_PROOF_LENGTH} pasos) en la {g_min}. " if g_min else "")
        + (f"La política voraz produce la demostración mínima desde la generación {g_greedy}. " if g_greedy else "")
        + f"Cometió {invalid_total} saltos lógicos, {distractors} con distractores: «es Newton con la derivada aproximada, luego cuadrático» "
        f"fue intentado {quad} veces, «es un punto fijo, luego lineal» {fixpt} veces, «los puntos encierran la raíz, luego (b−a)/2ⁿ» {brack} veces "
        f"y el non sequitur de Bolzano {bolz} veces; todos quedaron descartados. Total: {tr.qed_total} demostraciones completas y "
        f"{tr.minimal_total} mínimas en {tr.ep2} generaciones ({fmt_time(tr.phase2_time)}).",
        f"**Lectura.** Las dos fases se explican mutuamente. El teorema dice que la secante es superlineal de orden φ *cerca* de una raíz simple: "
        f"la identidad del error sale de la forma de Newton del interpolante (diferencias divididas, el papel que en Newton hacía el resto de "
        f"Lagrange), la cota es un producto |eₙ|·|eₙ₋₁| en vez de un cuadrado, y de ahí la recurrencia de Fibonacci y p² = p + 1. La fase 1 "
        f"muestra los dos lados: dentro del corchete el agente encadena secantes y ve el orden ≈ {_fmt(med)}; cuando la secante se sale "
        f"(arctan con corchete asimétrico, ∛x, ln x) el paso destruye la convergencia y hace falta el corchete. Y de paso descubre por qué "
        f"la regula falsi no es un método sino un refugio: un paso por los extremos es un buen paso de interpolación, encadenarlos es lineal. "
        f"Tiempo total: {fmt_time(tr.elapsed)}.",
    ]
    return "\n\n".join(p)


def build_report(tr: Trainer) -> str:
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    law = tr.control_law()
    law_rows = "\n".join(
        f"| {RATIO_LABELS[b]} | {'dentro' if i else 'fuera'} | {ACTION_SHORT[a] if a is not None else '?'} | "
        f"{'/'.join(ACTION_SHORT[x] for x in theoretical_action(b, i))} | "
        f"{'✔' if a is not None and a in theoretical_action(b, i) else ('—' if a is None else '✘ (matiz)')} | {tr.visits((b, i))} |"
        for (b, i), a in law.items()
    )
    fam_rows = "\n".join(
        f"| {k} | {len(v)} | {_fmt(order_summary(v))} |"
        for k, v in sorted(tr.orders_by_family.items(), key=lambda kv: -len(kv[1]))
    )
    depth = _depth()
    order_rows = "\n".join(
        f"| {k} | {step_by_key(k).title} | {depth[k]} | {tr.first_valid.get(k, '—')} | {tr.first_greedy.get(k, '—')} | {tr.step_usage.get(k, 0)} |"
        for k in sorted(REQUIRED_KEYS, key=lambda k: (tr.first_greedy.get(k, 10**9), depth[k]))
    )
    inval_rows = "\n".join(f"| {k} | {step_by_key(k).title} | {v} |" for k, v in tr.invalid_usage.most_common(8))
    hitos = "\n".join(f"| {m.phase} | {m.episode} | {m.t:.1f} s | {m.text} |" for m in tr.milestones)
    ok, n = tr.law_matches_theory()
    med = tr.order_median()
    usage = " · ".join(f"{ACTION_SHORT[a]}: {tr.action_usage[a]}" for a in (0, 1, 2))
    return f"""# SecantRL · informe de aprendizaje

Generado: {now} · semilla {tr.cfg.seed} · tolerancia {tr.cfg.tol:g} · tiempo total **{fmt_time(tr.elapsed)}**
(fase 1: {fmt_time(tr.phase1_time)}, fase 2: {fmt_time(tr.phase2_time)}).

## Síntesis: cómo aprendió

{synthesis(tr)}

## Demostración final (política voraz, sin exploración)

**Teorema (convergencia local de orden φ de la secante).** Sea f ∈ C² en un entorno de r con f(r) = 0 y f'(r) ≠ 0. Existe δ > 0 tal
que para todo par x₀ ≠ x₁ con |xᵢ − r| ≤ δ la sucesión xₙ₊₁ = xₙ − f(xₙ)(xₙ − xₙ₋₁)/(f(xₙ) − f(xₙ₋₁)) converge a r,
|xₙ₊₁ − r| ≤ (M/2m)·|xₙ − r|·|xₙ₋₁ − r|, y el orden de convergencia es φ = (1 + √5)/2 ≈ 1.618.

{final_proof_markdown(tr)}

## Fase 1 · ley de control aprendida ({ok}/{n} estados coinciden con Dekker; orden observado {_fmt(med)}, φ = {PHI:.3f})

Acciones: `sec` secante por los dos últimos puntos, `rf` regula falsi por los extremos del corchete, `bis` bisección.
Uso total: {usage}.

| ρ = |f(xₙ)|/|f(xₙ₋₁)| | la secante cae | acción aprendida | Dekker | | visitas |
|---|---|---|---|---|---|
{law_rows}

Convergencias: {tr.converged_total} · divergencias: {tr.diverged_total} · agotadas: {tr.timeout_total}
· éxito final (100 últimas): {100 * tr.success_curve[-1]:.0f} % · evaluaciones medias: {_mean(tr.iter_curve[-100:]):.1f}
· pasos de regula falsi encadenados (100 últimas): {_fmt(100 * tr.rf_chain_fraction(), 0)} %

Orden de convergencia observado por familia (mediana de las rachas de secante al converger):

| familia | rachas | orden |
|---|---|---|
{fam_rows}

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
            "evals_curve": tr.iter_curve,
            "observed_orders": list(tr.orders),
            "orders_by_family": tr.orders_by_family,
            "action_usage": {ACTION_SHORT[a]: n for a, n in tr.action_usage.items()},
            "rf_chain_fraction": tr.rf_chain_fraction(),
            "law": {str(s): a for s, a in tr.control_law().items()},
            "q": {str(s): tr.agent1.q.get(s) for s in ALL_STATES},
            "converged": tr.converged_total, "diverged": tr.diverged_total, "timeout": tr.timeout_total,
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
