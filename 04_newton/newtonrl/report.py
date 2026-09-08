"""Síntesis final: informe en Markdown y volcado JSON de la historia."""

from __future__ import annotations

import json
import math
from datetime import datetime
from pathlib import Path

from .env_newton import ACTION_NAMES, ACTION_SHORT, RATIO_LABELS, theoretical_action
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


def _mean(xs) -> float:
    xs = [x for x in xs if x == x]
    return sum(xs) / len(xs) if xs else float("nan")


def synthesis(tr: Trainer) -> str:
    def gen_of(prefix: str) -> int | None:
        for m in tr.milestones:
            if m.text.startswith(prefix):
                return m.episode
        return None

    def text_of(prefix: str) -> str:
        for m in tr.milestones:
            if m.text.startswith(prefix):
                return m.text
        return ""

    first_100 = int(round(tr.success_curve[min(99, len(tr.success_curve) - 1)] * 100)) if tr.success_curve else 0
    g_full = gen_of("aprendió que cuando el residuo")
    g_back = gen_of("aprendió a rechazar")
    g_quad = gen_of("orden de convergencia")
    g_law = gen_of("ley estable")
    g_dev = gen_of("matiz")
    g_95 = gen_of("≥95 %")
    g_qed = gen_of("¡primera demostración completa!")
    g_clean = gen_of("primera demostración sin ningún salto")
    g_min = gen_of("primera demostración mínima")
    g_greedy = gen_of("la política voraz")
    ok, n = tr.law_matches_theory()
    law = tr.control_law()
    law_words = "; ".join(f"{tr.law_text(s)} → {ACTION_NAMES[a]}" for s, a in law.items() if a is not None and sum(tr.agent1.counts.get(s, [0])) >= 20)
    order = sorted(tr.first_greedy.items(), key=lambda kv: (kv[1], kv[0]))
    order_txt = ", ".join(f"{k} (gen {g})" for k, g in order)
    slowest = order[-1] if order else ("—", 0)
    depth = _depth()
    invalid_total = sum(tr.invalid_usage.values())
    distractors = sum(v for k, v in tr.invalid_usage.items() if step_by_key(k).distractor)
    glob = tr.invalid_usage.get("D_GLOBAL", 0)
    mult = tr.invalid_usage.get("D_MULT", 0)
    bolz = tr.invalid_usage.get("D_BOLZ", 0)
    orders = list(tr.orders)
    mean_order = f"{sum(orders) / len(orders):.2f}" if orders else "—"

    p = [
        f"**Fase 1 (Newton con salvaguardas).** El agente conoce el paso de Newton pero no cuándo es seguro darlo entero. "
        f"Solo observa la reducción del residuo ρ = |f(xₙ)|/|f(xₙ₋₁)| y si el paso propuesto crece o decrece, y elige entre paso "
        f"completo, amortiguado (½, ¼) o retroceder con búsqueda lineal. En las primeras 100 generaciones convergió el {first_100} % "
        f"de las veces (Newton puro converge ~73 % en este banco: arctan diverge desde |x₀| > 1.39, ∛x diverge desde cualquier x₀, "
        f"ln x salta fuera del dominio, x·e^(−x²) cicla). "
        + (f"En la generación {g_full} fijó que, cuando el residuo cae rápido, el paso completo es el mejor: es la cuenca de "
           f"convergencia cuadrática. " if g_full else "")
        + (f"En la {g_back} aprendió a rechazar los pasos que empeoran |f| y retroceder con búsqueda lineal: Newton amortiguado. " if g_back else "")
        + (f"Desde la {g_quad} el orden de convergencia observado en sus últimas iteraciones es ≈ {mean_order}: los dígitos correctos "
           f"se duplican en cada paso, tal como predice el teorema. " if g_quad else "")
        + (f"En la {g_law} su ley quedó estable: coincide con la regla de Armijo en {ok}/{n} estados bien visitados ({law_words}). " if g_law else f"Su ley final coincide con Armijo en {ok}/{n} estados bien visitados ({law_words}). ")
        + (f"Y descubrió un matiz que la regla no contempla (gen {g_dev}): si el paso de Newton propuesto ya decrece, dar el paso completo "
           f"es más barato que la búsqueda lineal, que cuesta varias evaluaciones. " if g_dev else "")
        + (f"Desde la generación {g_95} converge en ≥ 95 % de los casos. " if g_95 else "")
        + f"Total: {tr.converged_total} convergencias, {tr.diverged_total} divergencias y {tr.timeout_total} episodios agotados en "
        f"{tr.ep1} generaciones ({fmt_time(tr.phase1_time)}); media final de {_mean(tr.iter_curve[-100:]):.1f} evaluaciones por convergencia.",
        f"**Fase 2 (demostración).** Los pasos se asentaron en la política en este orden: {order_txt}. El último fue {slowest[0]} "
        f"(gen {slowest[1]}); la conclusión tiene profundidad {depth['QED']}. "
        + (f"Primera demostración completa en la generación {g_qed}, " if g_qed else "No logró una demostración completa, ")
        + (f"primera sin saltos lógicos en la {g_clean}, " if g_clean else "")
        + (f"primera mínima ({MIN_PROOF_LENGTH} pasos) en la {g_min}. " if g_min else "")
        + (f"La política voraz produce la demostración mínima desde la generación {g_greedy}. " if g_greedy else "")
        + f"Cometió {invalid_total} saltos lógicos, {distractors} con distractores: «Newton converge desde cualquier x₀» fue intentado "
        f"{glob} veces, «con raíz múltiple sigue siendo cuadrático» {mult} veces y el non sequitur de Bolzano {bolz} veces; todos "
        f"quedaron descartados. Total: {tr.qed_total} demostraciones completas y {tr.minimal_total} mínimas en {tr.ep2} generaciones "
        f"({fmt_time(tr.phase2_time)}).",
        f"**Lectura.** Las dos fases se explican mutuamente. El teorema dice que Newton es cuadrático *cerca* de una raíz simple: la "
        f"identidad del error sale del resto de Lagrange (proyecto 03) y el entorno seguro exige f' ≠ 0. La fase 1 muestra exactamente "
        f"los dos lados: dentro de la cuenca el agente da pasos completos y ve duplicarse los dígitos; fuera de ella (arctan lejos, ∛x "
        f"con f' infinita en la raíz, ln x fuera del dominio) el paso completo destruye la convergencia y hace falta amortiguar. "
        f"Tiempo total: {fmt_time(tr.elapsed)}.",
    ]
    return "\n\n".join(p)


def build_report(tr: Trainer) -> str:
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    law = tr.control_law()
    law_rows = "\n".join(
        f"| {RATIO_LABELS[b]} | {'crece' if g else 'decrece'} | {ACTION_SHORT[a] if a is not None else '?'} | "
        f"{'/'.join(ACTION_SHORT[x] for x in theoretical_action(b, g))} | "
        f"{'✔' if a is not None and a in theoretical_action(b, g) else ('—' if a is None else '✘ (matiz)')} | {sum(tr.agent1.counts.get((b, g), [0]))} |"
        for (b, g), a in law.items()
    )
    depth = _depth()
    order_rows = "\n".join(
        f"| {k} | {step_by_key(k).title} | {depth[k]} | {tr.first_valid.get(k, '—')} | {tr.first_greedy.get(k, '—')} | {tr.step_usage.get(k, 0)} |"
        for k in sorted(REQUIRED_KEYS, key=lambda k: (tr.first_greedy.get(k, 10**9), depth[k]))
    )
    inval_rows = "\n".join(f"| {k} | {step_by_key(k).title} | {v} |" for k, v in tr.invalid_usage.most_common(8))
    hitos = "\n".join(f"| {m.phase} | {m.episode} | {m.t:.1f} s | {m.text} |" for m in tr.milestones)
    ok, n = tr.law_matches_theory()
    orders = list(tr.orders)
    mean_order = f"{sum(orders) / len(orders):.2f}" if orders else "—"
    return f"""# NewtonRL · informe de aprendizaje

Generado: {now} · semilla {tr.cfg.seed} · tolerancia {tr.cfg.tol:g} · tiempo total **{fmt_time(tr.elapsed)}**
(fase 1: {fmt_time(tr.phase1_time)}, fase 2: {fmt_time(tr.phase2_time)}).

## Síntesis: cómo aprendió

{synthesis(tr)}

## Demostración final (política voraz, sin exploración)

**Teorema (convergencia cuadrática local de Newton).** Sea f ∈ C² en un entorno de r con f(r) = 0 y f'(r) ≠ 0. Existe δ > 0 tal
que para todo x₀ con |x₀ − r| ≤ δ la sucesión xₙ₊₁ = xₙ − f(xₙ)/f'(xₙ) converge a r y |xₙ₊₁ − r| ≤ (M/2m)·|xₙ − r|².

{final_proof_markdown(tr)}

## Fase 1 · ley de control aprendida ({ok}/{n} estados coinciden con Armijo; orden observado {mean_order})

Acciones: `λ1` paso completo, `λ½` y `λ¼` amortiguados, `↩` retroceder con búsqueda lineal.

| ρ = |f(xₙ)|/|f(xₙ₋₁)| | paso propuesto | acción aprendida | Armijo (c = ½) | | visitas |
|---|---|---|---|---|---|
{law_rows}

Convergencias: {tr.converged_total} · divergencias: {tr.diverged_total} · agotadas: {tr.timeout_total}
· éxito final (100 últimas): {100 * tr.success_curve[-1]:.0f} % · evaluaciones medias: {_mean(tr.iter_curve[-100:]):.1f}

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
