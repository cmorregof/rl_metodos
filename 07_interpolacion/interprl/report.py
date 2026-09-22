"""Síntesis final: informe en Markdown y volcado JSON de la historia."""

from __future__ import annotations

import json
import math
from datetime import datetime
from pathlib import Path

from .env_interp import ACTION_SHORT, EST_LABELS, GAMMAS, TREND_LABELS, theoretical_action
from .interp import chebyshev_lobatto, equispaced, lebesgue_constant, max_abs_nodal_poly
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
    g_cheb = gen_of("prefiere γ = 1")
    g_stop = gen_of("descubrió el criterio de parada")
    g_law = gen_of("su regla coincide")
    g_95 = gen_of("≥95 %")
    g_qed = gen_of("¡primera demostración completa!")
    g_clean = gen_of("primera demostración sin ningún salto")
    g_min = gen_of("primera demostración mínima")
    g_greedy = gen_of("la política voraz")
    ok, n = tr.law_matches_theory()
    qs = tr.gamma_q()
    counts = tr.agent_gamma.counts[("gamma",)]
    order = sorted(tr.first_greedy.items(), key=lambda kv: (kv[1], kv[0]))
    order_txt = ", ".join(f"{k} (gen {g})" for k, g in order)
    slowest = order[-1] if order else ("—", 0)
    depth = _depth()
    invalid_total = sum(tr.invalid_usage.values())
    distractors = sum(v for k, v in tr.invalid_usage.items() if step_by_key(k).distractor)
    more = tr.invalid_usage.get("D_MORE", 0)
    cont = tr.invalid_usage.get("D_CONT", 0)
    weier = tr.invalid_usage.get("D_WEIER", 0)
    tay = tr.invalid_usage.get("D_TAYLOR", 0)

    p = [
        f"**Fase 1 (interpolación).** El agente elige una familia de nodos (γ = 0 equiespaciados … γ = 1 Chebyshev) y luego "
        f"decide, nodo a nodo, si sigue o entrega, viendo solo la diferencia entre los dos últimos interpolantes. En sus "
        f"primeras 100 generaciones acertó el {first_100} % de las veces. "
        + (f"En la generación {g_stop} fijó el criterio de parada: entregar cuando |pₙ − pₙ₋₁| es mucho menor que la tolerancia "
           f"y va bajando, que es el estimador de error de cualquiera que no conoce f. " if g_stop else "")
        + (f"En la {g_cheb} pasó a preferir de forma estable γ = 1, los nodos de Chebyshev: con equiespaciados las funciones "
           f"tipo Runge empeoran al añadir nodos y el episodio se agota. " if g_cheb else "No llegó a preferir de forma estable los nodos de Chebyshev. ")
        + f"Valores finales del bandido: " + ", ".join(f"Q(γ={g:.2f}) = {q:+.1f} ({c} usos)" for g, q, c in zip(GAMMAS, qs, counts)) + ". "
        + (f"Desde la {g_law} su regla de parada coincide con la teoría en todos los estados bien visitados. " if g_law else f"Al final coincide con la teoría en {ok}/{n} estados. ")
        + (f"Desde la generación {g_95} acierta en ≥ 95 % de los casos. " if g_95 else "")
        + f"Total: {tr.stop_ok_total} aproximaciones correctas, {tr.stop_bad_total} entregas prematuras y {tr.timeout_total} "
        f"agotadas en {tr.ep1} generaciones ({fmt_time(tr.phase1_time)}); media final de {_mean(tr.node_curve[-100:]):.1f} nodos.",
        f"**Fase 2 (demostración).** Los pasos se asentaron en la política en este orden: {order_txt}. El último fue "
        f"{slowest[0]} (gen {slowest[1]}); la conclusión tiene profundidad {depth['QED']} (la cadena larga es función auxiliar → "
        f"ceros → Rolle iterado → fórmula → cota). "
        + (f"Primera demostración completa en la generación {g_qed}, " if g_qed else "No logró una demostración completa, ")
        + (f"primera sin saltos lógicos en la {g_clean}, " if g_clean else "")
        + (f"primera mínima ({MIN_PROOF_LENGTH} pasos) en la {g_min}. " if g_min else "")
        + (f"La política voraz produce la demostración mínima desde la generación {g_greedy}. " if g_greedy else "")
        + f"Cometió {invalid_total} saltos lógicos, {distractors} con distractores: «más nodos ⇒ menos error» fue intentado "
        f"{more} veces, «f continua ⇒ pₙ → f» {cont}, el non sequitur de Weierstrass {weier} y la confusión con el resto de "
        f"Taylor {tay}; todos quedaron descartados. Total: {tr.qed_total} demostraciones completas y {tr.minimal_total} mínimas "
        f"en {tr.ep2} generaciones ({fmt_time(tr.phase2_time)}).",
        f"**Lectura.** En la práctica, el agente descubrió que dónde se ponen los nodos importa más que cuántos: con nodos "
        f"equiespaciados hay funciones para las que añadir nodos empeora (Runge), y con los de Chebyshev el error baja "
        f"geométricamente. En la teoría, aprendió que el error es f⁽ⁿ⁺¹⁾(ξ)·w(x)/(n+1)! y que el único factor que los nodos "
        f"controlan es máx|w|, que Chebyshev minimiza. Lo que la teoría clásica no da es el conjunto de nodos que minimiza la "
        f"constante de Lebesgue: eso es la fase 3 (`python -m interprl search`), donde no hay grafo sino un verificador. "
        f"Tiempo total: {fmt_time(tr.elapsed)}.",
    ]
    return "\n\n".join(p)


def build_report(tr: Trainer) -> str:
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    law = tr.control_law()
    law_rows = "\n".join(
        f"| {EST_LABELS[eb]} | {TREND_LABELS[trend]} | {'n > 6' if late else 'n ≤ 6'} | {ACTION_SHORT[a] if a is not None else '?'} | "
        f"{'/'.join(ACTION_SHORT[x] for x in theoretical_action(eb, trend, late))} | "
        f"{'✔' if a is not None and a in theoretical_action(eb, trend, late) else ('—' if a is None else '✘')} |"
        for (eb, trend, late), a in law.items()
    )
    qs, counts = tr.gamma_q(), tr.agent_gamma.counts[("gamma",)]
    best_g = tr.gamma_policy()
    gamma_rows = "\n".join(
        f"| {g:.2f} | {q:+.2f} | {c} | {'★' if abs(g - best_g) < 1e-9 else ''} |" for g, q, c in zip(GAMMAS, qs, counts)
    )
    leb_rows = "\n".join(
        f"| {n} | {lebesgue_constant(equispaced(n))[0]:.3f} | {lebesgue_constant(chebyshev_lobatto(n))[0]:.3f} | "
        f"{max_abs_nodal_poly(equispaced(n)):.2e} | {max_abs_nodal_poly(chebyshev_lobatto(n)):.2e} |"
        for n in (4, 8, 12, 16, 24)
    )
    depth = _depth()
    order_rows = "\n".join(
        f"| {k} | {step_by_key(k).title} | {depth[k]} | {tr.first_valid.get(k, '—')} | {tr.first_greedy.get(k, '—')} | {tr.step_usage.get(k, 0)} |"
        for k in sorted(REQUIRED_KEYS, key=lambda k: (tr.first_greedy.get(k, 10**9), depth[k]))
    )
    inval_rows = "\n".join(f"| {k} | {step_by_key(k).title} | {v} |" for k, v in tr.invalid_usage.most_common(8))
    hitos = "\n".join(f"| {m.phase} | {m.episode} | {m.t:.1f} s | {m.text} |" for m in tr.milestones)
    ok, n = tr.law_matches_theory()
    return f"""# InterpRL · informe de aprendizaje

Generado: {now} · semilla {tr.cfg.seed} · tolerancia {tr.cfg.tol:g} · tiempo total **{fmt_time(tr.elapsed)}**
(fase 1: {fmt_time(tr.phase1_time)}, fase 2: {fmt_time(tr.phase2_time)}).

## Síntesis: cómo aprendió

{synthesis(tr)}

## Demostración final (política voraz, sin exploración)

**Teorema (error de interpolación).** Sea f ∈ Cⁿ⁺¹[a, b] y pₙ el interpolante en n + 1 nodos distintos. Para cada x existe ξ con
f(x) − pₙ(x) = f⁽ⁿ⁺¹⁾(ξ)·w(x)/(n+1)!, w(t) = ∏(t − xᵢ). Con nodos de Chebyshev en [−1, 1], máx|w| = 2⁻ⁿ es mínimo.

{final_proof_markdown(tr)}

## Fase 1 · familia de nodos aprendida

| γ | Q(γ) = retorno medio del episodio | usos | mejor |
|---|---|---|---|
{gamma_rows}

γ = 0 son nodos equiespaciados; γ = 1, Chebyshev–Lobatto. Por qué gana Chebyshev, en números (independientes del agente):

| n | Λ equiespaciados | Λ Chebyshev | máx\\|w\\| equiespaciados | máx\\|w\\| Chebyshev |
|---|---|---|---|---|
{leb_rows}

## Fase 1 · regla de parada aprendida ({ok}/{n} estados bien visitados coinciden con la teoría)

Acciones: `+` añadir un nodo, `■` parar y entregar. Estimador: est = máx|pₙ − pₙ₋₁| / escala.

| estimador | tendencia | n | acción aprendida | teoría | |
|---|---|---|---|---|---|
{law_rows}

Aproximaciones correctas: {tr.stop_ok_total} · entregas prematuras: {tr.stop_bad_total} · agotadas: {tr.timeout_total}
· éxito final (100 últimas): {100 * tr.success_curve[-1]:.0f} % · nodos medios: {_mean(tr.node_curve[-100:]):.1f}

## Fase 2 · orden en que se asentó cada paso

| paso | título | profundidad | 1ª vez válido (gen) | entra en política voraz (gen) | usos |
|---|---|---|---|---|---|
{order_rows}

Saltos lógicos más frecuentes:

| paso | tipo | intentos inválidos |
|---|---|---|
{inval_rows}

Demostraciones completas: {tr.qed_total} · mínimas: {tr.minimal_total} · mejor retorno: {tr.best_reward:+.2f} (gen {tr.best_episode}).

## Fase 3 · lo que la teoría no da

El teorema dice que Chebyshev minimiza máx|w|, pero no qué nodos minimizan la constante de Lebesgue Λ (la amplificación
del error en los datos). Eso solo se conoce numéricamente. `python -m interprl search --n 8` busca nodos con Λ menor que
la de Chebyshev extendido usando el verificador `lebesgue_constant`, y `verify` recalcula el resultado con otra malla.

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
            "nodes_curve": tr.node_curve,
            "gamma_q": dict(zip(map(str, GAMMAS), tr.gamma_q())),
            "gamma_counts": dict(zip(map(str, GAMMAS), tr.agent_gamma.counts[("gamma",)])),
            "law": {str(s): a for s, a in tr.control_law().items()},
            "q": {str(s): tr.agent_stop.q.get(s) for s in ALL_STATES},
            "stop_ok": tr.stop_ok_total, "stop_bad": tr.stop_bad_total, "timeout": tr.timeout_total,
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
