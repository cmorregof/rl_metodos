"""Síntesis final: informe en Markdown y volcado JSON de la historia."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from .env_bisection import LAMBDAS
from .proof_kb import MIN_PROOF_LENGTH, REQUIRED_KEYS, STEPS, step_by_key
from .trainer import Trainer


def fmt_time(t: float) -> str:
    m, s = divmod(t, 60)
    return f"{int(m)} min {s:04.1f} s" if m else f"{s:.1f} s"


def final_proof_markdown(tr: Trainer) -> str:
    lines = []
    n = 0
    for a, ok in tr.greedy_proof:
        st = STEPS[a]
        if ok:
            n += 1
            lines.append(f"{n}. **{st.title}** ({st.key}). {st.text}")
        else:
            lines.append(f"   - ✗ *salto lógico:* {st.text}")
    return "\n".join(lines)


def synthesis(tr: Trainer) -> str:
    """Prosa generada a partir de los datos: cómo aprendió el agente."""
    ms = {m.text.split(":")[0]: m for m in tr.milestones}

    def gen_of(prefix: str) -> int | None:
        for m in tr.milestones:
            if m.text.startswith(prefix):
                return m.episode
        return None

    first_100_success = (
        sum(1 for v in tr.ratio_curve[:100] if v > 0) if tr.ratio_curve else 0
    )
    g_bolz = gen_of("aprendió el invariante de Bolzano")
    g_half = gen_of("descubrió el corte óptimo")
    g_95 = gen_of("≥95 %")
    g_qed = gen_of("¡primera demostración completa!")
    g_clean = gen_of("primera demostración sin ningún salto")
    g_min = gen_of("primera demostración mínima")
    g_greedy = gen_of("la política voraz")
    qs = tr.lambda_q()
    q_half = qs[LAMBDAS.index(0.5)]
    q_neighbors = (qs[LAMBDAS.index(0.4)] + qs[LAMBDAS.index(0.6)]) / 2

    order = sorted(tr.first_greedy.items(), key=lambda kv: (kv[1], kv[0]))
    order_txt = ", ".join(f"{k} (gen {g})" for k, g in order)
    slowest = order[-1] if order else ("—", 0)
    depth = _depth()
    deepest = max(depth, key=depth.get)

    non_seq = tr.invalid_usage.get("D_BOLZ", 0)
    distractors = sum(v for k, v in tr.invalid_usage.items() if step_by_key(k).distractor)
    invalid_total = sum(tr.invalid_usage.values())

    p = []
    p.append(
        f"**Fase 1 (bisección).** En las primeras 100 generaciones el agente convergió solo "
        f"{first_100_success} veces: cortaba en fracciones arbitrarias y con frecuencia conservaba el "
        f"subintervalo equivocado, perdiendo la raíz. "
        + (f"En la generación {g_bolz} su regla de conservación quedó fijada: guardar siempre el lado donde "
           f"f cambia de signo, que es exactamente el invariante f(aₙ)·f(bₙ) ≤ 0 de la demostración. "
           if g_bolz else "No llegó a fijar por completo la regla de conservación. ")
        + (f"En la generación {g_half} el corte λ = 1/2 pasó a ser, de forma estable, el de mayor valor "
           f"(Q = {q_half:+.3f} frente a {q_neighbors:+.3f} de sus vecinos 0.4 y 0.6): el agente redescubrió "
           f"que el punto medio maximiza el progreso esperado por evaluación, es decir, el método de bisección. "
           if g_half else "El corte λ = 1/2 no se estabilizó como el mejor. ")
        + (f"Desde la generación {g_95} converge en ≥ 95 % de los casos. " if g_95 else "")
        + f"Total: {tr.converged_total} convergencias y {tr.lost_total} raíces perdidas en {tr.ep1} generaciones "
        f"({fmt_time(tr.phase1_time)})."
    )
    p.append(
        f"**Fase 2 (demostración).** Los primeros pasos válidos fueron las hipótesis, que no dependen de nada; "
        f"después el agente fue asentando lemas en el orden en que sus dependencias quedaban disponibles: {order_txt}. "
        f"El último paso en entrar en la política fue {slowest[0]} (gen {slowest[1]}); el paso más profundo del "
        f"grafo de dependencias es {deepest} (profundidad {depth[deepest]}), lo que explica por qué la conclusión "
        f"se aprende al final: su valor solo se propaga hacia atrás una vez que la cadena completa ha sido recorrida. "
        + (f"La primera demostración completa apareció en la generación {g_qed}, " if g_qed else "No logró una demostración completa, ")
        + (f"la primera sin saltos lógicos en la {g_clean}, " if g_clean else "")
        + (f"y la primera mínima ({MIN_PROOF_LENGTH} pasos) en la {g_min}. " if g_min else "")
        + (f"La política voraz produce la demostración mínima desde la generación {g_greedy}. " if g_greedy else "")
        + f"En total cometió {invalid_total} saltos lógicos, {distractors} de ellos con lemas distractores; el "
        f"non sequitur «por Bolzano existe raíz, luego el método converge» fue intentado {non_seq} veces y "
        f"quedó descartado por su recompensa negativa. Total: {tr.qed_total} demostraciones completas y "
        f"{tr.minimal_total} mínimas en {tr.ep2} generaciones ({fmt_time(tr.phase2_time)})."
    )
    p.append(
        f"**Lectura.** Nada del método ni de la demostración estaba programado como regla: solo la recompensa "
        f"(progreso por evaluación, penalización por perder el cambio de signo, coste por línea y bono por ∎) y "
        f"un verificador de dependencias. Con eso, una tabla Q descubrió el punto medio, el invariante de Bolzano "
        f"y el orden lógico monotonía → convergencia → mismo límite → continuidad → raíz → cota → ∎. "
        f"Tiempo total: {fmt_time(tr.elapsed)}."
    )
    return "\n\n".join(p)


def _depth() -> dict[str, int]:
    memo: dict[str, int] = {}

    def d(k: str) -> int:
        if k not in memo:
            st = step_by_key(k)
            memo[k] = 0 if not st.deps else 1 + max(d(x) for x in st.deps)
        return memo[k]

    return {k: d(k) for k in REQUIRED_KEYS}


def build_report(tr: Trainer) -> str:
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    qs = tr.lambda_q()
    lam_rows = "\n".join(
        f"| {lam:.1f} | {q:+.3f} | {'★' if lam == tr.lambda_policy() else ''} |" for lam, q in zip(LAMBDAS, qs)
    )
    keep_rows = "\n".join(
        f"| ({'+' if sa > 0 else '−'}, {'+' if sx > 0 else '−'}, {'+' if sb > 0 else '−'}) | {k} | "
        f"{'✔' if k == ('izq' if sa * sx < 0 else 'der') else '✘'} |"
        for (sa, sx, sb), k in sorted(tr.keep_policy().items()) if sx != 0
    )
    depth = _depth()
    order_rows = "\n".join(
        f"| {k} | {step_by_key(k).title} | {depth[k]} | {tr.first_valid.get(k, '—')} | {tr.first_greedy.get(k, '—')} | {tr.step_usage.get(k, 0)} |"
        for k in sorted(REQUIRED_KEYS, key=lambda k: (tr.first_greedy.get(k, 10**9), depth[k]))
    )
    inval_rows = "\n".join(
        f"| {k} | {step_by_key(k).title} | {v} |" for k, v in tr.invalid_usage.most_common(8)
    )
    hitos = "\n".join(f"| {m.phase} | {m.episode} | {m.t:.1f} s | {m.text} |" for m in tr.milestones)
    return f"""# BisectRL · informe de aprendizaje

Generado: {now} · semilla {tr.cfg.seed} · tolerancia {tr.cfg.tol:g} · tiempo total **{fmt_time(tr.elapsed)}**
(fase 1: {fmt_time(tr.phase1_time)}, fase 2: {fmt_time(tr.phase2_time)}).

## Síntesis: cómo aprendió

{synthesis(tr)}

## Demostración final (política voraz, sin exploración)

**Teorema.** Sea f continua en [a, b] con f(a)·f(b) < 0. Las sucesiones de la bisección satisfacen
mₙ → c con f(c) = 0 y |mₙ − c| ≤ (b − a)/2ⁿ⁺¹.

{final_proof_markdown(tr)}

## Fase 1 · política de corte aprendida

| λ | Q(λ) = progreso medio − 1 | mejor |
|---|---|---|
{lam_rows}

Regla de conservación (signos de f(a), f(x), f(b) → lado conservado):

| signos | lado | ¿preserva el cambio de signo? |
|---|---|---|
{keep_rows}

Convergencias: {tr.converged_total} · raíces perdidas: {tr.lost_total} · éxito final (100 últimas): {100 * tr.success_curve[-1]:.0f} %
· iteraciones / óptimo (100 últimas): {_mean([v for v in tr.ratio_curve[-100:] if v > 0]):.2f}

## Fase 2 · orden en que se asentó cada paso

| paso | título | profundidad | 1ª vez válido (gen) | entra en política voraz (gen) | usos en demostraciones |
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


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else float("nan")


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
            "ratio_curve": tr.ratio_curve,
            "lambda_q": dict(zip(map(str, LAMBDAS), tr.lambda_q())),
            "keep_policy": {str(k): v for k, v in tr.keep_policy().items()},
            "converged": tr.converged_total,
            "lost": tr.lost_total,
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
    (run_dir / "historia.json").write_text(json.dumps(history, ensure_ascii=False, indent=1), encoding="utf-8")
    return run_dir
