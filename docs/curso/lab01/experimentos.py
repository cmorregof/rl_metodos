"""Laboratorio 1 · experimentos sobre el agente de bisección.

Se ejecuta desde 01_biseccion con su entorno virtual activado:

    cd 01_biseccion && source .venv/bin/activate
    python ../docs/curso/lab01/experimentos.py generaciones
    python ../docs/curso/lab01/experimentos.py recompensa
    python ../docs/curso/lab01/experimentos.py recompensa --perder-raiz 0 --paso -1 --converger 0
    python ../docs/curso/lab01/experimentos.py sin_medio
    python ../docs/curso/lab01/experimentos.py grafo

No modifica los archivos del paquete: cambia constantes en memoria y entrena de nuevo.
"""

from __future__ import annotations

import sys
from dataclasses import replace

import bisectrl.env_bisection as E
import bisectrl.proof_kb as KB
import bisectrl.trainer as T
from bisectrl.env_proof import ProofEnv


def entrenar(ep1: int, ep2: int, seed: int = 1) -> T.Trainer:
    tr = T.Trainer(T.Config(episodes_bisection=ep1, episodes_proof=ep2, seed=seed))
    for _ in tr.run():
        pass
    return tr


def resumen_fase1(tr: T.Trainer) -> str:
    return (
        f"λ* = {tr.lambda_policy()}  Bolzano = {tr.keep_policy_is_bolzano()}  "
        f"éxito últimas 100 = {tr.success_curve[-1]:.2f}  "
        f"convergencias = {tr.converged_total}  raíces perdidas = {tr.lost_total}"
    )


def resumen_fase2(tr: T.Trainer) -> str:
    keys = [KB.STEPS[a].key for a, ok in tr.greedy_proof if ok]
    limpia = all(ok for _, ok in tr.greedy_proof)
    return f"demostración voraz ({len(keys)} pasos, {'sin' if limpia else 'con'} saltos): {' → '.join(keys)}"


# ---------------------------------------------------------------------------
def exp_generaciones() -> None:
    """¿Cuántas generaciones necesita el agente? Cambia la semilla y compara."""
    for n in (100, 300, 1000, 3000):
        tr = entrenar(n, n)
        print(f"[{n:>4} generaciones]  {resumen_fase1(tr)}")
        print(f"                     {resumen_fase2(tr)}")
        for m in tr.milestones:
            if "descubrió" in m.text or "Bolzano" in m.text or "mínima" in m.text:
                print(f"                     hito gen {m.episode}: {m.text}")


def exp_recompensa(perder_raiz: float | None = None, paso: float | None = None, converger: float | None = None, seed: int = 1) -> None:
    """¿Qué pasa si perder la raíz no cuesta nada? ¿Y si cada paso es gratis?

    Sin argumentos corre los tres casos fijos. Con --perder-raiz / --paso / --converger
    entrena una vez con esos valores: es la forma de probar una recompensa propia."""
    base_lost, base_step, base_conv = E.REWARD_LOST_ROOT, E.REWARD_STEP, E.REWARD_CONVERGED
    if perder_raiz is not None or paso is not None or converger is not None:
        try:
            E.REWARD_LOST_ROOT = base_lost if perder_raiz is None else perder_raiz
            E.REWARD_STEP = base_step if paso is None else paso
            E.REWARD_CONVERGED = base_conv if converger is None else converger
            print(f"recompensa: perder la raíz = {E.REWARD_LOST_ROOT:+g}, cada paso = {E.REWARD_STEP:+g}, "
                  f"al converger = {E.REWARD_CONVERGED:+g} (semilla {seed})")
            print("resultado:  ", resumen_fase1(entrenar(2000, 10, seed)))
        finally:
            E.REWARD_LOST_ROOT, E.REWARD_STEP, E.REWARD_CONVERGED = base_lost, base_step, base_conv
        return
    try:
        print("[A] perder la raíz cuesta −10 (original):", resumen_fase1(entrenar(2000, 10)))
        E.REWARD_LOST_ROOT = 0.0
        print("[B] perder la raíz cuesta 0:              ", resumen_fase1(entrenar(2000, 10)))
        E.REWARD_LOST_ROOT = base_lost
        E.REWARD_STEP = 0.0
        print("[C] cada paso es gratis (0 en vez de −1): ", resumen_fase1(entrenar(2000, 10)))
    finally:
        E.REWARD_LOST_ROOT, E.REWARD_STEP = base_lost, base_step


def exp_sin_medio() -> None:
    """Si λ = 0.5 no está disponible, ¿qué elige el agente?"""
    original = E.LAMBDAS
    try:
        E.LAMBDAS = tuple(l for l in original if abs(l - 0.5) > 1e-9)
        T.LAMBDAS = E.LAMBDAS
        tr = entrenar(3000, 10)
        qs = tr.lambda_q()
        print("λ disponibles:", E.LAMBDAS)
        print("Q(λ):", {l: round(q, 3) for l, q in zip(E.LAMBDAS, qs)})
        print("elige λ* =", tr.lambda_policy())
    finally:
        E.LAMBDAS = original
        T.LAMBDAS = original


def exp_grafo() -> None:
    """Quitar una dependencia del grafo: ROOT deja de necesitar INV.
    El verificador acepta entonces una "demostración" que nunca prueba que el cambio
    de signo se conserva. Comprobación directa, sin entrenar."""
    i = KB.KEY_TO_INDEX["ROOT"]
    original = KB.STEPS[i]
    try:
        KB.STEPS[i] = replace(original, deps=("CONT",))
        KB.DEPS_MASK[i] = KB.deps_mask(KB.STEPS[i])
        orden = ["H1", "H2", "DEF", "WIDTH", "MONO_A", "MONO_B", "CONV_A", "CONV_B", "SAME", "CONT", "ROOT", "MID", "QED"]
        env = ProofEnv()
        env.reset()
        for k in orden:
            _, _, _, info = env.step(KB.KEY_TO_INDEX[k])
            print(f"  {k:7s} {info}")
        print("¿el verificador la acepta?", env.finished, "· ¿usa el invariante INV?", "INV" in orden)
        print("Pregunta: ¿dónde está el hueco? ¿Qué afirma ROOT y qué necesita de verdad?")
    finally:
        KB.STEPS[i] = original
        KB.DEPS_MASK[i] = KB.deps_mask(original)


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Experimentos del laboratorio 1 sobre el agente de bisección.")
    ap.add_argument("experimento", choices=["generaciones", "recompensa", "sin_medio", "grafo"], nargs="?", default="generaciones")
    ap.add_argument("--perder-raiz", type=float, help="recompensa por conservar el lado sin cambio de signo (original: -10)")
    ap.add_argument("--paso", type=float, help="recompensa por cada evaluación de f (original: -1)")
    ap.add_argument("--converger", type=float, help="bono al bajar de la tolerancia (original: 0)")
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()
    if args.experimento == "recompensa":
        exp_recompensa(args.perder_raiz, args.paso, args.converger, args.seed)
    else:
        {"generaciones": exp_generaciones, "sin_medio": exp_sin_medio, "grafo": exp_grafo}[args.experimento]()
