r"""Laboratorio 2 · experimentos sobre el agente de punto fijo.

Se ejecuta desde 02_punto_fijo con su entorno virtual:

    cd 02_punto_fijo
    .venv\Scripts\python ..\docs\curso\lab02\experimentos.py alfa_fijo     (Windows)
    .venv/bin/python ../docs/curso/lab02/experimentos.py alfa_fijo         (Mac)

Experimentos: alfa_fijo, hitos, ciego, potencial, dibujar, grafo.
No modifica los archivos del paquete: cambia cosas en memoria y entrena de nuevo.
"""

from __future__ import annotations

import math
from dataclasses import replace

import fixpointrl.env_fixpoint as E
import fixpointrl.proof_kb as KB
import fixpointrl.trainer as T
from fixpointrl.env_fixpoint import ACTION_SHORT, KEEP, FixpointEnv, ratio_bucket, theoretical_action
from fixpointrl.env_proof import ProofEnv


def entrenar(ep1: int, ep2: int, seed: int = 1, env_cls=None) -> T.Trainer:
    tr = T.Trainer(T.Config(episodes_fixpoint=ep1, episodes_proof=ep2, seed=seed))
    if env_cls is not None:
        tr.env1 = env_cls(tol=tr.cfg.tol, seed=seed)
    for _ in tr.run():
        pass
    return tr


def resumen_fase1(tr: T.Trainer) -> str:
    ok, n = tr.law_matches_theory()
    it = [v for v in tr.iter_curve[-100:] if not math.isnan(v)]
    return (
        f"éxito últimas 100 = {tr.success_curve[-1]:.2f}  convergencias = {tr.converged_total}  "
        f"divergencias = {tr.diverged_total}  agotadas = {tr.timeout_total}  "
        f"ley = teoría en {ok}/{n} estados  iteraciones medias = {(it[-1] if it else float('nan')):.1f}"
    )


def tabla_ley(tr: T.Trainer) -> None:
    law = tr.control_law()
    print(f"  {'ρ':<10} {'forma':<9} {'aprendida':<10} {'teoría'}")
    for (b, o), a in law.items():
        theo = "/".join(ACTION_SHORT[x] for x in theoretical_action(b, o))
        marca = "?" if a is None else ("✔" if a in theoretical_action(b, o) else "✘")
        print(f"  {E.RATIO_LABELS[b]:<10} {'oscila' if o else 'monótona':<9} {(ACTION_SHORT[a] if a is not None else '?'):<10} {theo:<8} {marca}")


# ---------------------------------------------------------------------------
def exp_alfa_fijo() -> None:
    """Sin agente: iterar x ← x − α f(x) con α fijo. ¿En cuántos problemas converge?"""
    print("x ← x − α·f(x) con α fijo, sin agente, 300 problemas por α:")
    for alpha in (-1.0, -0.5, -0.1, 0.1, 0.5, 1.0):
        env = FixpointEnv(seed=3)
        conv = 0
        for _ in range(300):
            env.reset()
            env.alpha = alpha
            done = False
            while not done:
                _, _, done, info = env.step(KEEP)
            conv += info == "converged"
        print(f"  α = {alpha:+5.2f}: converge en {conv / 3:5.1f} % de los problemas")
    print("Con el agente (ajusta α mirando ρ y la oscilación):")
    tr = entrenar(3000, 10)
    print("  " + resumen_fase1(tr))


def exp_hitos() -> None:
    """Entrena con 3000 generaciones e imprime los hitos y la ley de control aprendida."""
    tr = entrenar(3000, 3000)
    for m in tr.milestones:
        print(f"  gen {m.episode:>5} (fase {m.phase}): {m.text}")
    print()
    print(resumen_fase1(tr))
    tabla_ley(tr)


class EnvCiego(FixpointEnv):
    """Mismo entorno, pero el agente no ve si la iteración oscila: solo ve ρ."""

    def state(self) -> tuple:
        if self.n_iter == 0:
            return ("inicio",)
        rho = self.rho
        return (5 if not math.isfinite(rho) else ratio_bucket(rho), 0)


def exp_ciego() -> None:
    """¿Qué pasa si el estado no distingue 'oscila' de 'monótona'?"""
    print("[A] estado completo (ρ + oscila/monótona):", resumen_fase1(entrenar(3000, 10)))
    tr = entrenar(3000, 10, env_cls=EnvCiego)
    print("[B] estado ciego (solo ρ):                 ", resumen_fase1(tr))
    print("    Con ρ > 1 el agente no puede saber si α tiene el signo mal (invertir) o es demasiado grande (reducir).")
    print("    Lo que aprende en esos estados:")
    law = tr.control_law()
    for b in (4, 5):
        a = law.get((b, 0))
        print(f"      ρ {E.RATIO_LABELS[b]:<6}: {ACTION_SHORT[a] if a is not None else '?'}   (teoría: invertir si monótona, reducir si oscila)")


class EnvPotencialTramposo(FixpointEnv):
    """Shaping con un potencial que el agente SÍ controla: el tamaño del paso |xₙ₊₁ − xₙ|.
    Premia 'moverse cada vez menos', que se consigue reduciendo α hasta cero sin acercarse a la raíz."""

    def step(self, action: int) -> tuple[tuple, float, bool, str]:
        self._apply(action)
        old_step = abs(self.x - self.x_prev) if self.n_iter > 0 else abs(self.alpha * self.fx)
        self._iterate()
        self.n_iter += 1
        new_step = abs(self.x - self.x_prev)
        if not math.isfinite(self.x) or not math.isfinite(self.fx):
            return self.state(), E.REWARD_DIVERGED, True, "diverged"
        progress = math.log10(max(old_step, 1e-300) / max(new_step, 1e-300))
        progress = max(-3.0, min(6.0, progress))
        if abs(self.fx) < self.tol:
            return self.state(), E.REWARD_STEP + progress, True, "converged"
        if self.n_iter >= self.max_iter:
            return self.state(), E.REWARD_STEP + progress, True, "timeout"
        return self.state(), E.REWARD_STEP + progress, False, "step"


def exp_potencial() -> None:
    """La recompensa de progreso debe medir algo que el agente no pueda manipular."""
    print("[A] potencial = −log|f(x)| (el residuo; el agente no lo controla directamente):")
    print("    " + resumen_fase1(entrenar(3000, 10)))
    tr = entrenar(3000, 10, env_cls=EnvPotencialTramposo)
    print("[B] potencial = −log|xₙ₊₁ − xₙ| (el tamaño del paso; el agente lo controla con α):")
    print("    " + resumen_fase1(tr))
    alphas = [abs(a) for a in tr.env1.alpha_history]
    print(f"    último episodio: α empezó en {abs(tr.env1.alpha_history[0]) if alphas else float('nan'):.3g} y terminó en {alphas[-1] if alphas else float('nan'):.3g}; "
          f"|f(x)| final = {abs(tr.env1.fx):.2e} (tolerancia {tr.env1.tol:g})")
    print("    ley aprendida con el potencial tramposo (✘ = distinta de Banach):")
    tabla_ley(tr)
    print("    Pregunta: ¿por qué 'moverse cada vez menos' no es lo mismo que 'acercarse a la raíz'?")


def dibujar_grafo(resaltar: str | None = None, falta: str | None = None) -> None:
    from rich.console import Console
    from rich.tree import Tree

    console = Console()
    visto: set[str] = set()

    def rama(key: str, arbol: Tree) -> None:
        st = KB.step_by_key(key)
        etiqueta = f"[bold]{key}[/] [dim]{st.title}[/]"
        if key == resaltar:
            etiqueta = f"[bold red]{key}[/] [red]{st.title}[/]  [red]← ya no exige {falta}[/]"
        if key in visto:
            arbol.add(f"[dim]{key} (ya dibujado)[/]")
            return
        visto.add(key)
        nodo = arbol.add(etiqueta)
        for d in st.deps:
            rama(d, nodo)
        if key == resaltar and falta:
            nodo.add(f"[red strike]{falta}[/] [red]{KB.step_by_key(falta).title}  ← el hueco[/]")

    raiz = Tree("[bold green]∎ QED[/]  (cada paso necesita lo que cuelga de él)")
    for d in KB.step_by_key("QED").deps:
        rama(d, raiz)
    console.print(raiz)
    hojas = [k for k in KB.REQUIRED_KEYS if not KB.step_by_key(k).deps]
    console.print(f"[dim]hipótesis: {', '.join(sorted(hojas))} · pasos necesarios: {KB.MIN_PROOF_LENGTH} · distractores: {sum(1 for x in KB.STEPS if x.distractor)}[/]\n")


def exp_grafo() -> None:
    """Quitar la hipótesis de contracción (H2) de ERR: el verificador acepta una 'demostración'
    de convergencia que nunca usa que g es contracción. Es exactamente el distractor D_K1 colado."""
    i = KB.KEY_TO_INDEX["ERR"]
    original = KB.STEPS[i]
    print("=== el grafo original ===")
    dibujar_grafo()
    req, minlen = KB.REQUIRED_KEYS, KB.MIN_PROOF_LENGTH
    try:
        KB.STEPS[i] = replace(original, deps=("DEF", "EXIST"))
        KB.DEPS_MASK[i] = KB.deps_mask(KB.STEPS[i])
        KB.REQUIRED_KEYS = frozenset(KB._closure("QED", set()))
        KB.MIN_PROOF_LENGTH = len(KB.REQUIRED_KEYS)
        print("=== el grafo con la dependencia quitada ===")
        dibujar_grafo(resaltar="ERR", falta="H2")
        print("=== una demostración ante el verificador (fíjate en si H2 aparece antes de ERR) ===")
        orden = ["H1", "AUX", "EXIST", "DEF", "ERR", "GEO", "LIM", "PRIORI", "H2", "UNIQ", "STEP", "POST", "QED"]
        env = ProofEnv()
        env.reset()
        for k in orden:
            _, _, _, info = env.step(KB.KEY_TO_INDEX[k])
            print(f"  {k:7s} {info}")
        print("¿el verificador la acepta?", env.finished)
        print("Pregunta: ERR afirma |xₙ₊₁ − p| ≤ k·|xₙ − p|. ¿De dónde sale la k si todavía no se ha dicho que g es contracción?")
        print("          ¿Qué pasa con la convergencia (LIM) si k puede ser 1? (mira el distractor D_K1)")
    finally:
        KB.STEPS[i] = original
        KB.DEPS_MASK[i] = KB.deps_mask(original)
        KB.REQUIRED_KEYS, KB.MIN_PROOF_LENGTH = req, minlen


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Experimentos del laboratorio 2 sobre el agente de punto fijo.")
    ap.add_argument("experimento", choices=["alfa_fijo", "hitos", "ciego", "potencial", "dibujar", "grafo"], nargs="?", default="hitos")
    args = ap.parse_args()
    {"alfa_fijo": exp_alfa_fijo, "hitos": exp_hitos, "ciego": exp_ciego, "potencial": exp_potencial,
     "dibujar": dibujar_grafo, "grafo": exp_grafo}[args.experimento]()
