r"""Laboratorio 3 · experimentos sobre el agente de interpolación (proyecto 07).

Se ejecuta desde 07_interpolacion con su entorno virtual:

    cd 07_interpolacion
    .venv\Scripts\python ..\docs\curso\lab03\experimentos.py basales        (Windows)
    .venv/bin/python ../docs/curso/lab03/experimentos.py basales            (Mac)

Experimentos:
    basales [--nodos=-1,0,1] [--t=-0.5,0.5,2]   valores exactos de los basales (para revisar la Parte 1)
    familias                                   sin agente: ¿cuántos nodos necesita cada familia?
    hitos [--seed 1]                           entrena y lista los hitos (para el docente)
    solo_equi                                  el agente no puede elegir Chebyshev
    sin_runge                                  el agente solo ve funciones fáciles
    recompensa [--entregar-mal X --nodo Y --entregar-bien Z]
    dibujar                                    el grafo del teorema del error, desde ∎
    grafo                                      quitamos una dependencia y el verificador no se entera
    mi_grafo [archivo]                         revisa tu grafo (mi_grafo.py) y pone al agente a ordenarlo

No modifica los archivos del paquete: cambia cosas en memoria y entrena de nuevo.
"""

from __future__ import annotations

import argparse
import importlib.util
import math
import sys
import time
from dataclasses import replace
from fractions import Fraction
from pathlib import Path

import interprl.env_interp as E
import interprl.proof_kb as KB
import interprl.trainer as T
from interprl.agent import QAgent
from interprl.env_proof import ProofEnv
from interprl.interp import lebesgue_constant, nodes

AQUI = Path(__file__).resolve().parent


# ---------------------------------------------------------------------------
# utilidades
def entrenar(ep1: int, ep2: int = 10, seed: int = 1) -> T.Trainer:
    tr = T.Trainer(T.Config(episodes_interp=ep1, episodes_proof=ep2, seed=seed))
    for _ in tr.run():
        pass
    return tr


def tipo(nombre: str) -> str:
    """Familia del problema a partir de su nombre: 'difícil' si tiene una singularidad cerca de [−1, 1]."""
    if nombre.startswith("1/(1+"):
        return "Runge"
    if nombre.startswith("1/(x"):
        return "polo"
    if nombre.startswith("√"):
        return "raíz"
    return "entera"


def resumen_fase1(tr: T.Trainer) -> str:
    qs = tr.gamma_q()
    gam = T.GAMMAS
    qtxt = "  ".join(f"Q(γ={g:g}) = {q:+.1f}" for g, q in zip(gam, qs))
    nodos_medios = next((v for v in reversed(tr.node_curve) if not math.isnan(v)), float("nan"))
    return (
        f"éxito últimas 100 = {tr.success_curve[-1]:.0%}  correctas = {tr.stop_ok_total}  "
        f"prematuras = {tr.stop_bad_total}  agotadas = {tr.timeout_total}  nodos medios = {nodos_medios:.1f}\n"
        f"    familia preferida: γ = {tr.gamma_policy():g}   {qtxt}"
    )


def fraccion(v: float) -> str:
    f = Fraction(v).limit_denominator(1000)
    return str(f) if abs(float(f) - v) < 1e-12 else f"{v:.6g}"


# ---------------------------------------------------------------------------
def exp_basales(nodos_txt: str | None, t_txt: str | None) -> None:
    """Valores exactos de los basales de Lagrange, con fracciones, para revisar lo hecho a mano."""
    if nodos_txt is None:
        casos = [((-1, 0, 1), (-0.5, 0.5, 2)), ((0, 1, 3), (2,))]
    else:
        xs = tuple(Fraction(v) for v in nodos_txt.split(","))
        ts = tuple(Fraction(v) for v in (t_txt or "0").split(","))
        casos = [(xs, ts)]
    for xs, ts in casos:
        xs = [Fraction(v) for v in xs]
        if len(set(xs)) != len(xs):
            print("Los nodos deben ser distintos.")
            return
        print(f"nodos {', '.join(str(v) for v in xs)}")
        print(f"  {'t':>6}  " + "  ".join(f"{'L' + str(k):>7}" for k in range(len(xs))) + f"  {'suma':>6}  {'suma |L|':>9}")
        for t in ts:
            t = Fraction(t)
            L = []
            for k, xk in enumerate(xs):
                v = Fraction(1)
                for j, xj in enumerate(xs):
                    if j != k:
                        v *= (t - xj) / (xk - xj)
                L.append(v)
            print(f"  {str(t):>6}  " + "  ".join(f"{str(v):>7}" for v in L) + f"  {str(sum(L)):>6}  {str(sum(abs(v) for v in L)):>9}")
        if min(xs) == -1 and max(xs) == 1:
            lam, tmax = lebesgue_constant([float(v) for v in xs])
            print(f"  constante de Lebesgue en [−1, 1]: Λ = {fraccion(lam)} (el máximo de suma |L|, en t ≈ {tmax:+.4f})")
        print()
    print("La columna 'suma |L|' es la función de Lebesgue λ(t): cuánto puede amplificar el polinomio un error en los datos.")


# ---------------------------------------------------------------------------
def exp_familias(n_problemas: int = 120) -> None:
    """Sin agente: para cada problema, el menor número de nodos que logra el error pedido con cada familia."""
    env = E.InterpEnv(tol=1e-5, seed=3)
    filas: dict[str, dict[float, list]] = {}
    for _ in range(n_problemas):
        env.reset()
        k = tipo(env.problem.name)
        for g in (0.0, 1.0):
            filas.setdefault(k, {}).setdefault(g, []).append(env.optimal_nodes(g))
    print(f"{n_problemas} problemas, tolerancia 1e-5, máximo {E.N_MAX} nodos.")
    print(f"  {'tipo de f':<9} {'cuántos':>7}   {'equiespaciados (γ=0)':<30} {'Chebyshev (γ=1)':<30}")
    for k, d in sorted(filas.items()):
        celdas = []
        for g in (0.0, 1.0):
            v = d[g]
            ok = [n for n in v if n is not None]
            media = f"{sum(ok) / len(ok):.1f} nodos" if ok else "—"
            celdas.append(f"logra {len(ok):>3}/{len(v):<3} · {media}")
        print(f"  {k:<9} {len(d[0.0]):>7}   {celdas[0]:<30} {celdas[1]:<30}")
    print("'logra' = alcanza el error pedido antes de agotar los nodos; la media es sobre los que lo logran.")


# ---------------------------------------------------------------------------
def exp_hitos(seed: int = 1) -> None:
    t0 = time.perf_counter()
    tr = entrenar(3000, 3000, seed)
    for m in tr.milestones:
        print(f"  gen {m.episode:>5} (fase {m.phase}): {m.text}")
    print(f"\n{resumen_fase1(tr)}\n  ({time.perf_counter() - t0:.0f} s)")


def exp_solo_equi() -> None:
    """Le quitamos al agente la elección de familia: solo nodos equiespaciados."""
    base = T.GAMMAS
    print("[A] puede elegir familia (original)")
    print("    " + resumen_fase1(entrenar(2000)))
    try:
        T.GAMMAS = (0.0,)
        print("[B] solo equiespaciados (γ = 0)")
        print("    " + resumen_fase1(entrenar(2000)))
    finally:
        T.GAMMAS = base
    print("Pregunta: ¿en qué tipo de funciones falla [B]? Corre `familias` si no lo has hecho.")


def exp_sin_runge() -> None:
    """Si nunca ve funciones con una singularidad cerca, ¿sigue prefiriendo Chebyshev?"""
    original = E.make_problem
    import random

    def solo_enteras(rng: random.Random) -> E.Problem:
        while True:
            p = original(rng)
            if tipo(p.name) == "entera":
                return p

    print("[A] mezcla original (Runge, polos, raíces y funciones enteras)")
    print("    " + resumen_fase1(entrenar(2000)))
    try:
        E.make_problem = solo_enteras
        print("[B] solo funciones enteras: e^(kx) y sin(kx + c)")
        print("    " + resumen_fase1(entrenar(2000)))
    finally:
        E.make_problem = original
    print("Pregunta: ¿cuánto se separan ahora los Q(γ)? ¿Qué aprendió el agente sobre Chebyshev y qué no?")


def exp_recompensa(mal: float | None, nodo: float | None, bien: float | None) -> None:
    base = (E.REWARD_WRONG, E.REWARD_NODE, E.REWARD_RIGHT)

    def con(m: float, n: float, b: float, rotulo: str) -> None:
        E.REWARD_WRONG, E.REWARD_NODE, E.REWARD_RIGHT = m, n, b
        print(f"{rotulo} entregar mal = {m:+g}, cada nodo = {n:+g}, entregar bien = {b:+g}")
        print("    " + resumen_fase1(entrenar(2000)))

    try:
        if mal is None and nodo is None and bien is None:
            con(*base, "[A] original:")
            con(0.0, base[1], base[2], "[B] entregar mal no cuesta:")
            con(base[0], 0.0, base[2], "[C] cada nodo es gratis:")
        else:
            con(base[0] if mal is None else mal, base[1] if nodo is None else nodo, base[2] if bien is None else bien, "[tu recompensa]")
    finally:
        E.REWARD_WRONG, E.REWARD_NODE, E.REWARD_RIGHT = base


# ---------------------------------------------------------------------------
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
    console.print(f"[dim]hipótesis: {', '.join(sorted(hojas))} · pasos necesarios: {KB.MIN_PROOF_LENGTH} · "
                  f"distractores: {sum(1 for x in KB.STEPS if x.distractor)}[/]\n")


def exp_grafo() -> None:
    """Quitar FIX («x no es un nodo, luego w(x) ≠ 0») de las dependencias de AUX.

    AUX divide por w(x). Sin FIX el verificador acepta una demostración que nunca dice que
    w(x) ≠ 0: si x fuera un nodo, la función auxiliar ni siquiera estaría definida."""
    i = KB.KEY_TO_INDEX["AUX"]
    original = KB.STEPS[i]
    req, minlen = KB.REQUIRED_KEYS, KB.MIN_PROOF_LENGTH
    print("=== el grafo original ===")
    dibujar_grafo()
    try:
        KB.STEPS[i] = replace(original, deps=("H1", "EXIST"))
        KB.DEPS_MASK[i] = KB.deps_mask(KB.STEPS[i])
        KB.REQUIRED_KEYS = frozenset(KB._closure("QED", set()))
        KB.MIN_PROOF_LENGTH = len(KB.REQUIRED_KEYS)
        print("=== el grafo con la dependencia quitada ===")
        dibujar_grafo(resaltar="AUX", falta="FIX")
        print("=== una demostración ante el verificador (¿aparece FIX?) ===")
        orden = ["H1", "H2", "EXIST", "DEFW", "AUX", "ZEROS", "DERIV", "ROLLE", "FORMULA", "BOUND", "CHEB", "QED"]
        env = ProofEnv()
        env.reset()
        for k in orden:
            _, _, _, info = env.step(KB.KEY_TO_INDEX[k])
            print(f"  {k:8s} {info}")
        print(f"¿el verificador la acepta? {env.finished}   ({len(orden)} pasos; la original necesita {minlen})")
        print("Pregunta: AUX es g(t) = f(t) − pₙ(t) − [f(x) − pₙ(x)]·w(t)/w(x). ¿Qué pasa si x es uno de los nodos?")
    finally:
        KB.STEPS[i] = original
        KB.DEPS_MASK[i] = KB.deps_mask(original)
        KB.REQUIRED_KEYS, KB.MIN_PROOF_LENGTH = req, minlen


# ---------------------------------------------------------------------------
# mi_grafo: el primer teorema de la biblioteca escrito por ustedes
def cargar_grafo(ruta: Path):
    spec = importlib.util.spec_from_file_location("mi_grafo", ruta)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


def revisar_grafo(pasos: list) -> list[str]:
    """Problemas de estructura: claves repetidas, deps inexistentes, ciclos, ∎ ausente."""
    errores = []
    claves = [p.key for p in pasos]
    if len(set(claves)) != len(claves):
        errores.append("hay dos pasos con la misma clave")
    existentes = set(claves)
    for p in pasos:
        for d in p.deps:
            if d not in existentes:
                errores.append(f"{p.key} depende de {d}, que no existe")
        if p.distractor and p.deps:
            errores.append(f"{p.key} es distractor y tiene deps: los distractores nunca son válidos, sobran")
    qeds = [p for p in pasos if p.is_qed]
    if len(qeds) != 1:
        errores.append(f"debe haber exactamente un paso con is_qed=True (hay {len(qeds)})")
    elif not qeds[0].deps:
        errores.append(f"{qeds[0].key} (∎) no depende de nada: el verificador lo aceptaría como primera línea. "
                       "Escribe en sus deps los pasos que usa la conclusión.")
    por_clave = {p.key: p for p in pasos}
    estado: dict[str, int] = {}

    def visitar(k: str, camino: list[str]) -> None:
        if estado.get(k) == 1:
            errores.append("ciclo: " + " → ".join(camino + [k]))
            return
        if estado.get(k) == 2 or k not in por_clave:
            return
        estado[k] = 1
        for d in por_clave[k].deps:
            visitar(d, camino + [k])
        estado[k] = 2

    for k in claves:
        visitar(k, [])
    return errores


def exp_mi_grafo(ruta: Path) -> None:
    from rich.console import Console
    from rich.tree import Tree

    if not ruta.exists():
        print(f"No encuentro {ruta}.")
        return
    mod = cargar_grafo(ruta)
    pasos = list(mod.STEPS)
    print(f"Teorema: {getattr(mod, 'TEOREMA', '(sin enunciado: añade TEOREMA = \"...\")')}")
    print(f"Autores: {getattr(mod, 'AUTORES', '(añade AUTORES = \"...\")')}\n")
    errores = revisar_grafo(pasos)
    if errores:
        print("El grafo tiene problemas de estructura:")
        for e in errores:
            print("  ✘ " + e)
        return
    por_clave = {p.key: p for p in pasos}
    qed = next(p for p in pasos if p.is_qed)

    def cierre(k: str, acc: set[str]) -> set[str]:
        acc.add(k)
        for d in por_clave[k].deps:
            cierre(d, acc)
        return acc

    necesarios = cierre(qed.key, set())
    sobrantes = [p.key for p in pasos if p.key not in necesarios and not p.distractor]
    hipotesis = sorted(k for k in necesarios if not por_clave[k].deps)
    console = Console()
    arbol = Tree(f"[bold green]∎ {qed.key}[/]")
    visto: set[str] = set()

    def rama(k: str, t: Tree) -> None:
        if k in visto:
            t.add(f"[dim]{k} (ya dibujado)[/]")
            return
        visto.add(k)
        h = t.add(f"[bold]{k}[/] [dim]{por_clave[k].title}[/]")
        for d in por_clave[k].deps:
            rama(d, h)

    for d in qed.deps:
        rama(d, arbol)
    console.print(arbol)
    print(f"hipótesis: {', '.join(hipotesis)} · pasos necesarios: {len(necesarios)} · "
          f"distractores: {sum(p.distractor for p in pasos)}" + (f" · pasos que ∎ no usa: {', '.join(sobrantes)}" if sobrantes else ""))
    if len(hipotesis) == len(necesarios) - 1:
        print("✘ ∎ depende directamente de todo y nada depende de nada más: el verificador acepta cualquier orden.")
        print("  Un grafo así no dice qué se usa para qué. Escribe las deps de cada paso.")

    # el agente, el mismo de la fase 2, sobre su grafo
    idx = {p.key: i for i, p in enumerate(pasos)}
    deps_mask = [sum(1 << idx[d] for d in p.deps) for p in pasos]
    q_i = idx[qed.key]
    ag = QAgent(len(pasos), alpha=0.3, gamma=0.97, eps_start=1.0, eps_decay=0.995, eps_end=0.02, ucb_c=1.0, seed=1)

    def valido(a: int, mask: int) -> bool:
        return not pasos[a].distractor and not (mask >> a) & 1 and (deps_mask[a] & mask) == deps_mask[a]

    def episodio(aprender: bool) -> tuple[list[tuple[str, bool]], bool]:
        mask, intentos = 0, []
        for _ in range(3 * len(pasos)):
            a = ag.act(mask) if aprender else ag.greedy(mask)
            ok = valido(a, mask)
            intentos.append((pasos[a].key, ok))
            nuevo = mask | (1 << a) if ok else mask
            fin = ok and a == q_i
            r = (-0.5 + (20 if fin else 0)) if ok else -2.0
            if aprender:
                ag.learn(mask, a, r, nuevo, fin)
            mask = nuevo
            if fin:
                return intentos, True
        return intentos, False

    primera = None
    for ep in range(1, 1501):
        _, fin = episodio(True)
        ag.decay()
        if fin and primera is None:
            primera = ep
    intentos, fin = episodio(False)
    print(f"\nEl agente: primera demostración completa en la generación {primera}.")
    print("Demostración de la política voraz: " + " → ".join(k + ("" if ok else "✘") for k, ok in intentos)
          + ("   ∎" if fin else "   (no llegó a ∎)"))
    print("Recuerda: el agente solo encuentra un orden que respete tus flechas. Si una flecha falta, el hueco es tuyo.")


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Experimentos del laboratorio 3 sobre el agente de interpolación.")
    ap.add_argument("experimento", nargs="?", default="hitos",
                    choices=["basales", "familias", "hitos", "solo_equi", "sin_runge", "recompensa", "dibujar", "grafo", "mi_grafo"])
    ap.add_argument("archivo", nargs="?", default=None, help="para mi_grafo: ruta del archivo (por defecto mi_grafo.py junto a este script)")
    ap.add_argument("--nodos", default=None, help="para basales: nodos separados por comas, p. ej. --nodos=-1,0,1")
    ap.add_argument("--t", default=None, help="para basales: puntos t separados por comas, p. ej. --t=-0.5,2")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--entregar-mal", type=float, default=None)
    ap.add_argument("--nodo", type=float, default=None)
    ap.add_argument("--entregar-bien", type=float, default=None)
    args = ap.parse_args()
    if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")  # consolas de Windows antiguas
        except Exception:
            pass
    if args.experimento == "basales":
        exp_basales(args.nodos, args.t)
    elif args.experimento == "familias":
        exp_familias()
    elif args.experimento == "hitos":
        exp_hitos(args.seed)
    elif args.experimento == "solo_equi":
        exp_solo_equi()
    elif args.experimento == "sin_runge":
        exp_sin_runge()
    elif args.experimento == "recompensa":
        exp_recompensa(args.entregar_mal, args.nodo, args.entregar_bien)
    elif args.experimento == "dibujar":
        dibujar_grafo()
    elif args.experimento == "grafo":
        exp_grafo()
    elif args.experimento == "mi_grafo":
        exp_mi_grafo(Path(args.archivo) if args.archivo else AQUI / "mi_grafo.py")
