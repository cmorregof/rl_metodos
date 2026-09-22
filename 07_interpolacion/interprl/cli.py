"""Punto de entrada: `python -m interprl` o `interprl`.

    python -m interprl                     # fases 1 y 2 en vivo
    python -m interprl --no-tui            # sin interfaz
    python -m interprl search --n 8        # fase 3: nodos con Λ mínima
    python -m interprl verify runs/.../busqueda_n8.json
    python -m interprl lebesgue --nodes=-1,-0.5,0,0.5,1
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown

from .interp import lebesgue_constant
from .report import build_report, save_run
from .search import run_search, verify_file
from .trainer import Config, Trainer


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(prog="interprl", description="Un agente de RL aprende a interpolar (dónde y cuántos nodos) y demuestra el teorema del error.")
    ap.add_argument("--episodes1", type=int, default=3000, help="generaciones de la fase 1 (interpolación)")
    ap.add_argument("--episodes2", type=int, default=3000, help="generaciones de la fase 2 (demostración)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tol", type=float, default=1e-5, help="tolerancia relativa del error uniforme")
    ap.add_argument("--speed", type=float, default=1.0)
    ap.add_argument("--fast", action="store_true")
    ap.add_argument("--no-tui", action="store_true")
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--out", type=Path, default=Path("runs"))
    sub = ap.add_subparsers(dest="cmd")
    s = sub.add_parser("search", help="fase 3: buscar nodos con constante de Lebesgue mínima")
    s.add_argument("--n", type=int, default=8, help="grado (n + 1 nodos)")
    s.add_argument("--iters", type=int, default=40)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--out", type=Path, default=Path("runs"))
    v = sub.add_parser("verify", help="recalcular Λ de una búsqueda guardada con otra malla")
    v.add_argument("file", type=Path)
    l = sub.add_parser("lebesgue", help="constante de Lebesgue de unos nodos dados")
    l.add_argument("--nodes", type=str, required=True, help="lista separada por comas")
    return ap.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    console = Console()

    if args.cmd == "lebesgue":
        nodes = [float(t) for t in args.nodes.split(",")]
        lam, t = lebesgue_constant(nodes)
        console.print(f"Λ = {lam:.10f}  (máximo en t = {t:+.6f})")
        return 0
    if args.cmd == "verify":
        ok, lam, claimed = verify_file(args.file)
        console.print(f"afirmado Λ = {claimed:.12f}  recalculado Λ = {lam:.12f}  {'✔ coincide' if ok else '✘ NO coincide'}")
        return 0 if ok else 1
    if args.cmd == "search":
        stamp = time.strftime("%Y%m%d-%H%M%S")
        out = args.out / stamp / f"busqueda_n{args.n}.json"
        t0 = time.perf_counter()
        r = run_search(args.n, seed=args.seed, iters=args.iters, out=out)
        refs = r["referencias"]
        console.print(f"n = {args.n} · {time.perf_counter() - t0:.1f} s")
        for k, c in refs.items():
            console.print(f"  {k:<22} Λ = {c['lebesgue']:.8f}")
        console.print(f"  [bold]{'encontrado':<22} Λ = {r['mejor']['lebesgue']:.8f}[/]  ratio = {r['ratio_vs_mejor_referencia']:.6f}  "
                      + ("[bold green]bate la referencia ✔[/]" if r["bate_referencia"] else "[yellow]no bate la referencia[/]"))
        console.print("  nodos: " + ", ".join(f"{x:+.6f}" for x in r["mejor"]["nodes"]))
        console.print(f"  guardado en {out}  ·  comprobar con: python -m interprl verify {out}")
        return 0

    cfg = Config(episodes_interp=args.episodes1, episodes_proof=args.episodes2, seed=args.seed, tol=args.tol)
    tr = Trainer(cfg)
    if args.no_tui:
        for _ in tr.run():
            pass
    else:
        from .tui import build_layout

        iter_dwell = 0.0 if args.fast else 0.045 / args.speed
        ep_dwell = 0.0 if args.fast else 0.004 / args.speed
        min_frame = 1.0 / args.fps
        last_render = 0.0
        n_log = len(tr.log)
        with Live(build_layout(tr, console.size.height), console=console, screen=True, auto_refresh=False) as live:
            for kind in tr.run():
                now = time.perf_counter()
                new_milestone = len(tr.log) != n_log
                if now - last_render >= min_frame or new_milestone:
                    live.update(build_layout(tr, console.size.height), refresh=True)
                    last_render = now
                    n_log = len(tr.log)
                dwell = iter_dwell if kind == "iter" else ep_dwell
                if dwell:
                    time.sleep(dwell)
            live.update(build_layout(tr, console.size.height), refresh=True)
            if not args.fast:
                time.sleep(2.0)

    run_dir = save_run(tr, args.out)
    console.print(Markdown(build_report(tr)))
    console.print(f"\n[bold green]Informe guardado en[/] {run_dir}/informe.md  ·  historia.json  ·  demostracion.md")
    return 0
