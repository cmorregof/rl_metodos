"""Punto de entrada: `python -m newtonrl` o `newtonrl`."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown

from .report import build_report, save_run
from .trainer import Config, Trainer
from .tui import build_layout


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        prog="newtonrl",
        description="Mira en vivo cómo un agente de RL aprende Newton con salvaguardas y demuestra su convergencia cuadrática.",
    )
    ap.add_argument("--episodes1", type=int, default=3000, help="generaciones de la fase 1 (Newton)")
    ap.add_argument("--episodes2", type=int, default=3000, help="generaciones de la fase 2 (demostración)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tol", type=float, default=1e-9, help="tolerancia sobre |x − r|")
    ap.add_argument("--speed", type=float, default=1.0, help="multiplicador de velocidad de la animación")
    ap.add_argument("--fast", action="store_true", help="sin pausas: solo redibuja (útil en CI)")
    ap.add_argument("--no-tui", action="store_true", help="sin interfaz: entrena y escribe el informe")
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--out", type=Path, default=Path("runs"), help="carpeta donde guardar el informe")
    return ap.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    cfg = Config(episodes_newton=args.episodes1, episodes_proof=args.episodes2, seed=args.seed, tol=args.tol)
    tr = Trainer(cfg)
    console = Console()

    if args.no_tui:
        for _ in tr.run():
            pass
    else:
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
