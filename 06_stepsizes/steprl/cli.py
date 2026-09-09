"""Punto de entrada: `python -m steprl <subcomando>`.

  verify    comprueba el verificador contra PEPit, Drori–Teboulle y el silver schedule
  search    óptimos a horizonte fijo por entropía cruzada (capa numérica)
  evolve    bucle LLM → PEP (capa simbólica); --provider mock|openai|anthropic
  models    lista los modelos que acepta la clave del proveedor (--provider openai|anthropic)

Las claves se leen de OPENAI_API_KEY / ANTHROPIC_API_KEY o de un fichero .env en 06_stepsizes/.
"""

from __future__ import annotations

import argparse
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")


def cmd_verify(args) -> int:
    from .pep import constant_bound, fit_exponent, gd_worst_case, prefix_worst_cases, silver_bound, silver_schedule

    print("paso constante h=1 vs 1/(4n+2):")
    for n in (1, 2, 5, 10):
        print(f"  n={n:2d}: PEP {gd_worst_case([1.0] * n).value:.6f}  cota {constant_bound(n):.6f}")
    print("silver vs garantía 1/(1+√(4ρ²ᵏ−3)):")
    ns, vs = [], []
    for k in (1, 2, 3, 4):
        h = silver_schedule(k)
        v = gd_worst_case(h).value
        ns.append(len(h)); vs.append(v)
        print(f"  k={k} n={len(h):2d}: PEP {v:.6f}  cota {silver_bound(k):.6f}  {'✔' if v <= silver_bound(k) + 1e-6 else '✘'}")
    print(f"  exponente ajustado (n=3,7,15): {fit_exponent(ns[1:], vs[1:]):.4f}  (log₂(1+√2) = {np.log2(1 + np.sqrt(2)):.4f})")
    print("prefijos del silver n=15 (anytime):", np.round(prefix_worst_cases(silver_schedule(4)), 4).tolist())
    return 0


def cmd_search(args) -> int:
    from .evolve import REFERENCE_FIXED
    from .search import cross_entropy

    for n in range(args.n_min, args.n_max + 1):
        best = None
        for seed in range(args.restarts):
            r = cross_entropy(n, generations=args.generations, population=args.population, init_std=0.8, seed=seed)
            if best is None or r.value < best.value:
                best = r
        ref = REFERENCE_FIXED.get(n)
        extra = f"  referencia {ref:.6f}  ratio {best.value / ref:.4f}" if ref else ""
        print(f"n={n}: τ = {best.value:.6f}{extra}  h = {np.round(best.h, 4).tolist()}", flush=True)
    return 0


def cmd_evolve(args) -> int:
    from .evolve import run_evolution
    from .providers import make_provider

    kw = {}
    if args.provider == "anthropic" and args.effort:
        kw["effort"] = args.effort
    if args.provider == "openai" and args.effort:
        kw["reasoning_effort"] = args.effort
    provider = make_provider(args.provider, args.model, **kw)
    out = args.out / datetime.now().strftime("%Y%m%d-%H%M%S")
    pool = run_evolution(
        provider,
        generations=args.generations,
        objective=args.objective,
        pool_size=args.pool,
        anytime_N=args.anytime_n,
        out_dir=out,
        provider_name=args.provider,
        model_name=args.model or "",
    )
    print(f"\nmejor programa ({args.objective}):\n{pool[0].code}\n→ {pool[0].summary()}\nguardado en {out}")
    return 0


def cmd_models(args) -> int:
    from .providers import list_models

    for m in list_models(args.provider):
        print(m)
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="steprl", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("verify")
    s = sub.add_parser("search")
    s.add_argument("--n-min", type=int, default=1)
    s.add_argument("--n-max", type=int, default=6)
    s.add_argument("--generations", type=int, default=50)
    s.add_argument("--population", type=int, default=60)
    s.add_argument("--restarts", type=int, default=4)
    e = sub.add_parser("evolve")
    e.add_argument("--provider", choices=["mock", "openai", "anthropic"], default="mock")
    e.add_argument("--model", default=None, help="identificador del modelo (obligatorio con openai)")
    e.add_argument("--effort", default=None, help="esfuerzo de razonamiento (anthropic: low…max; openai: reasoning_effort)")
    e.add_argument("--objective", choices=["anytime", "fixed"], default="anytime")
    e.add_argument("--generations", type=int, default=10)
    e.add_argument("--pool", type=int, default=4)
    e.add_argument("--anytime-n", type=int, default=31)
    e.add_argument("--out", type=Path, default=Path("runs"))
    m = sub.add_parser("models")
    m.add_argument("--provider", choices=["openai", "anthropic"], default="openai")
    args = ap.parse_args(argv)
    return {"verify": cmd_verify, "search": cmd_search, "evolve": cmd_evolve, "models": cmd_models}[args.cmd](args)
