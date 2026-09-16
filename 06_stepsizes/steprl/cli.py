"""Punto de entrada: `python -m steprl <subcomando>`.

  verify    comprueba el verificador contra PEPit, Drori–Teboulle y el silver schedule
  search    óptimos a horizonte fijo por entropía cruzada (capa numérica)
  evolve    bucle LLM → PEP (capa simbólica); --provider mock|openai|anthropic
  models    lista los modelos que acepta la clave del proveedor (--provider openai|anthropic)
  certify   certificado racional exacto de una sucesión (--h 1.5,1.5 | --program f.py --n 8) [--mu, --objective]
  check     verifica un certificado guardado (cert.json) solo con aritmética exacta
  refs      construye/mejora las referencias τ_ref por entropía cruzada (refs.json)
  score     puntúa un programa en una instancia del entorno (--program f.py --n 8 [--mu 0.1] [--negative])
  bench     tiempo por SDP y por certificado según n

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
        anytime_N=tuple(int(x) for x in str(args.anytime_n).split(",") if x.strip()),
        target_p=args.target_p,
        out_dir=out,
        provider_name=args.provider,
        model_name=args.model or "",
    )
    print(f"\nmejor programa ({args.objective}):\n{pool[0].code}\n→ {pool[0].summary()}\nguardado en {out}")
    return 0


def _parse_h(args) -> list[float]:
    if args.h:
        return [float(x) for x in args.h.split(",") if x.strip()]
    from .env import run_program

    return run_program(Path(args.program).read_text(encoding="utf-8"), args.n)


def cmd_certify(args) -> int:
    from .certify import certify

    h = _parse_h(args)
    c = certify(h, mu=args.mu, objective=args.objective)
    print(c.statement() if c.ok else f"NO certificado: {c.message}")
    if args.out:
        c.save(args.out)
        print(f"guardado en {args.out}")
    return 0 if c.ok else 1


def cmd_check(args) -> int:
    from .certify import Certificate, verify

    ok, msg = verify(Certificate.load(args.cert))
    print(msg)
    return 0 if ok else 1


def cmd_refs(args) -> int:
    from .refs import build_refs, family

    ns = []
    for part in args.n.split(","):
        a, _, b = part.partition("-")
        ns.extend(range(int(a), int(b or a) + 1))
    insts = family(ns=tuple(ns), mus=tuple(float(x) for x in args.mu.split(",")), objectives=tuple(args.objective.split(",")))
    print(f"{len(insts)} instancias, {args.restarts} reinicios × {args.generations} generaciones × {args.population} individuos")
    build_refs(insts, restarts=args.restarts, generations=args.generations, population=args.population, workers=args.workers)
    return 0


def cmd_score(args) -> int:
    from .env import score
    from .refs import Instance

    sc = score(Path(args.program).read_text(encoding="utf-8"), Instance(args.n, args.mu, args.objective, args.negative))
    print(sc.summary())
    if sc.ok:
        print("h =", np.round(sc.h, 6).tolist())
    return 0


def cmd_bench(args) -> int:
    import time

    from .certify import certify
    from .pep import gd_worst_case

    for n in (int(x) for x in args.n.split(",")):
        h = [1.5] * n
        t = time.perf_counter(); gd_worst_case(h); t1 = time.perf_counter() - t
        t = time.perf_counter(); gd_worst_case(h, mu=0.1, objective="gradnorm"); t2 = time.perf_counter() - t
        t = time.perf_counter(); c = certify(h); t3 = time.perf_counter() - t
        print(f"n={n:3d}: SDP {t1 * 1000:6.0f} ms   SDP μ>0 gradnorm {t2 * 1000:6.0f} ms   certificado exacto {t3:5.1f} s ({'ok' if c.ok else c.message})")
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
    e.add_argument("--anytime-n", default="31,63", help="horizontes anytime separados por comas; la puntuación es el peor")
    e.add_argument("--target-p", type=float, default=1.12, help="exponente objetivo de la garantía anytime: se minimiza C(p*) = máx_t τ_t·t^p*")
    e.add_argument("--out", type=Path, default=Path("runs"))
    m = sub.add_parser("models")
    m.add_argument("--provider", choices=["openai", "anthropic"], default="openai")
    c = sub.add_parser("certify")
    c.add_argument("--h", default=None, help="pasos separados por comas")
    c.add_argument("--program", type=Path, default=None, help="archivo con schedule(n)")
    c.add_argument("--n", type=int, default=8)
    c.add_argument("--mu", type=float, default=0.0)
    c.add_argument("--objective", choices=["fval", "gradnorm"], default="fval")
    c.add_argument("--out", type=Path, default=None)
    k = sub.add_parser("check")
    k.add_argument("cert", type=Path)
    r = sub.add_parser("refs")
    r.add_argument("--n", default="1-8,10,12", help="p. ej. 1-8,10,12")
    r.add_argument("--mu", default="0,0.01,0.1")
    r.add_argument("--objective", default="fval,gradnorm")
    r.add_argument("--restarts", type=int, default=3)
    r.add_argument("--generations", type=int, default=40)
    r.add_argument("--population", type=int, default=50)
    r.add_argument("--workers", type=int, default=None)
    sc = sub.add_parser("score")
    sc.add_argument("--program", type=Path, required=True)
    sc.add_argument("--n", type=int, default=8)
    sc.add_argument("--mu", type=float, default=0.0)
    sc.add_argument("--objective", choices=["fval", "gradnorm"], default="fval")
    sc.add_argument("--negative", action="store_true", help="permitir pasos negativos")
    b = sub.add_parser("bench")
    b.add_argument("--n", default="4,8,12,16,24")
    args = ap.parse_args(argv)
    return {"verify": cmd_verify, "search": cmd_search, "evolve": cmd_evolve, "models": cmd_models, "certify": cmd_certify,
            "check": cmd_check, "refs": cmd_refs, "score": cmd_score, "bench": cmd_bench}[args.cmd](args)
