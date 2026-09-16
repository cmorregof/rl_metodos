"""Referencias τ_ref(instancia): lo mejor que conocemos para cada (n, μ, criterio).

La recompensa del entorno es log(τ_ref / τ): 0 significa «igualó lo conocido» y
positivo «hay que mirar». Por eso la referencia tiene que ser lo más fuerte
posible, y cada entrada guarda de dónde sale:

* `tabla`      – óptimos numéricos publicados (Das Gupta–Van Parys–Ryu, vía Grimmer–Shu–Wang; convexo, f(xₙ) − f*);
* `ce`         – lo mejor que encontró nuestra entropía cruzada + Nelder–Mead (`build_refs`);
* `constante`  – paso constante h = 1 (referencia débil: solo cuando no hay nada mejor; se avisa).

PENDIENTE: implementar las sucesiones compuestas OBS-F / OBS-G de Grimmer–Shu–Wang
(arXiv:2410.16249), que igualan o baten la tabla para todo n, y usarlas como referencia
analítica. Hasta entonces la referencia convexa es min(tabla, ce).
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from .pep import gd_worst_case

# Óptimos numéricos a horizonte fijo (Das Gupta et al.; tabla de Grimmer–Shu–Wang, /2 por convención)
# y el valor de n = 6 que reproduce nuestra búsqueda (0.25 % por debajo de la tabla; a contrastar con OBS-F).
REFERENCE_FIXED: dict[int, float] = {
    1: 0.125000, 2: 0.065945, 3: 0.042895, 4: 0.031170, 5: 0.024070,
    6: 0.020049, 7: 0.016330, 8: 0.014055, 9: 0.012280, 10: 0.010620,
}

REFS_PATH = Path(__file__).resolve().parent / "refs.json"


@dataclass(frozen=True)
class Instance:
    """Una instancia del entorno: horizonte n, fuerte convexidad μ (L = 1), criterio y si se permiten pasos negativos.
    La referencia no depende de `allow_negative`: la pregunta es precisamente si los pasos negativos la baten."""

    n: int
    mu: float = 0.0
    objective: str = "fval"  # fval | gradnorm
    allow_negative: bool = False

    def ref_key(self) -> str:
        return f"{self.objective}|mu={self.mu:g}|n={self.n}"

    def label(self) -> str:
        return self.ref_key() + ("|neg" if self.allow_negative else "")


def family(ns=(1, 2, 3, 4, 5, 6, 7, 8, 10, 12), mus=(0.0, 0.01, 0.1), objectives=("fval", "gradnorm"), allow_negative=False) -> list[Instance]:
    return [Instance(n, mu, obj, allow_negative) for obj in objectives for mu in mus for n in ns]


FAMILY_TRAIN = family()
FAMILY_TEST = family(ns=(16, 24), mus=(0.0, 0.03, 0.1))


def load_refs(path: Path = REFS_PATH) -> dict[str, dict]:
    if Path(path).exists():
        return json.loads(Path(path).read_text(encoding="utf-8"))
    return {}


def save_refs(refs: dict[str, dict], path: Path = REFS_PATH) -> None:
    Path(path).write_text(json.dumps(refs, indent=1, ensure_ascii=False), encoding="utf-8")


_MEM: dict[str, dict] = {}


def reference(inst: Instance, refs: dict[str, dict] | None = None) -> tuple[float, str]:
    """(τ_ref, fuente). Orden: refs.json (ce/tabla) → tabla publicada → paso constante (débil)."""
    refs = load_refs() if refs is None else refs
    k = inst.ref_key()
    best, src = float("inf"), ""
    if k in refs:
        best, src = refs[k]["tau"], refs[k]["source"]
    if inst.objective == "fval" and inst.mu == 0 and inst.n in REFERENCE_FIXED and REFERENCE_FIXED[inst.n] < best:
        best, src = REFERENCE_FIXED[inst.n], "tabla"
    if best == float("inf"):
        if k not in _MEM:
            _MEM[k] = {"tau": gd_worst_case([1.0] * inst.n, mu=inst.mu, objective=inst.objective).value, "source": "constante"}
        best, src = _MEM[k]["tau"], _MEM[k]["source"]
    return best, src


def _build_one(args) -> tuple[str, dict]:
    from .search import cross_entropy

    inst, restarts, generations, population = args
    best = None
    t0 = time.perf_counter()
    for seed in range(restarts):
        r = cross_entropy(inst.n, generations=generations, population=population, init_std=0.8, seed=seed, objective=inst.objective, mu=inst.mu)
        if best is None or r.value < best.value:
            best = r
    return inst.ref_key(), {"tau": float(best.value), "h": [float(x) for x in best.h], "source": "ce",
                            "restarts": restarts, "generations": generations, "population": population,
                            "seconds": round(time.perf_counter() - t0, 1)}


def build_refs(instances: list[Instance], restarts: int = 3, generations: int = 40, population: int = 50,
               workers: int | None = None, path: Path = REFS_PATH, verbose: bool = True) -> dict[str, dict]:
    """Calcula (o mejora) τ_ref por entropía cruzada para cada instancia y lo guarda en refs.json.
    Solo reemplaza una entrada existente si la nueva es mejor."""
    refs = load_refs(path)
    todo = {inst.ref_key(): inst for inst in instances}
    jobs = [(inst, restarts, generations, population) for inst in todo.values()]
    with Pool(workers) as pool:
        for key, entry in pool.imap_unordered(_build_one, jobs):
            old = refs.get(key)
            table = REFERENCE_FIXED.get(int(key.split("n=")[1])) if key.startswith("fval|mu=0|") else None
            if old is None or entry["tau"] < old["tau"]:
                refs[key] = entry
            if table is not None and table < refs[key]["tau"]:
                refs[key] = {"tau": table, "h": [], "source": "tabla"}
            if verbose:
                print(f"{key:24s} τ_ref = {refs[key]['tau']:.6f} ({refs[key]['source']}, ce={entry['tau']:.6f}, {entry['seconds']}s)", flush=True)
            save_refs(refs, path)
    return refs
