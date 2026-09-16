"""Entorno de recompensa: el PEP como juez de programas de pasos, listo para RL.

Una **instancia** es (n, μ, criterio, ¿pasos negativos?) (`refs.Instance`). Una
**acción** es un programa Python que define `schedule(n) -> list[float]`. El
entorno ejecuta el programa en un sandbox con límite de tiempo, valida la
sucesión, certifica su peor caso con el PEP y devuelve

    recompensa = max(−2, log(τ_ref / τ))        (0 = igualó lo conocido; > 0 = mejora)
    recompensa = −3                              si el programa no es válido.

La escala es comparable entre instancias porque está referida a lo mejor
conocido para cada una (`refs.reference`).

Contra la trampa del solver: cualquier τ que *afirme* una mejora (recompensa
> `GATE`) se recalcula con tolerancias finas y pasa por `certify.certify`.
Si el certificado exacto no valida, la mejora no cuenta (recompensa ≤ 0) y
queda marcada; si valida, la recompensa se calcula con la τ certificada (que
es ≥ la real). Así el agente no puede ganar explotando la tolerancia del SDP.

`score_batch` reparte el trabajo entre procesos (los SDP van en CPU, uno por
núcleo) y cachea τ por sucesión: bajo RL muchos programas producen la misma
sucesión, y así se resuelve una sola vez.
"""

from __future__ import annotations

import hashlib
import json
import math
import signal
import time
from dataclasses import dataclass, field
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from .certify import Certificate, certify
from .pep import TIGHT_CLARABEL, gd_worst_case
from .refs import Instance, load_refs, reference

INVALID_REWARD = -3.0
MIN_REWARD = -2.0
GATE = 1e-5  # log(τ_ref/τ) por encima de esto = «afirma una mejora» → tolerancias finas + certificado exacto
H_MAX = 50.0
H_MIN_NEG = -50.0
H_POS_MIN = 1e-9

# --- sandbox ----------------------------------------------------------------

import builtins as _bi

_ALLOWED_MODULES = {"math", "numpy", "np", "itertools", "functools", "fractions", "cmath"}


def _safe_import(name, globals=None, locals=None, fromlist=(), level=0):
    if level != 0 or name.split(".")[0] not in _ALLOWED_MODULES:
        raise ImportError(f"módulo no permitido en el sandbox: {name}")
    return _bi.__import__(name, globals, locals, fromlist, level)


_SAFE_BUILTINS = {k: getattr(_bi, k) for k in (
    "abs", "all", "any", "bool", "dict", "divmod", "enumerate", "filter", "float", "int", "isinstance", "len", "list",
    "map", "max", "min", "pow", "range", "reversed", "round", "set", "sorted", "sum", "tuple", "zip", "str", "ValueError",
    "ZeroDivisionError", "Exception", "True", "False", "None",
)}
_SAFE_BUILTINS["__import__"] = _safe_import


class ProgramTimeout(Exception):
    pass


def _alarm(signum, frame):
    raise ProgramTimeout("el programa excedió el tiempo límite")


def run_program(code: str, n: int, timeout_s: float = 2.0) -> list[float]:
    """Ejecuta `schedule(n)` del programa en un espacio de nombres restringido (solo math y numpy)."""
    ns: dict = {"__builtins__": _SAFE_BUILTINS, "math": math, "np": np, "numpy": np}
    old = signal.signal(signal.SIGALRM, _alarm)
    signal.setitimer(signal.ITIMER_REAL, timeout_s)
    try:
        exec(compile(code, "<candidate>", "exec"), ns, ns)
        fn = ns.get("schedule")
        if fn is None:
            raise ValueError("el programa no define schedule(n)")
        h = fn(n)
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old)
    return [float(x) for x in h]


def validate(h: list[float], inst: Instance) -> np.ndarray:
    a = np.asarray(h, dtype=float)
    if a.shape != (inst.n,):
        raise ValueError(f"schedule({inst.n}) devolvió {a.shape[0] if a.ndim == 1 else '?'} pasos")
    if not np.all(np.isfinite(a)):
        raise ValueError("pasos no finitos")
    lo = H_MIN_NEG if inst.allow_negative else H_POS_MIN
    if np.any(a < lo) or np.any(a > H_MAX):
        raise ValueError(f"pasos fuera de [{lo:g}, {H_MAX:g}]: {np.round(a, 4).tolist()}")
    return a


# --- puntuación --------------------------------------------------------------


@dataclass
class Score:
    instance: Instance
    code_hash: str
    h: list[float] = field(default_factory=list)
    tau: float = float("nan")
    tau_ref: float = float("nan")
    ref_source: str = ""
    reward: float = INVALID_REWARD
    error: str = ""
    improved: bool = False  # afirma mejora sobre τ_ref (antes de certificar)
    certified: bool | None = None  # None: no hizo falta; True/False: resultado del certificado exacto
    tau_cert: float = float("nan")
    cert_json: dict | None = None
    seconds: float = 0.0

    @property
    def ok(self) -> bool:
        return not self.error

    def summary(self) -> str:
        if not self.ok:
            return f"[{self.instance.label()}] inválido: {self.error}"
        s = f"[{self.instance.label()}] τ = {self.tau:.7f} vs τ_ref = {self.tau_ref:.7f} ({self.ref_source}) → r = {self.reward:+.5f}"
        if self.improved:
            s += f"  MEJORA {'certificada τ=' + format(self.tau_cert, '.8f') if self.certified else 'NO certificada'}"
        return s


def tau_key(h, inst: Instance) -> str:
    return f"{inst.objective}|mu={inst.mu:g}|" + ",".join(f"{x:.12g}" for x in h)


def _reward(tau: float, tau_ref: float) -> float:
    if not (tau > 0):
        return MIN_REWARD
    return max(MIN_REWARD, math.log(tau_ref / tau))


def score(code: str, inst: Instance, refs: dict | None = None, tau_cache: dict | None = None, certify_improvements: bool = True) -> Score:
    """Puntúa un programa en una instancia (secuencial; `score_batch` para muchos)."""
    t0 = time.perf_counter()
    sc = Score(inst, hashlib.sha1(code.encode()).hexdigest()[:10])
    try:
        h = validate(run_program(code, inst.n), inst)
    except Exception as e:
        sc.error = f"{type(e).__name__}: {e}"[:300]
        sc.seconds = time.perf_counter() - t0
        return sc
    sc.h = h.tolist()
    sc.tau_ref, sc.ref_source = reference(inst, refs)
    key = tau_key(sc.h, inst)
    if tau_cache is not None and key in tau_cache:
        sc.tau = tau_cache[key]
    else:
        sc.tau = gd_worst_case(sc.h, mu=inst.mu, objective=inst.objective).value
        if tau_cache is not None:
            tau_cache[key] = sc.tau
    sc.reward = _reward(sc.tau, sc.tau_ref)
    if sc.reward > GATE:
        sc.improved = True
        tight = gd_worst_case(sc.h, mu=inst.mu, objective=inst.objective, solver_opts=TIGHT_CLARABEL).value
        sc.tau = tight
        sc.reward = _reward(tight, sc.tau_ref)
        if certify_improvements and sc.reward > GATE:
            cert = certify(sc.h, mu=inst.mu, objective=inst.objective)
            sc.certified = cert.ok
            if cert.ok:
                sc.tau_cert = cert.tau_float
                sc.reward = _reward(cert.tau_float, sc.tau_ref)  # la τ certificada es ≥ la real: recompensa conservadora
                sc.cert_json = cert.to_json()
            else:
                sc.reward = min(sc.reward, 0.0)  # sin certificado, la mejora no cuenta
        elif sc.reward <= GATE:
            sc.improved = False
    sc.seconds = time.perf_counter() - t0
    return sc


# --- lotes en paralelo --------------------------------------------------------


def _run_h(args):
    code, inst = args
    try:
        return validate(run_program(code, inst.n), inst).tolist(), ""
    except Exception as e:
        return [], f"{type(e).__name__}: {e}"[:300]


def _solve(args):
    key, h, inst = args
    try:
        return key, gd_worst_case(h, mu=inst.mu, objective=inst.objective).value
    except Exception:
        return key, float("nan")


def _code_for(h) -> str:
    return "def schedule(n):\n    return " + repr([float(x) for x in h]) + "\n"


class TauCache:
    """Caché τ por sucesión, persistente en JSON (opcional)."""

    def __init__(self, path: Path | None = None):
        self.path = Path(path) if path else None
        self.d: dict[str, float] = {}
        if self.path and self.path.exists():
            self.d = json.loads(self.path.read_text(encoding="utf-8"))

    def save(self) -> None:
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps(self.d), encoding="utf-8")

    def __contains__(self, k):
        return k in self.d

    def __getitem__(self, k):
        return self.d[k]

    def __setitem__(self, k, v):
        self.d[k] = v

    def __len__(self):
        return len(self.d)


def score_batch(codes: list[str], instances: list[Instance], workers: int | None = None, cache: TauCache | None = None,
                refs: dict | None = None, certify_improvements: bool = True, verbose: bool = False,
                cert_dir: Path | None = None) -> list[Score]:
    """Puntúa cada par (codes[i], instances[i]) en paralelo. Tres fases: ejecutar programas, resolver los SDP
    no cacheados (una vez por sucesión distinta), y certificar en serie las mejoras (raras)."""
    assert len(codes) == len(instances)
    refs = load_refs() if refs is None else refs
    cache = cache if cache is not None else TauCache()
    t0 = time.perf_counter()
    scores = [Score(inst, hashlib.sha1(c.encode()).hexdigest()[:10]) for c, inst in zip(codes, instances)]
    with Pool(workers) as pool:
        for sc, (h, err) in zip(scores, pool.map(_run_h, list(zip(codes, instances)), chunksize=4)):
            sc.h, sc.error = h, err
        todo = {}
        for sc in scores:
            if sc.ok:
                k = tau_key(sc.h, sc.instance)
                if k not in cache and k not in todo:
                    todo[k] = (k, sc.h, sc.instance)
        for k, tau in pool.imap_unordered(_solve, list(todo.values()), chunksize=1):
            cache[k] = tau
    n_solved = len(todo)
    for sc in scores:
        if not sc.ok:
            continue
        sc.tau_ref, sc.ref_source = reference(sc.instance, refs)
        sc.tau = cache[tau_key(sc.h, sc.instance)]
        sc.reward = _reward(sc.tau, sc.tau_ref)
        if sc.reward > GATE:  # mejora afirmada: tolerancias finas + certificado exacto, en serie
            full = score(_code_for(sc.h), sc.instance, refs=refs, certify_improvements=certify_improvements)
            sc.tau, sc.reward, sc.improved, sc.certified, sc.tau_cert, sc.cert_json = full.tau, full.reward, full.improved, full.certified, full.tau_cert, full.cert_json
            if sc.certified and cert_dir is not None:  # cada mejora certificada queda en disco con su programa
                cert_dir.mkdir(parents=True, exist_ok=True)
                name = sc.instance.label().replace("|", "_").replace("=", "") + f"_{sc.code_hash}"
                (cert_dir / f"{name}.json").write_text(json.dumps({"program": codes[scores.index(sc)], **sc.cert_json}, indent=1, ensure_ascii=False), encoding="utf-8")
    cache.save()
    if verbose:
        ok = sum(s.ok for s in scores)
        print(f"score_batch: {len(scores)} programas, {ok} válidos, {n_solved} SDP nuevos, {len(cache)} en caché, "
              f"{sum(s.improved for s in scores)} mejoras afirmadas, {time.perf_counter() - t0:.1f}s")
    return scores
