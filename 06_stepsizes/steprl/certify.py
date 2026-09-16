"""Certificados exactos: del dual del SDP a una demostración verificable en aritmética racional.

El valor del PEP en punto flotante es evidencia numérica: el solver trabaja con
tolerancias finitas y un agente que optimiza τ contra el solver puede aprender a
explotarlas. Un **teorema** para un programa de pasos concreto exige un
certificado que se compruebe sin punto flotante. Aquí:

1. Se resuelve el **dual** del PEP con un margen δ: minimizar τ sujeto a
   λᵢⱼ ≥ 0, conservación de flujo en los coeficientes de fₖ, y
   M(λ, τ) := Σ λᵢⱼ Aᵢⱼ + τ·x₀x₀ᵀ [− gₙgₙᵀ si el criterio es ‖∇f(xₙ)‖²] ⪰ δI.
2. Se redondean λ y τ a racionales con `digits` decimales (τ hacia arriba) y se
   **repara** la conservación de flujo exactamente ajustando los multiplicadores
   de las desigualdades con el punto x* (que no cambian la conclusión).
3. Se comprueba en `fractions.Fraction` que λ ≥ 0, que el flujo cierra y que
   M(λ, τ) es semidefinida positiva (factorización LDLᵀ exacta).

Si 3 pasa, queda demostrado: para esos pasos h (los racionales exactos que
representan los flotantes del programa) y esa μ,

    f(xₙ) − f* ≤ τ · L · ‖x₀ − x*‖²       (o ‖∇f(xₙ)‖² ≤ τ · L² · ‖x₀ − x*‖²),

porque para toda f ∈ F_{μ,L} y todo x₀ la combinación Σ λᵢⱼ·(desigualdad ᵢⱼ)
+ τ·(R² − ‖x₀ − x*‖²) ≥ 0 se reescribe como  τR² − f(xₙ) + f* = ⟨M, G⟩ ≥ 0.
`verify()` rehace la comprobación 3 solo a partir de los racionales, sin usar
nada del solver, y es lo que un revisor puede ejecutar.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path

import cvxpy as cp
import numpy as np

from .pep import gd_worst_case, interpolation_matrices

Rational = Fraction
Matrix = list[list[Fraction]]

# ---------------------------------------------------------------------------
# Construcción exacta (espejo de pep.interpolation_matrices)
# ---------------------------------------------------------------------------


def _vectors_exact(h: list[Fraction], L: Fraction) -> tuple[list[list[Fraction]], list[list[Fraction]]]:
    n = len(h)
    dim = n + 2
    xs, gs = [], []
    x = [Fraction(0)] * dim
    x[0] = Fraction(1)
    for i in range(n + 1):
        g = [Fraction(0)] * dim
        g[i + 1] = Fraction(1)
        xs.append(list(x))
        gs.append(g)
        if i < n:
            x = [xk - (h[i] / L) * gk for xk, gk in zip(x, g)]
    xs.append([Fraction(0)] * dim)
    gs.append([Fraction(0)] * dim)
    return xs, gs


def _outer(u, v) -> Matrix:
    return [[ui * vj for vj in v] for ui in u]


def _sym(u, v) -> Matrix:
    return [[(u[i] * v[j] + v[i] * u[j]) / 2 for j in range(len(v))] for i in range(len(u))]


def _add(A: Matrix, B: Matrix, c: Fraction = Fraction(1)) -> Matrix:
    return [[a + c * b for a, b in zip(ra, rb)] for ra, rb in zip(A, B)]


def _zeros(dim: int) -> Matrix:
    return [[Fraction(0)] * dim for _ in range(dim)]


def exact_matrices(h: list[Fraction], L: Fraction, mu: Fraction) -> tuple[list[tuple[int, int]], list[Matrix]]:
    """Aᵢⱼ exactas: fᵢ − fⱼ − ⟨Aᵢⱼ, G⟩ ≥ 0 es la desigualdad de interpolación de F_{μ,L}."""
    if not (0 <= mu < L):
        raise ValueError("se requiere 0 ≤ μ < L")
    xs, gs = _vectors_exact(h, L)
    dim = len(h) + 2
    c = 1 / (2 * (1 - mu / L))
    pairs, mats = [], []
    for i in range(dim):
        for j in range(dim):
            if i == j:
                continue
            dx = [a - b for a, b in zip(xs[i], xs[j])]
            dg = [a - b for a, b in zip(gs[i], gs[j])]
            A = _sym(gs[j], dx)
            A = _add(A, _outer(dg, dg), c / L)
            A = _add(A, _outer(dx, dx), c * mu)
            A = _add(A, _sym(dg, dx), -2 * c * mu / L)
            pairs.append((i, j))
            mats.append(A)
    return pairs, mats


def assemble(h: list[Fraction], L: Fraction, mu: Fraction, objective: str, lam: dict[tuple[int, int], Fraction], tau: Fraction) -> Matrix:
    """M(λ, τ) = Σ λᵢⱼ Aᵢⱼ + τ x₀x₀ᵀ − [gₙgₙᵀ si gradnorm]."""
    pairs, mats = exact_matrices(h, L, mu)
    dim = len(h) + 2
    M = _zeros(dim)
    for p, A in zip(pairs, mats):
        lp = lam.get(p, Fraction(0))
        if lp:
            M = _add(M, A, lp)
    xs, gs = _vectors_exact(h, L)
    M = _add(M, _outer(xs[0], xs[0]), tau)
    if objective == "gradnorm":
        M = _add(M, _outer(gs[len(h)], gs[len(h)]), Fraction(-1))
    return M


def flow_residuals(n: int, objective: str, lam: dict[tuple[int, int], Fraction]) -> list[Fraction]:
    """Coeficiente de fₖ en la combinación, k = 0…n (debe ser 0): [k = n]·[fval] + Σⱼ λₖⱼ − Σᵢ λᵢₖ."""
    res = []
    for k in range(n + 1):
        out = sum((v for (i, j), v in lam.items() if i == k), Fraction(0))
        inn = sum((v for (i, j), v in lam.items() if j == k), Fraction(0))
        res.append(out - inn + (1 if (objective == "fval" and k == n) else 0))
    return res


def is_psd_exact(M: Matrix) -> bool:
    """LDLᵀ exacta sin pivoteo: M ⪰ 0 sii cada pivote es ≥ 0 y, cuando es 0, su fila/columna restante es 0."""
    A = [row[:] for row in M]
    n = len(A)
    for k in range(n):
        d = A[k][k]
        if d < 0:
            return False
        if d == 0:
            if any(A[k][j] != 0 for j in range(k + 1, n)):
                return False
            continue
        for i in range(k + 1, n):
            if A[i][k] == 0:
                continue
            f = A[i][k] / d
            for j in range(k + 1, n):
                A[i][j] -= f * A[k][j]
    return True


# ---------------------------------------------------------------------------
# Dual con margen, redondeo y reparación
# ---------------------------------------------------------------------------


def dual_sdp(h, L: float = 1.0, mu: float = 0.0, objective: str = "fval", delta: float = 1e-7, solver_opts: dict | None = None):
    """Minimiza τ con λ ≥ 0, flujo cerrado y M(λ, τ) ⪰ δI. Devuelve (τ, λ como dict, status)."""
    h = np.asarray(h, dtype=float)
    n = len(h)
    dim = n + 2
    pairs, mats = interpolation_matrices(h, L, mu)
    P = len(pairs)
    lam = cp.Variable(P, nonneg=True)
    tau = cp.Variable(nonneg=True)
    x0 = np.zeros(dim); x0[0] = 1.0
    gn = np.zeros(dim); gn[n + 1] = 1.0
    M = sum(lam[p] * mats[p] for p in range(P)) + tau * np.outer(x0, x0)
    if objective == "gradnorm":
        M = M - np.outer(gn, gn)
    cons = [M - delta * np.eye(dim) >> 0]
    idx = {p: k for k, p in enumerate(pairs)}
    for k in range(n + 1):
        out = sum(lam[idx[(k, j)]] for j in range(dim) if j != k)
        inn = sum(lam[idx[(i, k)]] for i in range(dim) if i != k)
        cons.append(out - inn + (1.0 if (objective == "fval" and k == n) else 0.0) == 0)
    prob = cp.Problem(cp.Minimize(tau), cons)
    opts = {"tol_gap_abs": 1e-10, "tol_gap_rel": 1e-10, "tol_feas": 1e-10, "max_iter": 500}
    opts.update(solver_opts or {})
    prob.solve(solver="CLARABEL", **opts)
    if prob.status not in ("optimal", "optimal_inaccurate"):
        return float("nan"), {}, prob.status
    return float(tau.value), {p: float(lam.value[k]) for p, k in idx.items()}, prob.status


def round_and_repair(n: int, objective: str, lam_f: dict, tau_f: float, digits: int) -> tuple[dict[tuple[int, int], Fraction], Fraction]:
    D = 10 ** digits
    lam = {p: Fraction(max(0, round(v * D)), D) for p, v in lam_f.items()}
    tau = Fraction(math.ceil(tau_f * D), D)
    star = n + 1
    for k, r in enumerate(flow_residuals(n, objective, lam)):
        if r > 0:  # sobra salida de k: añadimos entrada λ_{*k}
            lam[(star, k)] = lam.get((star, k), Fraction(0)) + r
        elif r < 0:  # falta salida: añadimos λ_{k*}
            lam[(k, star)] = lam.get((k, star), Fraction(0)) - r
    return {p: v for p, v in lam.items() if v != 0}, tau


# ---------------------------------------------------------------------------
# Certificado
# ---------------------------------------------------------------------------


@dataclass
class Certificate:
    h: list[Fraction]
    mu: Fraction
    L: Fraction
    objective: str
    tau: Fraction  # cota certificada
    lam: dict[tuple[int, int], Fraction]
    tau_sdp: float = float("nan")  # valor primal en punto flotante (informativo)
    delta: float = 0.0
    digits: int = 0
    ok: bool = False
    message: str = ""

    @property
    def n(self) -> int:
        return len(self.h)

    @property
    def tau_float(self) -> float:
        return float(self.tau)

    def statement(self) -> str:
        hs = ", ".join(f"{float(x):.12g}" for x in self.h)
        met = "f(xₙ) − f*" if self.objective == "fval" else "‖∇f(xₙ)‖²"
        return (f"Teorema (certificado exacto). Para el descenso de gradiente con pasos h = ({hs}) "
                f"(racionales exactos de esos flotantes) sobre F_{{μ,L}} con μ = {self.mu}, L = {self.L}, "
                f"y ‖x₀ − x*‖ ≤ R: {met} ≤ τ·L·R² con τ = {self.tau} ≈ {self.tau_float:.10f}. "
                f"[valor del SDP: {self.tau_sdp:.10f}; exceso: {self.tau_float - self.tau_sdp:.2e}]")

    def to_json(self) -> dict:
        return {
            "h": [str(x) for x in self.h], "h_float": [float(x) for x in self.h],
            "mu": str(self.mu), "L": str(self.L), "objective": self.objective,
            "tau": str(self.tau), "tau_float": self.tau_float, "tau_sdp": self.tau_sdp,
            "lam": {f"{i},{j}": str(v) for (i, j), v in sorted(self.lam.items())},
            "delta": self.delta, "digits": self.digits, "ok": self.ok, "message": self.message,
        }

    @classmethod
    def from_json(cls, d: dict) -> "Certificate":
        return cls(
            h=[Fraction(x) for x in d["h"]], mu=Fraction(d["mu"]), L=Fraction(d["L"]), objective=d["objective"],
            tau=Fraction(d["tau"]), lam={tuple(int(t) for t in k.split(",")): Fraction(v) for k, v in d["lam"].items()},
            tau_sdp=d.get("tau_sdp", float("nan")), delta=d.get("delta", 0.0), digits=d.get("digits", 0),
            ok=d.get("ok", False), message=d.get("message", ""),
        )

    def save(self, path: Path) -> None:
        Path(path).write_text(json.dumps(self.to_json(), indent=1, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "Certificate":
        return cls.from_json(json.loads(Path(path).read_text(encoding="utf-8")))


def verify(cert: Certificate) -> tuple[bool, str]:
    """Comprobación independiente y exacta: solo usa los racionales del certificado."""
    if any(v < 0 for v in cert.lam.values()):
        return False, "algún λ es negativo"
    if cert.tau < 0:
        return False, "τ negativo"
    res = flow_residuals(cert.n, cert.objective, cert.lam)
    if any(r != 0 for r in res):
        return False, f"la conservación de flujo no cierra: {[str(r) for r in res]}"
    M = assemble(cert.h, cert.L, cert.mu, cert.objective, cert.lam, cert.tau)
    if not is_psd_exact(M):
        return False, "M(λ, τ) no es semidefinida positiva"
    return True, "certificado válido"


def certify(h, mu: float = 0.0, L: float = 1.0, objective: str = "fval", delta: float = 1e-7, digits: int = 12,
            attempts: tuple[tuple[float, int], ...] = ((1e-7, 12), (1e-6, 12), (1e-5, 10), (1e-4, 9))) -> Certificate:
    """Certificado exacto para un programa de pasos. Prueba (δ, dígitos) crecientes hasta que `verify` pase."""
    h_exact = [Fraction(float(x)) for x in h]
    mu_exact, L_exact = Fraction(mu), Fraction(L)
    tau_sdp = gd_worst_case(h, L=L, mu=mu, objective=objective).value
    last = None
    for d, dg in ((delta, digits),) + tuple(a for a in attempts if a != (delta, digits)):
        tau_f, lam_f, status = dual_sdp(h, L=L, mu=mu, objective=objective, delta=d)
        if not (tau_f == tau_f):
            last = Certificate(h_exact, mu_exact, L_exact, objective, Fraction(0), {}, tau_sdp, d, dg, False, f"dual: {status}")
            continue
        lam, tau = round_and_repair(len(h), objective, lam_f, tau_f, dg)
        cert = Certificate(h_exact, mu_exact, L_exact, objective, tau, lam, tau_sdp, d, dg)
        ok, msg = verify(cert)
        cert.ok, cert.message = ok, msg
        if ok:
            return cert
        last = cert
    return last
