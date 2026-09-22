import math
import random

from interprl.env_interp import ADD, GAMMAS, STOP, InterpEnv, theoretical_action
from interprl.env_proof import ProofEnv
from interprl.interp import (
    barycentric_weights,
    chebyshev_first_kind,
    chebyshev_lobatto,
    equispaced,
    extended_chebyshev,
    interpolate,
    lebesgue_constant,
    max_abs_nodal_poly,
)
from interprl.proof_kb import KEY_TO_INDEX, MIN_PROOF_LENGTH, REQUIRED_KEYS, STEPS
from interprl.search import run_search, symmetric_nodes
from interprl.trainer import Config, Trainer


def test_interpolant_reproduces_polynomials_exactly():
    x = chebyshev_lobatto(5)
    f = lambda t: 3 * t**5 - t**2 + 0.5
    y = [f(t) for t in x]
    w = barycentric_weights(x)
    for t in (-0.9, -0.3, 0.1, 0.77):
        assert abs(interpolate(x, y, w, t) - f(t)) < 1e-12


def test_lebesgue_matches_classical_table():
    # Chebyshev de primera especie: Λ₁ = √2, Λ₂ = 5/3, Λ₄ ≈ 1.9889 (Brutman)
    assert abs(lebesgue_constant(chebyshev_first_kind(1))[0] - math.sqrt(2)) < 1e-9
    assert abs(lebesgue_constant(chebyshev_first_kind(2))[0] - 5 / 3) < 1e-9
    assert abs(lebesgue_constant(chebyshev_first_kind(4))[0] - 1.9889) < 1e-3
    # equiespaciados crecen exponencialmente, Chebyshev logarítmicamente
    assert lebesgue_constant(equispaced(16))[0] > 500
    assert lebesgue_constant(chebyshev_lobatto(16))[0] < 3
    # el extendido mejora al de primera especie
    assert lebesgue_constant(extended_chebyshev(8))[0] <= lebesgue_constant(chebyshev_first_kind(8))[0] + 1e-12


def test_chebyshev_minimizes_nodal_polynomial():
    n = 8
    cheb = max_abs_nodal_poly(chebyshev_first_kind(n))
    assert abs(cheb - 2.0 ** (-n)) < 1e-6
    assert max_abs_nodal_poly(equispaced(n)) > cheb


def test_runge_diverges_with_equispaced_and_converges_with_chebyshev():
    env = InterpEnv(seed=0)
    env.problem = type(env.problem)("runge", lambda x: 1 / (1 + 8 * x * x), False)
    env.fgrid = [env.problem.f(t) for t in env.grid]
    assert env.optimal_nodes(0.0) is None
    assert env.optimal_nodes(1.0) is not None


def test_theoretical_law_is_almost_always_right_with_chebyshev():
    env = InterpEnv(seed=11)
    ok = 0
    for _ in range(100):
        env.reset()
        s = env.set_gamma(1.0)
        done = False
        while not done:
            s, _, done, info = env.step(theoretical_action(*s)[0])
        ok += info == "stop_ok"
    assert ok / 100 > 0.9


def test_search_beats_or_matches_chebyshev():
    r = run_search(6, seed=0, iters=15)
    assert r["ratio_vs_mejor_referencia"] <= 1.0 + 1e-9
    assert len(r["mejor"]["nodes"]) == 7
    assert symmetric_nodes([0.0, 0.5]) == [-1.0, -0.5, 0.0, 0.5, 1.0]


def test_proof_env_validates_dependencies():
    env = ProofEnv()
    env.reset()
    assert not env.is_valid(KEY_TO_INDEX["AUX"])
    assert env.is_valid(KEY_TO_INDEX["H1"])
    env.step(KEY_TO_INDEX["H2"])
    assert env.is_valid(KEY_TO_INDEX["DEFW"])
    assert not env.is_valid(KEY_TO_INDEX["D_MORE"])


def test_topological_proof_reaches_qed():
    env = ProofEnv()
    env.reset()
    order = ["H1", "H2", "EXIST", "DEFW", "FIX", "AUX", "ZEROS", "ROLLE", "DERIV", "FORMULA", "BOUND", "CHEB", "QED"]
    assert set(order) == set(REQUIRED_KEYS) and len(order) == MIN_PROOF_LENGTH
    for k in order:
        _, _, done, info = env.step(KEY_TO_INDEX[k])
        assert info in ("valid", "qed"), k
    assert done and env.finished and env.n_invalid == 0


def test_short_training_learns_both_phases():
    tr = Trainer(Config(episodes_interp=2500, episodes_proof=1500, seed=1))
    for _ in tr.run():
        pass
    assert tr.success_curve[-1] >= 0.7
    assert tr.gamma_policy() == 1.0
    law = tr.control_law()
    assert law[(0, 0, 1)] == STOP
    assert law[(3, 0, 1)] == ADD
    keys = [STEPS[a].key for a, ok in tr.greedy_proof if ok]
    assert all(ok for _, ok in tr.greedy_proof)
    assert keys[-1] == "QED" and len(keys) == MIN_PROOF_LENGTH
