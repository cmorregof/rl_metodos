import random

from fixpointrl.env_fixpoint import DOUBLE, FLIP, HALVE, KEEP, FixpointEnv, make_problem, theoretical_action
from fixpointrl.env_proof import ProofEnv
from fixpointrl.proof_kb import KEY_TO_INDEX, MIN_PROOF_LENGTH, REQUIRED_KEYS, STEPS
from fixpointrl.trainer import Config, Trainer


def test_problems_have_root_and_derivative():
    rng = random.Random(1)
    for _ in range(200):
        p = make_problem(rng)
        assert abs(p.f(p.root)) < 1e-6
        h = 1e-6
        num = (p.f(p.root + h) - p.f(p.root - h)) / (2 * h)
        assert abs(num - p.dfr) < 1e-3 * max(1, abs(p.dfr))


def test_theoretical_law_converges_fast():
    """Seguir la ley de control teórica converge casi siempre y en pocas iteraciones."""
    env = FixpointEnv(seed=11)
    conv, iters = 0, []
    for _ in range(300):
        s = env.reset()
        done = False
        while not done:
            a = KEEP if s == ("inicio",) else theoretical_action(*s)[0]
            s, _, done, info = env.step(a)
        conv += info == "converged"
        if info == "converged":
            iters.append(env.n_iter)
    assert conv / 300 > 0.8  # muy por encima del ~35 % de un α fijo
    assert sum(iters) / len(iters) < 15


def test_divergence_is_penalized_and_terminal():
    env = FixpointEnv(seed=3)
    env.reset()
    env.alpha = 8.0 if env.problem.dfr > 0 else -8.0  # |1 − αf'| ≫ 1
    done, info, r = False, "", 0.0
    while not done:
        _, r, done, info = env.step(KEEP)
    assert info == "diverged" and r <= -10


def test_proof_env_validates_dependencies():
    env = ProofEnv()
    env.reset()
    assert not env.is_valid(KEY_TO_INDEX["EXIST"])  # necesita AUX
    env.step(KEY_TO_INDEX["H1"])
    env.step(KEY_TO_INDEX["AUX"])
    assert env.is_valid(KEY_TO_INDEX["EXIST"])
    assert not env.is_valid(KEY_TO_INDEX["H1"])  # repetido
    assert not env.is_valid(KEY_TO_INDEX["D_LOCAL"])  # distractor


def test_topological_proof_reaches_qed():
    env = ProofEnv()
    env.reset()
    order = ["H1", "H2", "AUX", "EXIST", "UNIQ", "DEF", "ERR", "GEO", "LIM", "PRIORI", "STEP", "POST", "QED"]
    assert set(order) == set(REQUIRED_KEYS) and len(order) == MIN_PROOF_LENGTH
    for k in order:
        _, _, done, info = env.step(KEY_TO_INDEX[k])
        assert info in ("valid", "qed"), k
    assert done and env.finished and env.n_invalid == 0


def test_short_training_learns_both_phases():
    tr = Trainer(Config(episodes_fixpoint=2000, episodes_proof=1500, seed=0))
    for _ in tr.run():
        pass
    ok, n = tr.law_matches_theory()
    assert n == 12 and ok >= 10
    assert tr.success_curve[-1] >= 0.8
    keys = [STEPS[a].key for a, ok_ in tr.greedy_proof if ok_]
    assert all(ok_ for _, ok_ in tr.greedy_proof)
    assert keys[-1] == "QED" and len(keys) == MIN_PROOF_LENGTH
