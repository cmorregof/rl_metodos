import random

from newtonrl.env_newton import BACK, FULL, HALF, NewtonEnv, make_problem, theoretical_action
from newtonrl.env_proof import ProofEnv
from newtonrl.proof_kb import KEY_TO_INDEX, MIN_PROOF_LENGTH, REQUIRED_KEYS, STEPS
from newtonrl.trainer import Config, Trainer


def test_problems_have_root_and_consistent_derivative():
    rng = random.Random(1)
    for _ in range(200):
        p = make_problem(rng)
        assert abs(p.f(p.root)) < 1e-6, p.name
        h = 1e-6
        x = p.x0
        num = (p.f(x + h) - p.f(x - h)) / (2 * h)
        assert abs(num - p.df(x)) < 1e-3 * max(1.0, abs(p.df(x))), p.name


def run_policy(policy, n=400, seed=7):
    env = NewtonEnv(seed=seed)
    conv = 0
    for _ in range(n):
        s = env.reset()
        done = False
        while not done:
            a = FULL if s == ("inicio",) else policy(s)
            s, _, done, info = env.step(a)
        conv += info == "converged"
    return conv / n


def test_safeguards_beat_pure_newton():
    pure = run_policy(lambda s: FULL)
    safe = run_policy(lambda s: BACK if s[0] == 3 else (HALF if s[0] == 2 else FULL))  # Newton amortiguado con backtracking
    assert pure < 0.85
    assert safe > 0.9
    assert safe > pure + 0.1


def test_cbrt_pure_newton_diverges_and_backtracking_saves_it():
    env = NewtonEnv(seed=0)
    while True:
        env.reset()
        if env.problem.name == "∛x":
            break
    x0 = env.problem.x0
    _, r, done, info = env.step(FULL)
    assert abs(env.x + 2 * x0) < 1e-9  # xₙ₊₁ = −2xₙ
    _, r, done, info = env.step(BACK)
    assert abs(env.x) < abs(x0) and not done


def test_proof_env_validates_dependencies():
    env = ProofEnv()
    env.reset()
    assert not env.is_valid(KEY_TO_INDEX["ERRID"])
    env.step(KEY_TO_INDEX["H1"])
    assert env.is_valid(KEY_TO_INDEX["TAYLOR"]) and env.is_valid(KEY_TO_INDEX["NBHD"])
    assert not env.is_valid(KEY_TO_INDEX["D_GLOBAL"])


def test_topological_proof_reaches_qed():
    env = ProofEnv()
    env.reset()
    order = ["H1", "NBHD", "DEF", "TAYLOR", "ERRID", "QUAD", "DELTA", "INV", "CONV", "ORDER", "QED"]
    assert set(order) == set(REQUIRED_KEYS) and len(order) == MIN_PROOF_LENGTH
    for k in order:
        _, _, done, info = env.step(KEY_TO_INDEX[k])
        assert info in ("valid", "qed"), k
    assert done and env.finished and env.n_invalid == 0


def test_short_training_learns_both_phases():
    tr = Trainer(Config(episodes_newton=2000, episodes_proof=1500, seed=0))
    for _ in tr.run():
        pass
    assert tr.success_curve[-1] >= 0.85
    law = tr.control_law()
    assert law[(0, 0)] == FULL and law[(1, 0)] == FULL  # cuenca cuadrática → paso completo
    assert law[(3, 1)] == BACK  # empeoró y el paso crece → retroceder
    orders = list(tr.orders)
    assert sum(orders) / len(orders) > 1.6  # convergencia (casi) cuadrática observada
    keys = [STEPS[a].key for a, ok in tr.greedy_proof if ok]
    assert all(ok for _, ok in tr.greedy_proof)
    assert keys[-1] == "QED" and len(keys) == MIN_PROOF_LENGTH
