import random

from bisectrl.env_bisection import KEEP_LEFT, KEEP_RIGHT, LAMBDAS, BisectionEnv, make_problem
from bisectrl.env_proof import ProofEnv
from bisectrl.proof_kb import KEY_TO_INDEX, MIN_PROOF_LENGTH, REQUIRED_KEYS, STEPS
from bisectrl.trainer import Config, Trainer


def test_problems_have_sign_change():
    rng = random.Random(1)
    for _ in range(200):
        p = make_problem(rng)
        assert p.f(p.a) * p.f(p.b) < 0
        assert p.a < p.root < p.b


def test_true_bisection_converges_and_reward_is_shaped():
    env = BisectionEnv(tol=1e-3, seed=3)
    total, done = 0.0, False
    while not done:
        s_keep = env.step_split(LAMBDAS.index(0.5))
        _, _, sx, _ = s_keep[:4]
        sa = 1 if env.fa > 0 else -1
        keep = KEEP_LEFT if sa * sx < 0 else KEEP_RIGHT
        _, r, done, info = env.step_keep(keep)
        total += r
    assert info == "converged"
    assert env.n_iter <= env.optimal_steps() + 1
    # retorno = log2(w0/wn) − n, con λ = 1/2 el progreso es exactamente 1 bit/paso
    assert abs(total) < 1e-9


def test_losing_sign_change_is_penalized():
    env = BisectionEnv(seed=5)
    env.step_split(LAMBDAS.index(0.5))
    sa = 1 if env.fa > 0 else -1
    sx = 1 if env.fx > 0 else -1
    wrong = KEEP_RIGHT if sa * sx < 0 else KEEP_LEFT
    _, r, done, info = env.step_keep(wrong)
    assert info == "lost" and done and r <= -10


def test_proof_env_validates_dependencies():
    env = ProofEnv()
    env.reset()
    assert not env.is_valid(KEY_TO_INDEX["DEF"])  # necesita H2
    assert env.is_valid(KEY_TO_INDEX["H2"])
    env.step(KEY_TO_INDEX["H2"])
    assert env.is_valid(KEY_TO_INDEX["DEF"])
    assert not env.is_valid(KEY_TO_INDEX["H2"])  # repetido
    assert not env.is_valid(KEY_TO_INDEX["D_BOLZ"])  # distractor


def test_topological_proof_reaches_qed():
    env = ProofEnv()
    env.reset()
    order = ["H1", "H2", "DEF", "INV", "WIDTH", "MONO_A", "MONO_B", "CONV_A", "CONV_B", "SAME", "CONT", "ROOT", "MID", "QED"]
    assert set(order) == set(REQUIRED_KEYS) and len(order) == MIN_PROOF_LENGTH
    for k in order:
        _, r, done, info = env.step(KEY_TO_INDEX[k])
        assert info in ("valid", "qed"), k
    assert done and env.finished and env.n_invalid == 0


def test_short_training_learns_both_phases():
    tr = Trainer(Config(episodes_bisection=1500, episodes_proof=1500, seed=0))
    for _ in tr.run():
        pass
    assert tr.lambda_policy() == 0.5
    assert tr.keep_policy_is_bolzano()
    keys = [STEPS[a].key for a, ok in tr.greedy_proof if ok]
    assert all(ok for _, ok in tr.greedy_proof)
    assert keys[-1] == "QED" and len(keys) == MIN_PROOF_LENGTH
