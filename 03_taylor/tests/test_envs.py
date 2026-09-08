import random

from taylorrl.env_taylor import ABORT, ADD, STOP, TaylorEnv, make_problem, theoretical_action
from taylorrl.env_proof import ProofEnv
from taylorrl.proof_kb import KEY_TO_INDEX, MIN_PROOF_LENGTH, REQUIRED_KEYS, STEPS
from taylorrl.trainer import Config, Trainer


def test_series_coefficients_sum_to_f_inside_radius():
    rng = random.Random(1)
    for _ in range(100):
        p = make_problem(rng)
        if p.diverges:
            continue
        s = sum(p.series.coef(k) * p.x**k for k in range(120))
        assert abs(s - p.series.f(p.x)) < 1e-6, p.name


def test_theoretical_law_is_almost_always_right():
    env = TaylorEnv(seed=11)
    ok = 0
    for _ in range(300):
        s = env.reset()
        done = False
        while not done:
            s, _, done, info = env.step(theoretical_action(*s)[0])
        ok += info in ("stop_ok", "abort_ok")
    assert ok / 300 > 0.95


def test_stop_and_abort_rewards():
    env = TaylorEnv(seed=2)
    env.reset()
    _, r, done, info = env.step(STOP)  # con un solo término nunca hay precisión 1e-6
    assert done and r < 0 and info == "stop_bad"
    env.reset()
    _, r, done, info = env.step(ABORT)
    assert done and (r > 0) == env.problem.diverges


def test_proof_env_validates_dependencies():
    env = ProofEnv()
    env.reset()
    assert not env.is_valid(KEY_TO_INDEX["LAGR"])
    assert env.is_valid(KEY_TO_INDEX["FACT"])  # lema sin dependencias
    env.step(KEY_TO_INDEX["H1"])
    assert env.is_valid(KEY_TO_INDEX["DEF"])
    assert not env.is_valid(KEY_TO_INDEX["D_CINF"])


def test_topological_proof_reaches_qed():
    env = ProofEnv()
    env.reset()
    order = ["H1", "DEF", "MATCH", "AUX", "AUXZ", "ROLLE1", "ROLLEN", "DERIV", "LAGR", "BOUND", "FACT", "CONV", "QED"]
    assert set(order) == set(REQUIRED_KEYS) and len(order) == MIN_PROOF_LENGTH
    for k in order:
        _, _, done, info = env.step(KEY_TO_INDEX[k])
        assert info in ("valid", "qed"), k
    assert done and env.finished and env.n_invalid == 0


def test_short_training_learns_both_phases():
    tr = Trainer(Config(episodes_taylor=2000, episodes_proof=1500, seed=0))
    for _ in tr.run():
        pass
    assert tr.success_curve[-1] >= 0.8
    law = tr.control_law()
    assert law[(3, 2, 1)] == ABORT  # cociente ≥ 1 sostenido → abandonar
    assert law[(1, 0, 1)] == STOP   # último término < tol y cociente pequeño → parar
    keys = [STEPS[a].key for a, ok in tr.greedy_proof if ok]
    assert all(ok for _, ok in tr.greedy_proof)
    assert keys[-1] == "QED" and len(keys) == MIN_PROOF_LENGTH
