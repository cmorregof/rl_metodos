import random
from statistics import median

from secantrl.env_secant import BISECT, FALSI, SEC, SecantEnv, make_problem, theoretical_action
from secantrl.env_proof import ProofEnv
from secantrl.proof_kb import KEY_TO_INDEX, MIN_PROOF_LENGTH, REQUIRED_KEYS, STEPS
from secantrl.trainer import Config, Trainer


def test_problems_have_root_inside_a_sign_changing_bracket():
    rng = random.Random(1)
    for _ in range(300):
        p = make_problem(rng)
        assert abs(p.f(p.root)) < 1e-6, p.name
        assert p.lo < p.root < p.hi, p.name
        assert p.f(p.lo) * p.f(p.hi) < 0, p.name


def run_policy(policy, n=400, seed=7):
    env = SecantEnv(seed=seed)
    conv, evals = 0, []
    for _ in range(n):
        s = env.reset()
        done = False
        while not done:
            a = SEC if s == ("inicio",) else policy(s)
            s, _, done, info = env.step(a)
        conv += info == "converged"
        if info == "converged":
            evals.append(env.n_evals)
    return conv / n, sum(evals) / max(1, len(evals))


def test_dekker_beats_pure_secant_regula_falsi_and_bisection():
    pure, _ = run_policy(lambda s: SEC)
    falsi, _ = run_policy(lambda s: FALSI)
    bis, bis_evals = run_policy(lambda s: BISECT)
    dekker, dekker_evals = run_policy(lambda s: theoretical_action(*s)[0])
    assert pure < 0.8  # ∛x oscila, x·e^(−x²) dispara puntos lejos, ln x cae fuera del dominio
    assert falsi < 0.8  # un extremo se estanca: lineal y lento
    assert bis > 0.99 and bis_evals > 25  # siempre converge, pero a 1 bit por evaluación
    assert dekker > 0.97 and dekker_evals < 0.5 * bis_evals


def test_cbrt_pure_secant_oscillates_and_bisection_saves_it():
    env = SecantEnv(seed=0)
    while True:
        s = env.reset()
        if env.problem.name == "∛x":
            break
    done = False
    while not done:
        s, _, done, info = env.step(SEC)
    assert info == "timeout"  # |xₙ| no decrece: la secante pura no converge en ∛x
    while True:
        s = env.reset()
        if env.problem.name == "∛x":
            break
    done = False
    while not done:
        s, _, done, info = env.step(SEC if s == ("inicio",) or s[1] else BISECT)
    assert info == "converged"


def test_bracket_is_maintained_and_shrinks():
    env = SecantEnv(seed=3)
    for _ in range(50):
        s = env.reset()
        done = False
        while not done:
            lo, hi = env.lo, env.hi
            s, _, done, _ = env.step(random.Random(env.n_iter).choice([SEC, FALSI, BISECT]) if s != ("inicio",) else SEC)
            assert env.lo <= env.hi and env.lo >= lo and env.hi <= hi
            assert env.lo <= env.problem.root <= env.hi


def test_observed_order_is_phi_on_clean_problems():
    env = SecantEnv(seed=11)
    orders = []
    while len(orders) < 40:
        s = env.reset()
        if env.problem.name != "x³ − x − 1":
            continue
        done = False
        while not done:
            s, _, done, info = env.step(SEC if s == ("inicio",) or s[1] else BISECT)
        o = env.observed_order()
        if info == "converged" and o == o:
            orders.append(o)
    assert 1.5 < median(orders) < 1.75


def test_proof_env_validates_dependencies():
    env = ProofEnv()
    env.reset()
    assert not env.is_valid(KEY_TO_INDEX["ERRID"])
    env.step(KEY_TO_INDEX["H1"])
    assert env.is_valid(KEY_TO_INDEX["INTERP"]) and env.is_valid(KEY_TO_INDEX["NBHD"])
    assert not env.is_valid(KEY_TO_INDEX["D_QUAD"])


def test_topological_proof_reaches_qed():
    env = ProofEnv()
    env.reset()
    order = ["H1", "NBHD", "DEF", "INTERP", "ERRID", "MVT", "PROD", "DELTA", "INV", "CONV", "FIB", "ORDER", "QED"]
    assert set(order) == set(REQUIRED_KEYS) and len(order) == MIN_PROOF_LENGTH
    for k in order:
        _, _, done, info = env.step(KEY_TO_INDEX[k])
        assert info in ("valid", "qed"), k
    assert done and env.finished and env.n_invalid == 0


def test_short_training_learns_both_phases():
    tr = Trainer(Config(episodes_secant=2000, episodes_proof=1500, seed=0))
    for _ in tr.run():
        pass
    assert tr.success_curve[-1] >= 0.9
    law = tr.control_law()
    assert law[(0, 1)] == SEC and law[(1, 1)] == SEC  # dentro del corchete y bajando → secante
    assert law[(2, 0)] in (BISECT, FALSI) and law[(3, 0)] in (BISECT, FALSI)  # se sale del corchete → refugio
    assert 1.4 < tr.order_median() < 1.9  # orden ≈ φ observado
    keys = [STEPS[a].key for a, ok in tr.greedy_proof if ok]
    assert all(ok for _, ok in tr.greedy_proof)
    assert keys[-1] == "QED" and len(keys) == MIN_PROOF_LENGTH
