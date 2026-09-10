from pathlib import Path

from steprl.evolve import REFERENCE_FIXED, anytime_exponent, baselines, doubling_exponent, evaluate, extract_code, guarantee_constant, run_evolution
from steprl.providers import MOCK_SCRIPTS, MockProvider
from steprl.search import cross_entropy


def test_extract_code_handles_fences():
    assert extract_code("bla ```python\nx = 1\n``` bla") == "x = 1"
    assert extract_code("def schedule(n): return [1.0]*n") == "def schedule(n): return [1.0]*n"


def test_evaluate_rejects_bad_programs():
    assert not evaluate("def schedule(n): return [0.0] * n", 1, fixed_ns=(2,), anytime_N=0).ok
    assert not evaluate("def schedule(n): return [1.0] * (n + 1)", 1, fixed_ns=(2,), anytime_N=0).ok
    assert "NameError" in evaluate("def schedule(n): return foo(n)", 1, fixed_ns=(2,), anytime_N=0).error


def test_silver_baseline_scores():
    b = baselines(anytime_N=7)
    s = b["silver"]
    assert s.ok and s.fixed[1] > 0 and 1.2 <= s.fixed_ratio < 2.0  # silver truncado a n ≠ 2ᵏ−1 acaba en un pico: mucho peor que el óptimo
    assert b["constante"].fixed_ratio > s.fixed_ratio


def test_anytime_exponent_of_power_law():
    taus = [0.5 / t**1.2 for t in range(1, 32)]
    p, slope = anytime_exponent(taus)
    assert abs(p - 1.2) < 1e-9 and abs(slope - 1.2) < 1e-9
    taus[7] = 0.1  # un prefijo malo hunde la garantía anytime
    assert anytime_exponent(taus)[0] < 0.8


def test_doubling_exponent_is_asymptotic_and_punishes_spikes():
    for C in (0.5, 3.0):  # independiente de la constante
        p, t = doubling_exponent([C / t**1.2 for t in range(1, 64)])
        assert abs(p - 1.2) < 1e-9
    const = [1 / (4 * t + 2) for t in range(1, 64)]
    p, t = doubling_exponent(const)
    assert 0.9 < p < 1.0 and t == 8  # paso constante: O(1/t), el mínimo lo da el t más pequeño
    spiky = [0.5 / t**1.2 for t in range(1, 64)]
    spiky[15] = spiky[7]  # τ_16 = τ_8: pico tipo silver
    p, t = doubling_exponent(spiky)
    assert abs(p) < 1e-9 and t == 16


def test_guarantee_constant_punishes_spikes_and_slow_starts():
    clean = [0.2 / t**1.12 for t in range(1, 64)]
    C, t = guarantee_constant(clean, 1.12)
    assert abs(C - 0.2) < 1e-9
    spiky = list(clean); spiky[15] = spiky[7]
    assert guarantee_constant(spiky, 1.12)[0] > 0.3 and guarantee_constant(spiky, 1.12)[1] == 16
    slow = [0.4] + clean[1:]  # arranque lento: τ₁ grande manda
    assert guarantee_constant(slow, 1.12) == (0.4, 1)
    const = [1 / (4 * t + 2) for t in range(1, 64)]
    assert guarantee_constant(const, 1.12)[1] == 63  # orden 1: la constante crece con N y manda el último t


def test_overfitting_the_horizon_is_punished():
    code = "def schedule(n):\n    return [1.5] * min(n, 8) + [0.01] * max(0, n - 8)  # solo funciona hasta n=8"
    c = evaluate(code, 1, fixed_ns=(1,), anytime_N=(8, 16))
    assert c.ok and c.per_horizon[8]["p"] > 0.8 and c.doubling_p < 0.3  # el peor horizonte manda
    assert c.C_target == max(r["C"] for r in c.per_horizon.values())


def test_mock_evolution_runs_and_logs(tmp_path: Path):
    pool = run_evolution(MockProvider(list(MOCK_SCRIPTS)), generations=2, objective="fixed", anytime_N=(7,), out_dir=tmp_path, verbose=False)
    assert (tmp_path / "evolucion.json").exists() and (tmp_path / "mejor.py").exists()
    assert pool[0].ok and pool[0].fixed_ratio < 2.0


def test_cross_entropy_recovers_n2_optimum():
    r = cross_entropy(2, generations=20, population=30, seed=0)
    assert abs(r.value - REFERENCE_FIXED[2]) < 2e-5
