from pathlib import Path

from steprl.evolve import REFERENCE_FIXED, anytime_exponent, baselines, evaluate, extract_code, run_evolution
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


def test_mock_evolution_runs_and_logs(tmp_path: Path):
    pool = run_evolution(MockProvider(list(MOCK_SCRIPTS)), generations=2, objective="fixed", anytime_N=7, out_dir=tmp_path, verbose=False)
    assert (tmp_path / "evolucion.json").exists() and (tmp_path / "mejor.py").exists()
    assert pool[0].ok and pool[0].fixed_ratio < 2.0


def test_cross_entropy_recovers_n2_optimum():
    r = cross_entropy(2, generations=20, population=30, seed=0)
    assert abs(r.value - REFERENCE_FIXED[2]) < 2e-5
