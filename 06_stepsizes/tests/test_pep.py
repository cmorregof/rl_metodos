import numpy as np
import pytest

from steprl.pep import (SILVER_RATIO, ZHANG_EXPONENT, constant_bound, fit_exponent, gd_worst_case, prefix_worst_cases,
                        silver_bound, silver_schedule, zhang_schedule, zhang_silver_block)


def test_constant_step_matches_drori_teboulle():
    for n in (1, 2, 4, 7):
        assert abs(gd_worst_case([1.0] * n).value - constant_bound(n)) < 1e-5
    # h < 1 también es tight: 1/(4nh+2)
    assert abs(gd_worst_case([0.5] * 3).value - constant_bound(3, 0.5)) < 1e-5


def test_matches_pepit_on_random_schedules():
    from PEPit import PEP
    from PEPit.functions import SmoothConvexFunction

    rng = np.random.default_rng(0)
    for n in (2, 4):
        h = rng.uniform(0.3, 3.0, n)
        p = PEP()
        f = p.declare_function(SmoothConvexFunction, L=1.0)
        xs = f.stationary_point()
        x0 = p.set_initial_point()
        p.set_initial_condition((x0 - xs) ** 2 <= 1)
        x = x0
        for hi in h:
            x = x - hi * f.gradient(x)
        p.set_performance_metric(f(x) - f(xs))
        assert abs(gd_worst_case(h).value - p.solve(verbose=0)) < 1e-4


def test_silver_schedule_definition_and_guarantee():
    assert np.allclose(silver_schedule(2), [np.sqrt(2), 2, np.sqrt(2)])
    rho = 1 + np.sqrt(2)
    assert np.allclose(silver_schedule(3), [np.sqrt(2), 2, np.sqrt(2), 1 + rho, np.sqrt(2), 2, np.sqrt(2)])  # paso t: 1 + ρ^{ν(t)−1}
    for k in (1, 2, 3):
        assert gd_worst_case(silver_schedule(k)).value <= silver_bound(k) + 1e-6


def test_n1_optimum_is_1_5():
    hs = np.linspace(1.2, 1.8, 13)
    vals = [gd_worst_case([h]).value for h in hs]
    assert abs(hs[int(np.argmin(vals))] - 1.5) < 1e-9
    assert abs(min(vals) - 0.125) < 1e-5


def test_certificate_is_a_proof():
    """El dual del SDP certifica f(xₙ) − f* ≤ τ: multiplicadores ≥ 0 y τ_dual = valor."""
    r = gd_worst_case([1.5, 1.5], want_certificate=True)
    assert r.dual is not None and np.all(r.dual >= -1e-7)
    assert abs(r.tau_dual - r.value) < 1e-5  # el multiplicador de ‖x₀−x*‖² ≤ 1 es la tasa (homogeneidad)


def test_prefix_and_exponent():
    taus = prefix_worst_cases(silver_schedule(3))
    assert len(taus) == 7 and taus[-1] == pytest.approx(gd_worst_case(silver_schedule(3)).value, abs=1e-6)
    p = fit_exponent([3, 7, 15], [gd_worst_case(silver_schedule(k)).value for k in (2, 3, 4)])
    assert 1.0 < p < 1.4


def test_zhang_schedule_construction():
    """Zhang et al. (arXiv:2411.17668): s̄ᵢ = concat(s̄ᵢ₋₁, s̄ᵢ₋₁) es el silver; kⱼ = ⌊2ρʲ⌋ = 4, 11, 28, …"""
    for i in (1, 2, 3):
        b = zhang_silver_block(i)
        assert np.allclose(b, silver_schedule(i)) and abs(sum(b) - (SILVER_RATIO**i - 1)) < 1e-9
    s = zhang_schedule(60)
    assert len(s) == 60 and np.allclose(zhang_schedule(20), s[:20])  # no depende del horizonte
    assert s[1] == pytest.approx(np.sqrt(2)) and s[7] == pytest.approx(np.sqrt(2))  # 4 bloques s̄₁ con sus uniones: 8 pasos
    assert abs(ZHANG_EXPONENT - 1.119) < 1e-3
