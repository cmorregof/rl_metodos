from fractions import Fraction

import numpy as np
import pytest

from steprl.certify import Certificate, assemble, certify, exact_matrices, flow_residuals, is_psd_exact, verify
from steprl.pep import constant_bound, gd_worst_case, interpolation_matrices, silver_schedule


def test_exact_matrices_mirror_float_ones():
    h = [1.5, 0.7, 2.2]
    for mu in (0.0, 0.1):
        pairs_f, mats_f = interpolation_matrices(h, 1.0, mu)
        pairs_e, mats_e = exact_matrices([Fraction(x) for x in h], Fraction(1), Fraction(mu))
        assert pairs_f == pairs_e
        for A, B in zip(mats_f, mats_e):
            assert np.allclose(A, np.array([[float(x) for x in row] for row in B]), atol=1e-12)


def test_psd_exact():
    assert is_psd_exact([[Fraction(2), Fraction(1)], [Fraction(1), Fraction(2)]])
    assert is_psd_exact([[Fraction(1), Fraction(1)], [Fraction(1), Fraction(1)]])  # singular pero PSD
    assert not is_psd_exact([[Fraction(1), Fraction(2)], [Fraction(2), Fraction(1)]])
    assert not is_psd_exact([[Fraction(0), Fraction(1)], [Fraction(1), Fraction(1)]])


@pytest.mark.parametrize("h,mu,obj", [([1.0] * 3, 0.0, "fval"), (silver_schedule(3), 0.0, "fval"), ([1.5] * 4, 0.1, "fval"),
                                      ([1.5] * 4, 0.0, "gradnorm"), ([1.5, -0.3, 1.5], 0.0, "fval")])
def test_certificate_is_exact_and_tight(h, mu, obj):
    c = certify(h, mu=mu, objective=obj)
    assert c.ok, c.message
    assert verify(c) == (True, "certificado válido")
    sdp = gd_worst_case(h, mu=mu, objective=obj).value
    assert c.tau_float >= sdp - 1e-9  # una cota certificada nunca está por debajo del peor caso real
    assert c.tau_float - sdp < 1e-5  # y el exceso es del orden del margen δ
    if h == [1.0] * 3 and mu == 0.0 and obj == "fval":
        assert c.tau_float >= constant_bound(3) and c.tau_float - constant_bound(3) < 1e-5


def test_verify_rejects_tampering():
    c = certify([1.5, 1.5])
    d = Certificate.from_json(c.to_json())
    assert verify(d)[0]
    d.tau -= Fraction(1, 10**6)
    assert not verify(d)[0]
    e = Certificate.from_json(c.to_json())
    k = next(iter(e.lam))
    e.lam[k] += Fraction(1, 1000)  # rompe la conservación de flujo
    assert "flujo" in verify(e)[1]
    f = Certificate.from_json(c.to_json())
    f.lam[k] = -f.lam[k]
    assert "negativo" in verify(f)[1]


def test_flow_and_assembly_consistency():
    c = certify([1.2, 2.0, 1.4], objective="gradnorm")
    assert all(r == 0 for r in flow_residuals(3, "gradnorm", c.lam))
    M = assemble(c.h, c.L, c.mu, "gradnorm", c.lam, c.tau)
    assert is_psd_exact(M)


def test_json_roundtrip(tmp_path):
    c = certify([1.5, 1.5, 1.5], mu=0.05)
    c.save(tmp_path / "c.json")
    d = Certificate.load(tmp_path / "c.json")
    assert d.h == c.h and d.tau == c.tau and d.lam == c.lam and verify(d)[0]
