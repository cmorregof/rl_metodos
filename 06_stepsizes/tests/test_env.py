import numpy as np
import pytest

from steprl.env import GATE, INVALID_REWARD, TauCache, run_program, score, score_batch, validate
from steprl.refs import FAMILY_TEST, FAMILY_TRAIN, Instance, family, reference

SILVER = "import math\nRHO = 1 + math.sqrt(2)\ndef schedule(n):\n    def v2(t):\n        k = 0\n        while t % 2 == 0:\n            t //= 2; k += 1\n        return k\n    return [1 + RHO ** (v2(t) - 1) for t in range(1, n + 1)]\n"
CONST = "def schedule(n):\n    return [1.0] * n\n"
H15 = "def schedule(n):\n    return [1.5] * n\n"


def test_sandbox_blocks_imports_files_and_loops():
    assert "no permitido" in score("import os\ndef schedule(n):\n    return [1.0]*n\n", Instance(3)).error
    assert "open" in score("def schedule(n):\n    open('x')\n    return [1.0]*n\n", Instance(3)).error
    assert "tiempo" in score("def schedule(n):\n    while True:\n        pass\n", Instance(3)).error
    assert "define" in score("x = 1\n", Instance(3)).error
    assert run_program("import numpy as np\ndef schedule(n):\n    return list(np.ones(n))\n", 3) == [1.0, 1.0, 1.0]


def test_validation_and_negative_flag():
    with pytest.raises(ValueError):
        validate([1.0, 1.0], Instance(3))
    with pytest.raises(ValueError):
        validate([1.5, -0.2, 1.5], Instance(3))
    assert validate([1.5, -0.2, 1.5], Instance(3, allow_negative=True)).tolist() == [1.5, -0.2, 1.5]
    with pytest.raises(ValueError):
        validate([1.0, float("nan"), 1.0], Instance(3))


def test_rewards_are_relative_to_reference():
    s = score(CONST, Instance(3))
    assert s.ok and s.reward == pytest.approx(np.log(0.042895 / (1 / 14)), abs=1e-4) and s.ref_source == "tabla"
    assert score("def schedule(n):\n    return [1.4142, 1.8768]\n", Instance(2)).reward == pytest.approx(0.0, abs=1e-3)
    assert score(CONST, Instance(3)).reward > score("def schedule(n):\n    return [0.1]*n\n", Instance(3)).reward
    assert score("def schedule(n):\n    return [1.0]*(n+1)\n", Instance(3)).reward == INVALID_REWARD


def test_reference_fallbacks():
    tau, src = reference(Instance(4), refs={})
    assert src == "tabla" and tau == 0.031170
    tau, src = reference(Instance(4, mu=0.1), refs={})
    assert src == "constante" and tau > 0
    tau, src = reference(Instance(4), refs={"fval|mu=0|n=4": {"tau": 0.03, "source": "ce"}})
    assert src == "ce" and tau == 0.03  # una referencia mejor que la tabla gana


def test_improvement_gate_certifies():
    s = score(H15, Instance(4, mu=0.1), refs={})  # frente a la referencia débil (paso 1) hay mejora real
    assert s.ok and s.improved and s.certified and s.reward > GATE and s.tau_cert >= s.tau - 1e-9
    assert s.cert_json is not None and s.cert_json["ok"]
    s2 = score(H15, Instance(4, mu=0.1), refs={}, certify_improvements=False)
    assert s2.improved and s2.certified is None and s2.reward > GATE


def test_score_batch_parallel_and_cache(tmp_path):
    cache = TauCache(tmp_path / "tau.json")
    codes = [SILVER, CONST, "import os\ndef schedule(n):\n    return [1]*n\n", H15] * 3
    insts = [Instance(3), Instance(5), Instance(7), Instance(4, objective="gradnorm")] * 3
    sc = score_batch(codes, insts, workers=2, cache=cache, refs={}, cert_dir=tmp_path / "certs")
    assert len(sc) == 12 and sum(s.ok for s in sc) == 9 and (tmp_path / "tau.json").exists()
    assert len(cache) == 3  # tres sucesiones distintas → tres SDP
    assert sc[0].reward == pytest.approx(score(SILVER, Instance(3)).reward, abs=1e-9)
    assert sc[3].improved and sc[3].certified and list((tmp_path / "certs").glob("*.json"))
    sc2 = score_batch(codes, insts, workers=2, cache=cache, refs={})
    assert [s.reward for s in sc2] == pytest.approx([s.reward for s in sc])


def test_families():
    assert len(FAMILY_TRAIN) == 60 and len(FAMILY_TEST) == 12
    assert all(i.n <= 12 for i in FAMILY_TRAIN) and all(i.n >= 16 for i in FAMILY_TEST)
    assert family(ns=(3,), mus=(0.0,), objectives=("fval",), allow_negative=True)[0].label().endswith("|neg")
