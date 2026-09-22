"""Bucle de entrenamiento de las dos fases de RL y registro de hitos (generador).

La fase 3 (verificador + búsqueda de nodos) no es RL tabular y vive en
`search.py`; el informe la incluye si se pidió con `--search`.
"""

from __future__ import annotations

import time
from collections import Counter, deque
from dataclasses import dataclass, field
from typing import Iterator

from .agent import QAgent
from .env_interp import (
    ACTION_SHORT,
    ADD,
    EST_LABELS,
    GAMMAS,
    N_ACTIONS,
    STOP,
    TREND_LABELS,
    InterpEnv,
    theoretical_action,
)
from .env_proof import ProofEnv
from .proof_kb import MIN_PROOF_LENGTH, N_STEPS, STEPS


@dataclass
class Milestone:
    phase: int
    episode: int
    t: float
    text: str


@dataclass
class Config:
    episodes_interp: int = 3000
    episodes_proof: int = 3000
    seed: int = 0
    tol: float = 1e-5


ALL_STATES = [(eb, trend, late) for late in (0, 1) for eb in range(4) for trend in (0, 1)]


@dataclass
class Trainer:
    cfg: Config = field(default_factory=Config)
    t0: float = field(default_factory=time.perf_counter)
    phase: int = 1
    milestones: list[Milestone] = field(default_factory=list)
    log: list[str] = field(default_factory=list)

    # ---- Fase 1 --------------------------------------------------------
    env1: InterpEnv = field(init=False)
    agent_gamma: QAgent = field(init=False)  # cabeza de nodos: ¿qué familia?
    agent_stop: QAgent = field(init=False)   # cabeza de parada: ¿otro nodo o entregar?
    ep1: int = 0
    outcomes1: deque = field(default_factory=lambda: deque(maxlen=100))
    nodes1: deque = field(default_factory=lambda: deque(maxlen=100))
    gamma_hist: deque = field(default_factory=lambda: deque(maxlen=300))
    success_curve: list[float] = field(default_factory=list)
    node_curve: list[float] = field(default_factory=list)
    stop_ok_total: int = 0
    stop_bad_total: int = 0
    timeout_total: int = 0
    phase1_time: float = 0.0
    last_outcome1: str = ""
    last_action1: int = ADD
    law_streak: int = 0
    cheb_streak: int = 0

    # ---- Fase 2 --------------------------------------------------------
    env2: ProofEnv = field(init=False)
    agent2: QAgent = field(init=False)
    ep2: int = 0
    rewards2: deque = field(default_factory=lambda: deque(maxlen=100))
    finished2: deque = field(default_factory=lambda: deque(maxlen=100))
    reward_curve: list[float] = field(default_factory=list)
    best_reward: float = float("-inf")
    best_proof: list[tuple[int, bool]] = field(default_factory=list)
    best_episode: int = 0
    greedy_proof: list[tuple[int, bool]] = field(default_factory=list)
    first_valid: dict[str, int] = field(default_factory=dict)
    first_greedy: dict[str, int] = field(default_factory=dict)
    step_usage: Counter = field(default_factory=Counter)
    invalid_usage: Counter = field(default_factory=Counter)
    qed_total: int = 0
    minimal_total: int = 0
    phase2_time: float = 0.0
    done: bool = False

    def __post_init__(self) -> None:
        self.env1 = InterpEnv(tol=self.cfg.tol, seed=self.cfg.seed)
        # bandido sobre γ con tasa constante: el retorno de una familia cambia a medida que la
        # cabeza de parada mejora (no estacionario), así que el promedio 1/N se quedaría anclado
        # en los episodios malos del principio; con α = 0.05 olvida. UCB alto para no fijarse pronto.
        self.agent_gamma = QAgent(len(GAMMAS), alpha=0.05, gamma=0.0, eps_start=1.0, eps_decay=0.995, eps_end=0.05, ucb_c=2.0, seed=self.cfg.seed)
        self.agent_stop = QAgent(N_ACTIONS, alpha=0.1, gamma=0.95, eps_start=1.0, eps_decay=0.995, eps_end=0.02, ucb_c=0.5, seed=self.cfg.seed + 7)
        self.env2 = ProofEnv()
        self.agent2 = QAgent(N_STEPS, alpha=0.3, gamma=0.97, eps_start=1.0, eps_decay=0.997, eps_end=0.02, ucb_c=1.0, seed=self.cfg.seed + 1)

    @property
    def elapsed(self) -> float:
        return time.perf_counter() - self.t0

    def mark(self, text: str) -> None:
        ep = self.ep1 if self.phase == 1 else self.ep2
        self.milestones.append(Milestone(self.phase, ep, self.elapsed, text))
        self.log.append(f"[{self.elapsed:6.1f}s] F{self.phase} gen {ep:>5}: {text}")

    # ---- políticas aprendidas en fase 1 ----------------------------------
    def gamma_q(self) -> list[float]:
        return list(self.agent_gamma.q[("gamma",)])

    def gamma_policy(self) -> float:
        qs = self.gamma_q()
        return GAMMAS[max(range(len(GAMMAS)), key=qs.__getitem__)]

    def gamma_policy_is_chebyshev(self) -> bool:
        if abs(self.gamma_policy() - 1.0) < 1e-9 and sum(self.agent_gamma.counts[("gamma",)]) > 50:
            self.cheb_streak += 1
        else:
            self.cheb_streak = 0
        return self.cheb_streak >= 200

    def control_law(self) -> dict[tuple[int, int, int], int | None]:
        out: dict[tuple[int, int, int], int | None] = {}
        for s in ALL_STATES:
            qs = self.agent_stop.q.get(s)
            out[s] = None if qs is None or all(q == 0.0 for q in qs) else max(range(N_ACTIONS), key=qs.__getitem__)
        return out

    def law_matches_theory(self) -> tuple[int, int]:
        ok = n = 0
        for s, a in self.control_law().items():
            if a is None or sum(self.agent_stop.counts.get(s, [0])) < 20:
                continue
            n += 1
            ok += a in theoretical_action(*s)
        return ok, n

    def law_text(self, s: tuple[int, int, int]) -> str:
        eb, trend, late = s
        return f"{EST_LABELS[eb]:<12} {TREND_LABELS[trend]:<5} {'n>6' if late else 'n≤6'}"

    # ---- fase 1 ------------------------------------------------------------
    def run_phase1(self) -> Iterator[str]:
        env, ag_g, ag_s = self.env1, self.agent_gamma, self.agent_stop
        got_first = got_cheb = got_stop = got_law = got_95 = False
        for ep in range(1, self.cfg.episodes_interp + 1):
            self.ep1 = ep
            env.reset()
            a_g = ag_g.act(("gamma",))
            self.gamma_hist.append(GAMMAS[a_g])
            s = env.set_gamma(GAMMAS[a_g])
            done = False
            total = 0.0
            while not done:
                a = ag_s.act(s)
                self.last_action1 = a
                s2, r, done, info = env.step(a)
                ag_s.learn(s, a, r, s2, done)
                total += r
                s = s2
                self.last_outcome1 = info
                if ep <= 40 or ep % 200 == 0:
                    yield "iter"
            ag_g.learn(("gamma",), a_g, total, ("gamma",), True)
            ag_g.decay()
            ag_s.decay()
            self.outcomes1.append(info)
            self.nodes1.append(env.n)
            self.success_curve.append(sum(1 for o in self.outcomes1 if o == "stop_ok") / len(self.outcomes1))
            conv = [n for n, o in zip(self.nodes1, self.outcomes1) if o == "stop_ok"]
            self.node_curve.append(sum(conv) / len(conv) if conv else float("nan"))
            if info == "stop_ok":
                self.stop_ok_total += 1
                if not got_first:
                    got_first = True
                    self.mark(f"primera aproximación correcta: {env.problem.name} con n = {env.n} (γ = {env.gamma}); óptimo con Chebyshev n = {env.optimal_nodes()}")
            elif info == "stop_bad":
                self.stop_bad_total += 1
            else:
                self.timeout_total += 1
            if not got_cheb and self.gamma_policy_is_chebyshev():
                got_cheb = True
                qs = self.gamma_q()
                self.mark(f"prefiere γ = 1 de forma estable: nodos de Chebyshev (Q = {qs[-1]:+.1f} frente a {qs[0]:+.1f} con equiespaciados)")
            law = self.control_law()
            if not got_stop and law.get((0, 0, 1)) == STOP and law.get((2, 0, 1)) == ADD and law.get((3, 1, 1)) == ADD:
                got_stop = True
                self.mark("descubrió el criterio de parada: entregar cuando |pₙ − pₙ₋₁| ≪ tol y va bajando")
            ok, n = self.law_matches_theory()
            self.law_streak = self.law_streak + 1 if (n >= 8 and ok == n) else 0
            if not got_law and self.law_streak >= 100:
                got_law = True
                self.mark(f"su regla coincide con la teoría en los {n} estados visitados (estable 100 gen.)")
            if not got_95 and len(self.outcomes1) == 100 and self.success_curve[-1] >= 0.95:
                got_95 = True
                self.mark("≥95 % de aciertos en las últimas 100 generaciones")
            yield "episode"
        self.phase1_time = self.elapsed
        self.mark(f"fase 1 terminada: {self.stop_ok_total} aproximaciones correctas, {self.stop_bad_total} entregas prematuras, {self.timeout_total} agotadas")

    # ---- fase 2 ------------------------------------------------------------
    def greedy_rollout(self) -> list[tuple[int, bool]]:
        env = ProofEnv()
        s = env.reset()
        done = False
        while not done:
            a = self.agent2.greedy(s)
            s, _, done, _ = env.step(a)
        return list(env.attempts)

    def run_phase2(self) -> Iterator[str]:
        self.phase = 2
        env, ag = self.env2, self.agent2
        got_valid = got_qed = got_clean = got_min = got_greedy = False
        for ep in range(1, self.cfg.episodes_proof + 1):
            self.ep2 = ep
            s = env.reset()
            done = False
            total = 0.0
            while not done:
                a = ag.act(s)
                s2, r, done, info = env.step(a)
                ag.learn(s, a, r, s2, done)
                total += r
                s = s2
                if info == "invalid":
                    self.invalid_usage[STEPS[a].key] += 1
                if info == "valid" and STEPS[a].key not in self.first_valid:
                    self.first_valid[STEPS[a].key] = ep
                    if not got_valid:
                        got_valid = True
                        self.mark(f"primer paso válido: {STEPS[a].title} ({STEPS[a].key})")
                if ep <= 30 or ep % 250 == 0:
                    yield "iter"
            ag.decay()
            self.rewards2.append(total)
            self.reward_curve.append(total)
            self.finished2.append(env.finished)
            if env.finished:
                self.qed_total += 1
                for k in env.proof_keys:
                    self.step_usage[k] += 1
                clean = env.n_invalid == 0
                minimal = clean and len(env.proof_keys) == MIN_PROOF_LENGTH
                if minimal:
                    self.minimal_total += 1
                if not got_qed:
                    got_qed = True
                    self.mark(f"¡primera demostración completa! {len(env.proof_keys)} pasos, {env.n_invalid} saltos lógicos")
                if clean and not got_clean:
                    got_clean = True
                    self.mark("primera demostración sin ningún salto lógico")
                if minimal and not got_min:
                    got_min = True
                    self.mark(f"primera demostración mínima ({MIN_PROOF_LENGTH} pasos, sin desvíos)")
            if total > self.best_reward:
                self.best_reward = total
                self.best_proof = list(env.attempts)
                self.best_episode = ep
            if ep % 25 == 0 or ep <= 30:
                self.greedy_proof = self.greedy_rollout()
                g_keys = [STEPS[a].key for a, ok in self.greedy_proof if ok]
                g_clean = all(ok for _, ok in self.greedy_proof)
                for k in g_keys:
                    self.first_greedy.setdefault(k, ep)
                if not got_greedy and g_clean and "QED" in g_keys and len(g_keys) == MIN_PROOF_LENGTH:
                    got_greedy = True
                    self.mark("la política voraz (sin exploración) ya produce la demostración mínima")
            yield "episode"
        self.phase2_time = self.elapsed - self.phase1_time
        self.mark(f"fase 2 terminada: {self.qed_total} demostraciones completas, {self.minimal_total} mínimas")
        self.greedy_proof = self.greedy_rollout()
        self.done = True

    def run(self) -> Iterator[str]:
        yield from self.run_phase1()
        yield from self.run_phase2()
