"""Bucle de entrenamiento de las dos fases y registro de hitos (generador)."""

from __future__ import annotations

import time
from collections import Counter, deque
from dataclasses import dataclass, field
from typing import Iterator

from .agent import QAgent
from .env_fixpoint import (
    ACTION_SHORT,
    DOUBLE,
    FLIP,
    HALVE,
    KEEP,
    N_ACTIONS,
    RATIO_LABELS,
    FixpointEnv,
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
    episodes_fixpoint: int = 3000
    episodes_proof: int = 3000
    seed: int = 0
    tol: float = 1e-6


ALL_STATES = [(b, o) for b in range(6) for o in (0, 1)]


@dataclass
class Trainer:
    cfg: Config = field(default_factory=Config)
    t0: float = field(default_factory=time.perf_counter)
    phase: int = 1
    milestones: list[Milestone] = field(default_factory=list)
    log: list[str] = field(default_factory=list)

    # ---- Fase 1 --------------------------------------------------------
    env1: FixpointEnv = field(init=False)
    agent1: QAgent = field(init=False)
    ep1: int = 0
    outcomes1: deque = field(default_factory=lambda: deque(maxlen=100))
    iters1: deque = field(default_factory=lambda: deque(maxlen=100))
    success_curve: list[float] = field(default_factory=list)
    iter_curve: list[float] = field(default_factory=list)
    diverged_total: int = 0
    converged_total: int = 0
    timeout_total: int = 0
    phase1_time: float = 0.0
    last_outcome1: str = ""
    last_action1: int = KEEP
    law_streak: int = 0

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
        self.env1 = FixpointEnv(tol=self.cfg.tol, seed=self.cfg.seed)
        self.agent1 = QAgent(N_ACTIONS, alpha=0.1, gamma=0.9, eps_start=1.0, eps_decay=0.995, eps_end=0.02, ucb_c=0.5, seed=self.cfg.seed)
        self.env2 = ProofEnv()
        self.agent2 = QAgent(N_STEPS, alpha=0.3, gamma=0.97, eps_start=1.0, eps_decay=0.997, eps_end=0.02, ucb_c=1.0, seed=self.cfg.seed + 1)

    @property
    def elapsed(self) -> float:
        return time.perf_counter() - self.t0

    def mark(self, text: str) -> None:
        ep = self.ep1 if self.phase == 1 else self.ep2
        self.milestones.append(Milestone(self.phase, ep, self.elapsed, text))
        self.log.append(f"[{self.elapsed:6.1f}s] F{self.phase} gen {ep:>5}: {text}")

    # ---- ley de control aprendida ----------------------------------------
    def control_law(self) -> dict[tuple[int, int], int | None]:
        out: dict[tuple[int, int], int | None] = {}
        for s in ALL_STATES:
            qs = self.agent1.q.get(s)
            out[s] = None if qs is None or all(q == 0.0 for q in qs) else max(range(N_ACTIONS), key=qs.__getitem__)
        return out

    def law_matches_theory(self) -> tuple[int, int]:
        """(estados en que la política coincide con la teoría, estados evaluados)."""
        ok = n = 0
        for (b, o), a in self.control_law().items():
            if a is None:
                continue
            n += 1
            ok += a in theoretical_action(b, o)
        return ok, n

    def law_text(self, s: tuple[int, int]) -> str:
        b, o = s
        return f"ρ {RATIO_LABELS[b]:<9} {'oscila' if o else 'monót.'}"

    # ---- fase 1 ------------------------------------------------------------
    def run_phase1(self) -> Iterator[str]:
        env, ag = self.env1, self.agent1
        got_first = got_flip = got_halve = got_law = got_95 = False
        for ep in range(1, self.cfg.episodes_fixpoint + 1):
            self.ep1 = ep
            s = env.reset()
            done = False
            while not done:
                a = ag.act(s)
                self.last_action1 = a
                s2, r, done, info = env.step(a)
                ag.learn(s, a, r, s2, done)
                s = s2
                self.last_outcome1 = info
                if ep <= 40 or ep % 200 == 0:
                    yield "iter"
            ag.decay()
            self.outcomes1.append(info)
            self.iters1.append(env.n_iter)
            self.success_curve.append(sum(1 for o in self.outcomes1 if o == "converged") / len(self.outcomes1))
            conv = [n for n, o in zip(self.iters1, self.outcomes1) if o == "converged"]
            self.iter_curve.append(sum(conv) / len(conv) if conv else float("nan"))
            if info == "diverged":
                self.diverged_total += 1
            elif info == "timeout":
                self.timeout_total += 1
            else:
                self.converged_total += 1
                if not got_first:
                    got_first = True
                    self.mark(f"primera convergencia ({env.n_iter} iteraciones, f = {env.problem.name})")
            law = self.control_law()
            if not got_flip and law.get((4, 0)) == FLIP and law.get((5, 0)) in (FLIP, HALVE):
                got_flip = True
                self.mark("aprendió que si diverge monótonamente el signo de α está mal: invertirlo")
            if not got_halve and law.get((3, 1)) == HALVE and law.get((4, 1)) == HALVE and law.get((5, 1)) == HALVE:
                got_halve = True
                self.mark("aprendió que si oscila α es demasiado grande: reducirlo (|g'| < 1 es lo que importa)")
            ok, n = self.law_matches_theory()
            self.law_streak = self.law_streak + 1 if (n == 12 and ok == 12) else 0
            if not got_law and self.law_streak >= 100:
                got_law = True
                self.mark("su ley de control coincide con la teoría de Banach en los 12 estados (estable 100 gen.)")
            if not got_95 and len(self.outcomes1) == 100 and self.success_curve[-1] >= 0.95:
                got_95 = True
                self.mark("≥95 % de convergencia en las últimas 100 generaciones")
            yield "episode"
        self.phase1_time = self.elapsed
        self.mark(f"fase 1 terminada: {self.converged_total} convergencias, {self.diverged_total} divergencias, {self.timeout_total} agotadas")

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
