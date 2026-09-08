"""Bucle de entrenamiento de las dos fases y registro de hitos.

El entrenador es un generador: cede el control después de cada "cuadro" para
que la interfaz pueda redibujar. Toda la información que la interfaz muestra
vive en este objeto (métricas, hitos, mejor demostración, política aprendida).
"""

from __future__ import annotations

import time
from collections import Counter, deque
from dataclasses import dataclass, field
from typing import Iterator

from .agent import QAgent
from .env_bisection import KEEP_LEFT, KEEP_RIGHT, LAMBDAS, BisectionEnv
from .env_proof import ProofEnv
from .proof_kb import MIN_PROOF_LENGTH, N_STEPS, QED_INDEX, REQUIRED_KEYS, STEPS


@dataclass
class Milestone:
    phase: int
    episode: int
    t: float
    text: str


@dataclass
class Config:
    episodes_bisection: int = 3000
    episodes_proof: int = 3000
    seed: int = 0
    tol: float = 1e-3


@dataclass
class Trainer:
    cfg: Config = field(default_factory=Config)
    t0: float = field(default_factory=time.perf_counter)
    phase: int = 1
    milestones: list[Milestone] = field(default_factory=list)
    log: list[str] = field(default_factory=list)

    # ---- Fase 1 --------------------------------------------------------
    env1: BisectionEnv = field(init=False)
    agent_split: QAgent = field(init=False)  # cabeza de corte: ¿dónde evaluar f?
    agent_keep: QAgent = field(init=False)   # cabeza de conservación: ¿qué lado guardar?
    ep1: int = 0
    outcomes1: deque = field(default_factory=lambda: deque(maxlen=100))
    steps1: deque = field(default_factory=lambda: deque(maxlen=100))
    lam_hist: deque = field(default_factory=lambda: deque(maxlen=300))
    ratio_curve: list[float] = field(default_factory=list)  # pasos / óptimo, por episodio
    success_curve: list[float] = field(default_factory=list)
    lost_total: int = 0
    converged_total: int = 0
    phase1_time: float = 0.0
    last_outcome1: str = ""
    current_lam: float = 0.5

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
    first_valid: dict[str, int] = field(default_factory=dict)   # paso -> primera generación en que se usó válidamente
    first_greedy: dict[str, int] = field(default_factory=dict)  # paso -> primera generación en que la política voraz lo incluye
    step_usage: Counter = field(default_factory=Counter)
    invalid_usage: Counter = field(default_factory=Counter)  # saltos lógicos por paso
    qed_total: int = 0
    minimal_total: int = 0
    phase2_time: float = 0.0
    done: bool = False

    def __post_init__(self) -> None:
        self.env1 = BisectionEnv(tol=self.cfg.tol, seed=self.cfg.seed)
        # Gracias al shaping potencial la recompensa de cada iteración ya mide
        # el progreso, así que ambas cabezas aprenden con γ = 0 y tasa 1/N.
        self.agent_split = QAgent(len(LAMBDAS), alpha=None, gamma=0.0, eps_start=1.0, eps_decay=0.998, eps_end=0.2, ucb_c=0.2, seed=self.cfg.seed)
        self.agent_keep = QAgent(2, alpha=None, gamma=0.0, eps_start=0.5, eps_decay=0.99, eps_end=0.0, ucb_c=0.5, seed=self.cfg.seed + 7)
        self.env2 = ProofEnv()
        self.agent2 = QAgent(N_STEPS, alpha=0.3, gamma=0.97, eps_start=1.0, eps_decay=0.997, eps_end=0.02, ucb_c=1.0, seed=self.cfg.seed + 1)

    # ---- utilidades ------------------------------------------------------
    @property
    def elapsed(self) -> float:
        return time.perf_counter() - self.t0

    def mark(self, text: str) -> None:
        ep = self.ep1 if self.phase == 1 else self.ep2
        self.milestones.append(Milestone(self.phase, ep, self.elapsed, text))
        self.log.append(f"[{self.elapsed:6.1f}s] F{self.phase} gen {ep:>5}: {text}")

    # ---- política aprendida en fase 1 ------------------------------------
    def lambda_q(self) -> list[float]:
        return list(self.agent_split.q[("split",)])

    def lambda_policy(self) -> float:
        qs = self.lambda_q()
        return LAMBDAS[max(range(len(LAMBDAS)), key=qs.__getitem__)]

    def keep_policy(self) -> dict[tuple[int, int, int], str]:
        """Regla de conservación aprendida, agregada sobre λ por voto mayoritario."""
        votes: dict[tuple[int, int, int], Counter] = {}
        for s, qs in self.agent_keep.q.items():
            if s[0] == "keep" and any(q != 0.0 for q in qs[:2]):
                k = "izq" if qs[KEEP_LEFT] >= qs[KEEP_RIGHT] else "der"
                votes.setdefault((s[1], s[2], s[3]), Counter())[k] += 1
        return {k: v.most_common(1)[0][0] for k, v in votes.items()}

    def keep_policy_is_bolzano(self) -> bool:
        """¿La regla aprendida conserva siempre el cambio de signo?"""
        pol = self.keep_policy()
        if len(pol) < 4:
            return False
        for (sa, sx, sb), k in pol.items():
            if sx == 0:
                continue
            correct = "izq" if sa * sx < 0 else "der"
            if k != correct:
                return False
        return True

    half_streak: int = 0

    def lambda_policy_is_half(self) -> bool:
        """λ = 1/2 ha sido la mejor opción durante 300 generaciones seguidas."""
        if abs(self.lambda_policy() - 0.5) < 1e-9:
            self.half_streak += 1
        else:
            self.half_streak = 0
        return self.half_streak >= 300

    # ---- fase 1 ------------------------------------------------------------
    def run_phase1(self) -> Iterator[str]:
        env, ag_s, ag_k = self.env1, self.agent_split, self.agent_keep
        streak = 0
        got_half = got_bolz = got_95 = got_first = False
        for ep in range(1, self.cfg.episodes_bisection + 1):
            self.ep1 = ep
            s = env.reset()
            done = False
            total = 0.0
            while not done:
                a_lam = ag_s.act(s)
                self.current_lam = LAMBDAS[a_lam]
                self.lam_hist.append(self.current_lam)
                s_keep = env.step_split(a_lam)
                a_keep = ag_k.act(s_keep)
                s2, r, done, info = env.step_keep(a_keep)
                ag_k.learn(s_keep, a_keep, r, s2, done)
                if info != "lost":
                    # perder la raíz es culpa de la cabeza de conservación, no del corte
                    ag_s.learn(s, a_lam, r, s2, done)
                total += r
                s = s2
                self.last_outcome1 = info
                if ep <= 40 or ep % 200 == 0:
                    yield "iter"
            ag_s.decay()
            ag_k.decay()
            self.outcomes1.append(info)
            self.steps1.append(env.n_iter)
            self.ratio_curve.append(env.n_iter / env.optimal_steps() if info == "converged" else 0.0)
            self.success_curve.append(sum(1 for o in self.outcomes1 if o == "converged") / len(self.outcomes1))
            if info == "lost":
                self.lost_total += 1
                streak = 0
            else:
                streak += 1
            if info == "converged":
                self.converged_total += 1
                if not got_first:
                    got_first = True
                    self.mark(f"primera convergencia ({env.n_iter} iteraciones, f = {env.problem.name})")
            if streak == 25 and not got_bolz and self.keep_policy_is_bolzano():
                got_bolz = True
                self.mark("aprendió el invariante de Bolzano: siempre conserva el subintervalo con cambio de signo")
            if not got_half and self.lambda_policy_is_half():
                got_half = True
                self.mark("descubrió el corte óptimo λ = 1/2: esto ES el método de bisección")
            if not got_95 and len(self.outcomes1) == 100 and self.success_curve[-1] >= 0.95:
                got_95 = True
                self.mark("≥95 % de convergencia en las últimas 100 generaciones")
            yield "episode"
        self.phase1_time = self.elapsed
        self.mark(f"fase 1 terminada: {self.converged_total} convergencias, {self.lost_total} raíces perdidas")

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
