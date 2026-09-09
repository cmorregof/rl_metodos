"""Bucle de entrenamiento de las dos fases y registro de hitos (generador)."""

from __future__ import annotations

import time
from collections import Counter, deque
from dataclasses import dataclass, field
from statistics import median
from typing import Iterator

from .agent import QAgent
from .env_secant import (
    ACTION_SHORT,
    BISECT,
    FALSI,
    N_ACTIONS,
    PHI,
    RATIO_LABELS,
    SEC,
    SecantEnv,
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
    episodes_secant: int = 3000
    episodes_proof: int = 3000
    seed: int = 0
    tol: float = 1e-9


ALL_STATES = [(b, i) for i in (1, 0) for b in range(4)]


def order_summary(orders) -> float:
    """Mediana del orden observado (robusta: familias con f'' (r) = 0 disparan la media)."""
    xs = [o for o in orders if o == o]
    return median(xs) if xs else float("nan")


@dataclass
class Trainer:
    cfg: Config = field(default_factory=Config)
    t0: float = field(default_factory=time.perf_counter)
    phase: int = 1
    milestones: list[Milestone] = field(default_factory=list)
    log: list[str] = field(default_factory=list)

    # ---- Fase 1 --------------------------------------------------------
    env1: SecantEnv = field(init=False)
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
    last_action1: int = SEC
    action_usage: Counter = field(default_factory=Counter)
    rf_total: deque = field(default_factory=lambda: deque(maxlen=100))  # pasos de regula falsi por episodio
    rf_chained: deque = field(default_factory=lambda: deque(maxlen=100))  # de ellos, los que siguen a otro paso de regula falsi
    orders: deque = field(default_factory=lambda: deque(maxlen=100))  # orden de convergencia observado
    orders_by_family: dict[str, list[float]] = field(default_factory=dict)

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
        self.env1 = SecantEnv(tol=self.cfg.tol, seed=self.cfg.seed)
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

    def visits(self, s: tuple) -> int:
        return sum(self.agent1.counts.get(s, [0]))

    def law_matches_theory(self) -> tuple[int, int]:
        """(estados bien visitados en que la política coincide con la teoría, estados bien visitados)."""
        ok = n = 0
        for s, a in self.control_law().items():
            if a is None or self.visits(s) < 20:
                continue
            n += 1
            ok += a in theoretical_action(*s)
        return ok, n

    def law_text(self, s: tuple[int, int], pad: bool = True) -> str:
        b, i = s
        label = f"{RATIO_LABELS[b]:<9}" if pad else RATIO_LABELS[b]
        return f"{label} secante {'dentro' if i else ('fuera ' if pad else 'fuera')}"

    def order_median(self) -> float:
        return order_summary(self.orders)

    def rf_chain_fraction(self) -> float:
        """Fracción de pasos de regula falsi (últimos 100 episodios) que siguen a otro paso de regula falsi."""
        n = sum(self.rf_total)
        return sum(self.rf_chained) / n if n else float("nan")

    def refuge_text(self) -> str:
        """Qué hace el agente cuando la secante se sale del corchete (estados bien visitados)."""
        law = self.control_law()
        acts = {law[(b, 0)] for b in range(4) if law[(b, 0)] is not None and self.visits((b, 0)) >= 20}
        if acts == {BISECT}:
            return "bisecar"
        if acts == {FALSI}:
            return "un paso de regula falsi"
        return "bisecar o un paso de regula falsi según el residuo"

    # ---- fase 1 ------------------------------------------------------------
    def run_phase1(self) -> Iterator[str]:
        env, ag = self.env1, self.agent1
        got_first = got_sec = got_safe = got_law = got_95 = got_phi = got_dev = False
        prev_law: dict = {}
        stable = 0
        dev_streak = 0
        for ep in range(1, self.cfg.episodes_secant + 1):
            self.ep1 = ep
            s = env.reset()
            done = False
            prev_a = -1
            n_rf = n_rf_chain = 0
            while not done:
                a = ag.act(s)
                self.last_action1 = a
                self.action_usage[a] += 1
                if a == FALSI:
                    n_rf += 1
                    n_rf_chain += prev_a == FALSI
                prev_a = a
                s2, r, done, info = env.step(a)
                ag.learn(s, a, r, s2, done)
                s = s2
                self.last_outcome1 = info
                if ep <= 40 or ep % 200 == 0:
                    yield "iter"
            ag.decay()
            self.rf_total.append(n_rf)
            self.rf_chained.append(n_rf_chain)
            self.outcomes1.append(info)
            self.iters1.append(env.n_evals)
            self.success_curve.append(sum(1 for o in self.outcomes1 if o == "converged") / len(self.outcomes1))
            conv = [n for n, o in zip(self.iters1, self.outcomes1) if o == "converged"]
            self.iter_curve.append(sum(conv) / len(conv) if conv else float("nan"))
            if info == "converged":
                self.converged_total += 1
                order = env.observed_order()
                if order == order and 0 < order < 5:
                    self.orders.append(order)
                    self.orders_by_family.setdefault(env.problem.family, []).append(order)
                if not got_first:
                    got_first = True
                    self.mark(f"primera convergencia: {env.problem.name} en [{env.problem.lo:.2f}, {env.problem.hi:.2f}] con {env.n_evals} evaluaciones")
                med = self.order_median()
                if not got_phi and len(self.orders) >= 30 and 1.5 <= med <= 1.75:
                    got_phi = True
                    self.mark(f"orden de convergencia observado ≈ {med:.2f} (mediana): φ = (1+√5)/2 ≈ {PHI:.3f}, superlineal pero no cuadrático")
            elif info == "timeout":
                self.timeout_total += 1
            else:
                self.diverged_total += 1
            law = self.control_law()
            if not got_sec and law.get((0, 1)) == SEC and law.get((1, 1)) == SEC and min(self.visits((0, 1)), self.visits((1, 1))) >= 50:
                got_sec = True
                self.mark("aprendió que cuando la secante cae dentro del corchete y el residuo baja, el paso de secante es el mejor (régimen superlineal)")
            if not got_safe and law.get((2, 0)) in (BISECT, FALSI) and law.get((3, 0)) in (BISECT, FALSI) and self.visits((2, 0)) >= 50 and self.visits((3, 0)) >= 30:
                got_safe = True
                self.mark(f"aprendió a no fiarse de la secante cuando se sale del corchete y volver a él ({self.refuge_text()}): método de Dekker")
            well = {s_: a for s_, a in law.items() if a is not None and self.visits(s_) >= 100}
            stable = stable + 1 if (well and well == prev_law) else 0
            prev_law = well
            ok, n = self.law_matches_theory()
            if not got_law and stable >= 200 and len(well) >= 5:
                got_law = True
                devs = [f"{self.law_text(s_, pad=False)} → {ACTION_SHORT[a]}" for s_, a in well.items() if a not in theoretical_action(*s_)]
                if devs:
                    self.mark(f"ley estable 200 gen.: coincide con Dekker en {ok}/{n} estados; discrepa en: " + "; ".join(devs))
                else:
                    self.mark(f"ley estable 200 gen.: coincide con Dekker en los {n} estados bien visitados")
            dev_streak = dev_streak + 1 if any(a == SEC and s_[0] >= 2 and s_[1] == 1 for s_, a in well.items()) else 0
            if not got_dev and dev_streak >= 300:
                got_dev = True
                self.mark("matiz que Dekker no contempla: si la secante cae dentro del corchete, seguirla aunque el residuo apenas baje sale más barato que bisecar (bisecar tira el punto actual)")
            if not got_95 and len(self.outcomes1) == 100 and self.success_curve[-1] >= 0.95:
                got_95 = True
                self.mark("≥95 % de convergencia en las últimas 100 generaciones (secante pura: ~69 %, regula falsi: ~63 %)")
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
