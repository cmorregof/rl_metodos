"""Agente Q-learning tabular con exploración ε-greedy más un bono de
curiosidad tipo UCB (acciones poco probadas parecen más atractivas).

Es deliberadamente simple: la gracia del proyecto es *ver* cómo una tabla de
valores, partiendo de cero, converge al método de bisección y al orden lógico
de la demostración.
"""

from __future__ import annotations

import math
import random
from collections import defaultdict
from typing import Hashable, Sequence


class QAgent:
    def __init__(
        self,
        n_actions: int,
        alpha: float | None = 0.2,
        gamma: float = 0.97,
        eps_start: float = 1.0,
        eps_end: float = 0.03,
        eps_decay: float = 0.999,
        ucb_c: float = 0.0,
        seed: int = 0,
    ) -> None:
        self.ucb_c = ucb_c
        self.n_actions = n_actions
        self.alpha, self.gamma = alpha, gamma
        self.eps = eps_start
        self.eps_end, self.eps_decay = eps_end, eps_decay
        self.rng = random.Random(seed)
        self.q: dict[Hashable, list[float]] = defaultdict(lambda: [0.0] * n_actions)
        # alpha=None → tasa 1/N(s,a): promedio muestral, ideal para señales finas.
        self.counts: dict[Hashable, list[int]] = defaultdict(lambda: [0] * n_actions)
        self.updates = 0

    def act(self, state: Hashable, actions: Sequence[int] | None = None) -> int:
        actions = list(range(self.n_actions)) if actions is None else list(actions)
        if self.rng.random() < self.eps:
            return self.rng.choice(actions)
        if self.ucb_c > 0:
            qs, ns = self.q[state], self.counts[state]
            n_state = sum(ns) + 1
            score = {a: qs[a] + self.ucb_c * math.sqrt(math.log(n_state) / (ns[a] + 1)) for a in actions}
            best = max(score.values())
            return self.rng.choice([a for a in actions if score[a] == best])
        return self.greedy(state, actions)

    def greedy(self, state: Hashable, actions: Sequence[int] | None = None) -> int:
        actions = list(range(self.n_actions)) if actions is None else list(actions)
        qs = self.q[state]
        best = max(qs[a] for a in actions)
        ties = [a for a in actions if qs[a] == best]
        return self.rng.choice(ties)

    def learn(self, s: Hashable, a: int, r: float, s2: Hashable, done: bool) -> None:
        target = r if done else r + self.gamma * max(self.q[s2])
        self.counts[s][a] += 1
        alpha = self.alpha if self.alpha is not None else 1.0 / self.counts[s][a]
        self.q[s][a] += alpha * (target - self.q[s][a])
        self.updates += 1

    def decay(self) -> None:
        self.eps = max(self.eps_end, self.eps * self.eps_decay)
