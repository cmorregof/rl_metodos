"""Entorno de RL para *aprender a demostrar* el teorema de Taylor.

Estado: máscara de bits con los pasos ya establecidos.
Acción: índice del siguiente paso a afirmar.

Recompensas:
* −0.5 por cada paso válido (cada línea escrita cuesta: premia demostraciones
  cortas, p. ej. omitir el lema de Bolzano, que es correcto pero innecesario);
* −2 por cada paso inválido ("salto lógico": dependencias sin establecer,
  repetición o distractor); el episodio continúa;
* +20 al alcanzar QED;
* el episodio termina al llegar a QED o al agotar `max_len` intentos.

Con esto el agente descubre el orden lógico: hipótesis → polinomio y resto →
función auxiliar → Rolle iterado → resto de Lagrange → cota → convergencia → ∎.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .proof_kb import DEPS_MASK, N_STEPS, QED_INDEX, STEPS

REWARD_VALID = -0.5
REWARD_INVALID = -2.0
REWARD_QED = 20.0


@dataclass
class ProofEnv:
    max_len: int = 30
    mask: int = field(init=False, default=0)
    attempts: list[tuple[int, bool]] = field(init=False, default_factory=list)

    def reset(self) -> int:
        self.mask = 0
        self.attempts = []
        return self.mask

    def is_valid(self, action: int, mask: int | None = None) -> bool:
        mask = self.mask if mask is None else mask
        step = STEPS[action]
        if step.distractor or (mask >> action) & 1:
            return False
        return (DEPS_MASK[action] & mask) == DEPS_MASK[action]

    def valid_actions(self, mask: int | None = None) -> list[int]:
        return [i for i in range(N_STEPS) if self.is_valid(i, mask)]

    def step(self, action: int) -> tuple[int, float, bool, str]:
        ok = self.is_valid(action)
        self.attempts.append((action, ok))
        if not ok:
            done = len(self.attempts) >= self.max_len
            return self.mask, REWARD_INVALID, done, "invalid"
        self.mask |= 1 << action
        if action == QED_INDEX:
            return self.mask, REWARD_VALID + REWARD_QED, True, "qed"
        done = len(self.attempts) >= self.max_len
        return self.mask, REWARD_VALID, done, "valid"

    @property
    def proof_keys(self) -> list[str]:
        return [STEPS[a].key for a, ok in self.attempts if ok]

    @property
    def n_invalid(self) -> int:
        return sum(1 for _, ok in self.attempts if not ok)

    @property
    def finished(self) -> bool:
        return (self.mask >> QED_INDEX) & 1 == 1
