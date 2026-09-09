"""Proveedores de LLM para la capa simbólica. La capa es agnóstica: cualquier
objeto con `.complete(system, user) -> str` sirve. Se incluyen:

* `OpenAIProvider`   – API de OpenAI (`OPENAI_API_KEY`), modelo configurable;
* `AnthropicProvider`– API de Anthropic (`ANTHROPIC_API_KEY`), por defecto
                       Claude Fable 5.1 con fallback de servidor;
* `MockProvider`     – devuelve programas fijos: para tests y ensayos en seco.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field


class ProviderError(RuntimeError):
    pass


@dataclass
class OpenAIProvider:
    model: str
    max_output_tokens: int = 16000
    reasoning_effort: str | None = None  # p. ej. "high" en modelos con razonamiento
    calls: int = 0

    def __post_init__(self) -> None:
        if not os.environ.get("OPENAI_API_KEY"):
            raise ProviderError("falta OPENAI_API_KEY en el entorno")
        from openai import OpenAI

        self._client = OpenAI()

    def complete(self, system: str, user: str) -> str:
        kwargs = {}
        if self.reasoning_effort:
            kwargs["reasoning_effort"] = self.reasoning_effort
        resp = self._client.chat.completions.create(
            model=self.model,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            max_completion_tokens=self.max_output_tokens,
            **kwargs,
        )
        self.calls += 1
        choice = resp.choices[0]
        if choice.finish_reason == "length":
            raise ProviderError("la respuesta se cortó por longitud")
        return choice.message.content or ""


@dataclass
class AnthropicProvider:
    model: str = "claude-fable-5-1"
    max_output_tokens: int = 16000
    effort: str = "high"  # low | medium | high | xhigh | max
    fallback_model: str = "claude-opus-4-8"
    calls: int = 0

    def __post_init__(self) -> None:
        import anthropic

        self._anthropic = anthropic
        self._client = anthropic.Anthropic()  # ANTHROPIC_API_KEY o perfil de `ant auth login`

    def complete(self, system: str, user: str) -> str:
        # Claude Fable 5.1: el thinking siempre está activo (no se pasa el parámetro); la
        # profundidad se controla con output_config.effort; fallback de servidor ante rechazo.
        try:
            resp = self._client.beta.messages.create(
                model=self.model,
                max_tokens=self.max_output_tokens,
                system=system,
                output_config={"effort": self.effort},
                betas=["server-side-fallback-2026-06-01"],
                fallbacks=[{"model": self.fallback_model}],
                messages=[{"role": "user", "content": user}],
            )
        except self._anthropic.RateLimitError as e:
            raise ProviderError(f"límite de tasa: reintentar en {e.response.headers.get('retry-after', '?')} s") from e
        except self._anthropic.APIStatusError as e:
            raise ProviderError(f"error de la API ({e.status_code}): {e.message}") from e
        except self._anthropic.APIConnectionError as e:
            raise ProviderError("error de red") from e
        self.calls += 1
        if resp.stop_reason == "refusal":
            raise ProviderError("la petición fue rechazada por el modelo y por el fallback")
        if resp.stop_reason == "max_tokens":
            raise ProviderError("la respuesta se cortó por max_tokens")
        return "".join(block.text for block in resp.content if block.type == "text")


@dataclass
class MockProvider:
    """Devuelve, por turnos, los programas de `scripts` (para tests y --dry-run)."""

    scripts: list[str] = field(default_factory=list)
    calls: int = 0

    def complete(self, system: str, user: str) -> str:
        i = min(self.calls, len(self.scripts) - 1)
        self.calls += 1
        return self.scripts[i] if self.scripts else ""


MOCK_SCRIPTS = [
    # 1: el silver schedule tal cual (baseline)
    '''```python
import math
RHO = 1 + math.sqrt(2)
def schedule(n):
    def v2(t):
        k = 0
        while t % 2 == 0:
            t //= 2; k += 1
        return k
    return [1 + RHO ** (v2(t) - 1) for t in range(1, n + 1)]
```''',
    # 2: variante con último paso largo (tipo Grimmer: un paso final grande a horizonte fijo)
    '''```python
import math
RHO = 1 + math.sqrt(2)
def schedule(n):
    def v2(t):
        k = 0
        while t % 2 == 0:
            t //= 2; k += 1
        return k
    h = [1 + RHO ** (v2(t) - 1) for t in range(1, n)]
    return h + [1.5 + 0.5 * math.log2(n + 1)]
```''',
]


def make_provider(name: str, model: str | None, **kw):
    if name == "openai":
        if not model:
            raise ProviderError("indica --model para OpenAI (p. ej. el identificador de GPT que uses)")
        return OpenAIProvider(model=model, **kw)
    if name == "anthropic":
        return AnthropicProvider(model=model or "claude-fable-5-1", **kw)
    if name == "mock":
        return MockProvider(scripts=list(MOCK_SCRIPTS))
    raise ProviderError(f"proveedor desconocido: {name}")
