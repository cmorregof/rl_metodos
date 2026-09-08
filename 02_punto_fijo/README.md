# 02 · FixpointRL — iteración de punto fijo

**Un agente de Reinforcement Learning aprende, en vivo en la terminal, a controlar la iteración de punto fijo `x ← g(x)` y a demostrar el teorema del punto fijo de Banach.**

Segundo proyecto de la serie [rl_metodos](../README.md). Mismo formato que [01 · bisección](../01_biseccion/README.md): dos fases, interfaz en vivo con cronómetro y bitácora, y una síntesis final de cómo aprendió.

```
python -m fixpointrl
```

![captura](docs/captura.svg)

## Qué aprende el agente

### Fase 1 · controlar la iteración

Para resolver `f(x) = 0` se usa el mapa de relajación `g_α(x) = x − α·f(x)`. La raíz es punto fijo para cualquier `α`, pero la iteración solo converge si `g_α` es una contracción cerca de la raíz: `|g_α'(r)| = |1 − α·f'(r)| < 1`. El agente **no conoce `f'` ni el teorema**. Solo observa, tras cada iteración, dos cosas medibles:

* la razón de residuos `ρ = |f(xₙ₊₁)| / |f(xₙ)|`, que cerca de la raíz es la constante de contracción empírica `|g_α'(r)|`;
* si el residuo cambia de signo (la iteración **oscila** alrededor de la raíz) o no (**monótona**).

Con esos 12 estados (6 cubos de `ρ` × 2 formas) decide una de cuatro acciones: mantener `α`, doblarlo, reducirlo a la mitad o invertir su signo. Recompensa: `−1 + log₁₀(|f(xₙ)|/|f(xₙ₊₁)|)` por iteración (shaping basado en potencial con `Φ = −log₁₀|f|`, que no depende de `α` y por tanto no se puede engañar cambiando el paso), `−10` si diverge, fin cuando `|f(x)| < tol`.

Lo que emerge es la **ley de control que dicta el teorema de Banach**, y coincide con la teoría en los 12 estados:

| observación | significado | acción aprendida |
|---|---|---|
| `ρ > 1`, monótona | `α·f' < 0`: el signo de `α` está mal | invertir signo |
| `ρ > 1`, oscila | `α·f' > 2`: `α` demasiado grande | reducir |
| `ρ < 1`, oscila | `1 < α·f' < 2`: contractivo pero pasado | reducir |
| `ρ ∈ (0.5, 1)`, monótona | `α·f'` pequeño: contractivo pero lento | doblar |
| `ρ < 0.25` | casi Newton (`α ≈ 1/f'`) | mantener |

Un `α` fijo converge en ~35 % de los problemas; el agente, en ~90–96 % y con ~11 iteraciones de media.

### Fase 2 · demostrar el teorema

**Teorema.** Sea `g` continua en `[a, b]` con `g([a, b]) ⊆ [a, b]` y `|g(x) − g(y)| ≤ k·|x − y|`, `0 ≤ k < 1`. Entonces `g` tiene un único punto fijo `p`, la iteración `xₙ₊₁ = g(xₙ)` converge a `p` para todo `x₀ ∈ [a, b]`, y
`|xₙ − p| ≤ kⁿ·max(x₀ − a, b − x₀)`, `|xₙ − p| ≤ kⁿ/(1 − k)·|x₁ − x₀|`.

Grafo de 13 pasos (hipótesis, función auxiliar, existencia vía Bolzano, unicidad, definición, contracción del error, decaimiento geométrico, convergencia, cota a priori, pasos consecutivos, cota a posteriori, ∎) más 7 **distractores** con errores clásicos: creer que `|g'(p)| < 1` basta para cualquier `x₀`, que `k = 1` es suficiente, que una sucesión acotada converge, que `g` creciente basta, el non sequitur «existe punto fijo, luego la iteración converge», Newton y derivabilidad.

Estado: conjunto de pasos establecidos. Acción: siguiente paso. Recompensa: `−0.5` por línea válida, `−2` por salto lógico, `+20` en ∎. Q-learning tabular con ε-greedy y bono UCB.

## Interfaz en vivo

| panel | qué muestra |
|---|---|
| **Fase 1** | `f`, la raíz ◆ y los últimos iterados ○● sobre el eje; debajo, `|f(xₙ)|` en escala log con el `α` usado en cada iteración |
| **Ley de control** | los 12 estados con la acción aprendida, la teórica y ✔/✘, los valores Q y el éxito reciente |
| **Fase 2** | el intento de demostración en curso, pasos válidos en verde y saltos lógicos tachados en rojo |
| **Lo que el agente cree ahora** | la demostración de la política voraz en este momento |
| **Marcador / Bitácora** | cronómetro, generación, ε, convergencias, divergencias, demostraciones e hitos con tiempo |

Al terminar guarda en `runs/<fecha>/` el `informe.md` (síntesis en prosa, demostración final, ley de control, orden de asentamiento de cada paso, saltos lógicos, hitos), `demostracion.md` e `historia.json`.

## Uso

```bash
cd 02_punto_fijo
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
python -m fixpointrl              # ~1–2 min con animación
python -m fixpointrl --speed 3    # más rápido
python -m fixpointrl --no-tui     # solo entrena y escribe el informe
pytest -q
```

Requiere Python ≥ 3.10 y una terminal de al menos 120 × 40.

## Referencias

* Burden y Faires. *Análisis numérico*, §2.2 (iteración de punto fijo).
* Banach (1922). *Sur les opérations dans les ensembles abstraits et leur application aux équations intégrales.*
* Ng, Harada y Russell (1999). *Policy invariance under reward transformations.* ICML.
