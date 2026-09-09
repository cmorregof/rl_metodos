# 06 · StepRL — programas de pasos para el descenso de gradiente, con el PEP como juez

**Primer proyecto de la serie en la frontera: no hay grafo de lemas escrito a mano. Un verificador independiente (un programa semidefinido) certifica el peor caso de cada programa de pasos, y esa certificación es la única recompensa. La búsqueda la hacen una capa numérica (entropía cruzada) y una capa simbólica (un LLM propone programas que generan programas de pasos).**

Sexto proyecto de [rl_metodos](../README.md). Es el [proyecto 02](../02_punto_fijo/) (elegir el paso α) llevado a un problema abierto de 2023–2026.

## El problema

Descenso de gradiente sin momento, `xₖ₊₁ = xₖ − (hₖ/L)·∇f(xₖ)`, sobre funciones convexas L-suaves con `‖x₀ − x*‖ ≤ R`. ¿Qué sucesión de pasos `h₁…hₙ` minimiza el peor caso de `f(xₙ) − f*`?

* **Horizonte fijo.** El *silver stepsize schedule* (Altschuler y Parrilo, 2023) da `O(n^−1.2716)` con un patrón fractal 2-ádico; Grimmer, Shu y Wang (2024) componen schedules que igualan o baten a los óptimos numéricos y conjeturan que son minimax-óptimos. Nadie lo ha probado.
* **Anytime** (sin conocer n de antemano; problema abierto planteado en COLT 2024 por Kornowski y Shamir). Mejor exponente conocido: 1.119 (Zhang et al., 2024). Cota inferior para cualquier schedule: 1.334 (2026). Nesterov con momento: 2.

## El verificador

`steprl/pep.py` escribe desde cero el *performance estimation problem* de Drori–Teboulle / Taylor–Hendrickx–Glineur: matriz de Gram de `(x₀, g₀, …, gₙ)`, desigualdades de interpolación de las funciones L-suaves convexas para todos los pares, condición inicial, y se maximiza `f(xₙ) − f*`. El valor del SDP es exacto y **el dual es una demostración**: multiplicadores `λᵢⱼ ≥ 0` que, combinando las desigualdades de interpolación, prueban `f(xₙ) − f* ≤ τ·L·R²`. `gd_worst_case(h, want_certificate=True)` devuelve ese certificado.

Validación (`python -m steprl verify` y `pytest`):

| comprobación | resultado |
|---|---|
| contra PEPit en schedules aleatorios | coincide hasta 1e-6 |
| paso constante h=1: cota de Drori–Teboulle 1/(4n+2) | exacta |
| silver de longitud 2ᵏ−1 vs garantía 1/(1+√(4ρ²ᵏ−3)) | se cumple para k = 1…4 |
| óptimo para n=1 | h = 1.5, τ = 0.125 (coincide con la literatura) |

## Capa numérica: reproducir los óptimos conocidos

`python -m steprl search --n-min 1 --n-max 8` (entropía cruzada con reinicios + Nelder–Mead). Referencia: la tabla de óptimos numéricos de Das Gupta et al. tal como la cita Grimmer–Shu–Wang (su convención es el doble de nuestro τ, comprobado con n = 1, 2).

| n | τ encontrado | referencia | ratio | h |
|---|---|---|---|---|
| 1 | 0.125000 | 0.125000 | 1.0000 | 1.5 |
| 2 | 0.065946 | 0.065945 | 1.0000 | 1.4142, 1.8768 |
| 3 | 0.042893 | 0.042895 | 1.0000 | 1.4142, 2.4142, 1.5 |
| 4 | 0.031921 | 0.031170 | 1.0241 | 1.4477, 2.6213, 1.4017, 1.8611 |
| 5 | 0.024451 | 0.024070 | 1.0158 | 1.4865, 1.5139, 3.4352, 1.4161, 1.873 |
| 6 | **0.020049** | 0.020100 | **0.9975** | 1.2712, 1.9699, 1.4089, 4.0874, 1.7318, 1.5 |
| 7 | 0.016707 | 0.016330 | 1.0231 | 1.3442, 1.5951, 2.6437, 1.6066, 1.407, 4.3367, 1.5303 |
| 8 | 0.014130 | 0.014055 | 1.0054 | 1.3624, 1.5941, 2.265, 1.3907, 5.2352, 1.4137, 2.4334, 1.4979 |

El valor de n = 6 está un 0.25 % por debajo del tabulado y lo confirman tres solvers (Clarabel, SCS y PEPit). Es minúsculo y la tabla citada puede ser un óptimo local de otro grupo, así que queda como **dato a contrastar** con los schedules compuestos de Grimmer–Shu–Wang, no como hallazgo. Para n ≥ 4 el paisaje es no convexo y la entropía cruzada se queda a 1–2 % con 4 reinicios.

## Capa simbólica: el LLM propone, el SDP certifica

`python -m steprl evolve --provider {mock,openai,anthropic} --model … --objective {anytime,fixed}`

Un candidato es código Python que define `schedule(n) -> list[float]`. Se puntúa con el PEP en dos regímenes:

* **fijo**: `τ(n)/referencia` para n = 1…8 y 10 (media geométrica);
* **anytime**: para `schedule(N)` se certifica el peor caso de cada prefijo `τ₁…τ_N` y se reporta el mayor `p` tal que `τₜ ≤ ½·t^(−p)` para todo `t ≤ N` (½ es la cota trivial `f(x₀) − f* ≤ LR²/2`). Un solo prefijo malo hunde `p`: el silver truncado a n = 4 u 8 termina en un pico y da `τ₄ = τ₈ = 0.0858`. La métrica solo es comparable a **igual N**, y las afirmaciones asintóticas exigen N grande (63 o más).

El prompt lleva las referencias (silver y paso constante a ese mismo N) y los mejores programas con sus puntuaciones; el LLM devuelve un programa nuevo; todo queda en `runs/<fecha>/evolucion.json` y `mejor.py`. El proveedor es intercambiable:

```bash
export OPENAI_API_KEY=…      # o ANTHROPIC_API_KEY
python -m steprl evolve --provider openai    --model <id del modelo GPT> --effort high --generations 20 --anytime-n 31
python -m steprl evolve --provider anthropic --model claude-fable-5-1     --effort high --generations 20 --anytime-n 31
python -m steprl evolve --provider mock      # ensayo en seco sin claves
```

## Qué sería publicable

1. Un schedule con `p` anytime certificado por encima de 1.119 para N grande (aunque sea numérico: la cota inferior de 2026 salió como nota corta).
2. Una **gramática de composición** descubierta por el LLM que generalice a todo n y bata a los óptimos numéricos: el ángulo que nadie ha tomado (la comunidad usa branch-and-bound y análisis a mano).
3. Extensiones donde no hay óptimos conocidos: proximal, proyectado, estocástico.

## Uso

```bash
cd 06_stepsizes
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,llm]"
python -m steprl verify
python -m steprl search --n-max 4
python -m steprl evolve --provider mock --generations 2 --anytime-n 15
pytest -q
```

## Referencias

* Drori y Teboulle (2014). *Performance of first-order methods for smooth convex minimization: a novel approach.* Math. Prog.
* Taylor, Hendrickx y Glineur (2017). *Smooth strongly convex interpolation and exact worst-case performance of first-order methods.* Math. Prog.
* Goujaud et al. (2022). *PEPit: computer-assisted worst-case analyses of first-order optimization methods in Python.* [arXiv:2201.04040](https://arxiv.org/abs/2201.04040)
* Das Gupta, Van Parys y Ryu (2024). *Branch-and-bound performance estimation programming.* Math. Prog. [arXiv:2203.07305](https://arxiv.org/abs/2203.07305)
* Altschuler y Parrilo (2023). *Acceleration by stepsize hedging II: silver stepsize schedule for smooth convex optimization.* [arXiv:2309.16530](https://arxiv.org/abs/2309.16530)
* Grimmer, Shu y Wang (2024). *Composing optimized stepsize schedules for gradient descent.* [arXiv:2410.16249](https://arxiv.org/abs/2410.16249)
* Kornowski y Shamir (2024). *Open problem: anytime convergence rate of gradient descent.* COLT. [arXiv:2406.13888](https://arxiv.org/abs/2406.13888)
* Zhang et al. (2024). *Anytime acceleration of gradient descent.* [arXiv:2411.17668](https://arxiv.org/abs/2411.17668)
* *Lower bounds for anytime acceleration of gradient descent* (2026). [arXiv:2607.02053](https://arxiv.org/abs/2607.02053)
