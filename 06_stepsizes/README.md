# 06 · StepRL — programas de pasos para el descenso de gradiente, con el PEP como juez

**Primer proyecto de la serie en la frontera: no hay grafo de lemas escrito a mano. Un verificador independiente (un programa semidefinido) certifica el peor caso de cada programa de pasos, y esa certificación es la única recompensa. La búsqueda la hacen una capa numérica (entropía cruzada) y una capa simbólica (un LLM propone programas que generan programas de pasos).**

Sexto proyecto de [rl_metodos](../README.md). Es el [proyecto 02](../02_punto_fijo/) (elegir el paso α) llevado a un problema abierto de 2023–2026.

## El problema

Descenso de gradiente sin momento, `xₖ₊₁ = xₖ − (hₖ/L)·∇f(xₖ)`, sobre funciones convexas L-suaves con `‖x₀ − x*‖ ≤ R`. ¿Qué sucesión de pasos `h₁…hₙ` minimiza el peor caso de `f(xₙ) − f*`?

* **Horizonte fijo.** El *silver stepsize schedule* (Altschuler y Parrilo, 2023) da `O(n^−1.2716)` con un patrón fractal 2-ádico; Grimmer, Shu y Wang (2024) componen schedules que igualan o baten a los óptimos numéricos y conjeturan que son minimax-óptimos. Nadie lo ha probado.
* **Anytime** (sin conocer n de antemano; problema abierto planteado en COLT 2024 por Kornowski y Shamir). Mejor exponente conocido: 1.119 (Zhang et al., COLT 2025). **Cerrado en septiembre de 2026**: Ye y Liu ([arXiv:2609.09152](https://arxiv.org/abs/2609.09152)) prueban que ningún schedule de pasos *no negativos* supera 2p/(1+p) ≈ 1.1195 (p = log₂(1+√2)) salvo n^{o(1)}, y que el silver es óptimo en exponente a horizonte fijo. Las cotas que admiten pasos negativos son más débiles (1.2408 anytime, 1.6342 fijo; [arXiv:2609.02855](https://arxiv.org/abs/2609.02855)). Nesterov con momento: 2.

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
* **anytime**: para `schedule(N)` se certifica el peor caso de cada prefijo `τ₁…τ_N` y la puntuación es el **exponente por duplicación** `p = mín_{t ≥ 8} log(τ_{⌊t/2⌋}/τ_t)/log 2`: el orden observado en la última duplicación del horizonte, independiente de la constante y que un pico hunde a cero (el silver truncado a n = 8 o 16 da `τ₁₆ = τ₈ = 0.0858`). Se evalúa en varios horizontes (31 y 63 por defecto) y manda el peor, para que no se pueda sobreajustar al horizonte. También se reporta la garantía finita `τₜ ≤ ½·t^(−p)` para todo `t ≤ N`, pero solo como información: con ½ está inflada a N pequeño (el paso constante da 1.21 a N = 31 y tiende a 1) y no es comparable con los exponentes publicados.

El prompt lleva las referencias (silver y paso constante a ese mismo N) y los mejores programas con sus puntuaciones; el LLM devuelve un programa nuevo; todo queda en `runs/<fecha>/evolucion.json` y `mejor.py`. El proveedor es intercambiable:

```bash
export OPENAI_API_KEY=…      # o ANTHROPIC_API_KEY
python -m steprl evolve --provider openai    --model <id del modelo GPT> --effort high --generations 20 --anytime-n 31,63
python -m steprl evolve --provider anthropic --model claude-fable-5-1     --effort high --generations 20 --anytime-n 31,63
python -m steprl evolve --provider mock      # ensayo en seco sin claves
```

## Ejecuciones

* [01 · gpt-6-astra, 2026-09-09](docs/ejecucion-01-gpt-6-astra.md): 6 programas válidos de 20; todos con pasos acotados (orden 1, constante 2–2.5× mejor que el paso constante); el modelo converge a las composiciones de Grimmer–Shu–Wang; una generación sobreajustó el horizonte. Motivó la métrica por duplicación, los dos horizontes y el límite de 64k.

## Qué sería publicable (revisado el 16 de septiembre de 2026)

~~1. Un schedule con `p` anytime certificado por encima de 1.119.~~ Cerrado para pasos no negativos (Ye–Liu, 8 de septiembre de 2026).

1. **El PEP como entorno de RL** (`steprl/env.py`): instancias (n, μ, criterio, ¿pasos negativos?), recompensa `log(τ_ref/τ)` referida a lo mejor conocido, sandbox, caché, y puerta de certificación exacta para cualquier mejora. Sobre él: entrenar un modelo abierto (expert iteration primero; GRPO si el Air aguanta) y compararlo con el bucle de LLM congelado a igual cómputo. El análogo más cercano es AutoOPT (Kim, Ryu y Das Gupta, [arXiv:2608.07407](https://arxiv.org/abs/2608.07407), agosto de 2026: BnB-PEP → LLM → Lean 4), que no entrena nada ni tiene entorno.
2. **Teoremas certificados** en variantes abiertas, como salida del entorno: pasos negativos (¿baten al silver?), fuertemente convexo (n, κ) sin cotas inferiores conocidas, norma del gradiente a horizonte fijo, proximal. Cada mejora sale con un certificado racional exacto (`steprl/certify.py`), no con un número del solver.
3. Una **gramática de composición** que generalice a todo n, si el agente la encuentra: hoy la referencia analítica son las sucesiones OBS-F/OBS-G de Grimmer–Shu–Wang (pendientes de implementar como `τ_ref`).

## Entorno, certificados y referencias

| módulo | qué hace |
|---|---|
| `pep.py` | ahora acepta `mu` (F_{μ,L}, interpolación de Taylor–Hendrickx–Glineur), `solver_opts` y pasos negativos; validado contra PEPit (`pytest`) |
| `certify.py` | dual del PEP con margen δ → redondeo a racionales → reparación exacta del flujo → LDLᵀ exacta con `fractions`. `verify()` recomprueba solo con los racionales. Exceso típico sobre el valor del SDP: 1e-7 |
| `refs.py` | instancias, familias de entrenamiento (n ≤ 12; μ ∈ {0, 0.01, 0.1}; f(xₙ)−f* y ‖∇f(xₙ)‖²) y de prueba (n ∈ {16, 24}); `τ_ref` = min(tabla, entropía cruzada), con la fuente anotada |
| `env.py` | `score(programa, instancia)` y `score_batch` en paralelo con caché; recompensa `max(−2, log(τ_ref/τ))`, −3 si inválido; toda mejora afirmada se recalcula con tolerancias finas y se certifica, y sin certificado no cuenta |

```bash
python -m steprl bench                              # ms por SDP y segundos por certificado según n
python -m steprl refs --workers 8                   # construye refs.json (una vez; horas en el Air, incremental)
python -m steprl score --program mejor.py --n 8 --mu 0.1 [--negative]
python -m steprl certify --h 1.4142,1.8768 --out cert.json && python -m steprl check cert.json
```

Tiempos medidos (un núcleo, Clarabel): SDP n = 8 ≈ 80 ms, n = 12 ≈ 200 ms, n = 24 ≈ 1 s; certificado exacto n = 12 ≈ 1 s, n = 16 ≈ 2 s.

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
* Tsai, Fatkhullin, Zhang y He (2026). *Lower bounds for anytime acceleration of gradient descent.* [arXiv:2607.02053](https://arxiv.org/abs/2607.02053)
* Jung, Cho y Yun (2026). *Stronger lower bounds for (non-)anytime acceleration of gradient descent.* [arXiv:2609.04032](https://arxiv.org/abs/2609.04032)
* Ye y Liu (2026). *Improved gradient descent lower bounds beyond Nesterov.* [arXiv:2609.02855](https://arxiv.org/abs/2609.02855) · *Silver rate is (almost) optimal for gradient descent acceleration.* [arXiv:2609.09152](https://arxiv.org/abs/2609.09152)
* Kim, Ryu y Das Gupta (2026). *A domain-specific harness for end-to-end automation of optimization research* (AutoOPT). [arXiv:2608.07407](https://arxiv.org/abs/2608.07407)
