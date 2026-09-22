# rl_metodos · contexto para Claude

Serie de agentes de Reinforcement Learning (RL) tabular que aprenden métodos numéricos
y "demuestran" su teorema de convergencia, en vivo en la terminal. Autor: Carlos Orrego
(cmorregofranco@gmail.com). Idioma del repo y de toda la comunicación: **español**.

El repo tiene dos usos que conviven:

1. **Investigación / divulgación** (proyectos 01–06): el material original.
2. **Docencia**: base de un curso de métodos numéricos (matemáticas, semestres 2–4).
   Todo lo docente vive en `docs/curso/`. Empezar por `docs/curso/contexto.md`.

## Estructura

```
01_biseccion/   bisectrl    Q-learning aprende λ=1/2 e invariante de Bolzano; demuestra convergencia
02_punto_fijo/  fixpointrl  aprende a ajustar α en x←x−αf(x); demuestra Banach
03_taylor/      taylorrl    aprende cuántos términos sumar / cuándo parar; demuestra Taylor–Lagrange
04_newton/      newtonrl    aprende cuándo amortiguar el paso (Armijo emerge); convergencia cuadrática
05_secante/     secantrl    aprende Dekker (secante / regula falsi / bisección); orden φ
06_stepsizes/   steprl      FRONTERA: pasos del descenso de gradiente, verificador PEP (SDP), LLM propone
docs/curso/     material docente (contexto, laboratorios, notas del docente)
```

Cada proyecto 01–05 es un paquete independiente con la misma anatomía:

| módulo | qué es |
|---|---|
| `env_<metodo>.py` | entorno de la fase 1: problemas aleatorios, estado observable, acciones, recompensa con *shaping* por potencial |
| `proof_kb.py` | grafo de lemas del teorema: `Step(key, title, text, deps, distractor, is_qed)`; `REQUIRED_KEYS`, `MIN_PROOF_LENGTH` |
| `env_proof.py` | entorno de la fase 2: estado = máscara de pasos establecidos; válido ⇔ deps cumplidas, no repetido, no distractor |
| `agent.py` | `QAgent`: Q-learning tabular, ε-greedy, tasa 1/N opcional, bono UCB |
| `trainer.py` | bucle de ambas fases, hitos (`mark`), política voraz, curvas |
| `tui.py` | interfaz `rich` (necesita terminal ≥ 120×40) |
| `report.py` | `informe.md`, `demostracion.md`, `historia.json` en `runs/<fecha>/` |
| `cli.py` | flags: `--episodes1 --episodes2 --seed --tol --speed --fast --no-tui --out` |

El 06 rompe el patrón: no hay grafo de lemas; un SDP (`pep.py`) certifica cada propuesta,
`certify.py` produce certificados racionales exactos y `evolve.py` hace que un LLM proponga
programas. Necesita `.[llm]` y claves en `.env` (ignorado por git). Es material de
investigación, no de laboratorio para semestres 2–4.

## Cómo ejecutar

Cada proyecto tiene su propio `.venv` ya creado en esta máquina (Python 3.13 funciona).

```bash
cd 01_biseccion && source .venv/bin/activate
python -m bisectrl                      # con animación, ~1–2 min
python -m bisectrl --no-tui --seed 1    # sin interfaz: < 1 s
pytest -q
```

Tiempos medidos (22 sep 2026, MacBook): `--no-tui` con 3000+3000 generaciones tarda
0.35 s en 01–05. El 06 `verify` tarda ~3 s. La animación es lo que consume tiempo.

Calibración de 01 con semilla 1: λ = 1/2 estable en la gen ~345, invariante de Bolzano
en la ~358, primera demostración completa en la ~256 de fase 2, primera mínima en la ~459.
Con 100 generaciones no aprende nada; con 300 aprende a medias. Útil para laboratorios.

## Convenciones

* Commits en español, imperativo o descriptivo corto, con prefijo del proyecto (`06: ...`).
* `runs/`, `.venv/`, `*.egg-info`, `.env*` están ignorados. No versionar corridas.
* Los README de cada proyecto son la documentación de referencia y están al día;
  el README raíz resume la serie y las ideas de diseño repetidas (shaping por potencial,
  estados observables, distractores que son errores reales de estudiante).
* CI (`.github/workflows`): pytest + smoke `--no-tui` por proyecto en Python 3.12.
* No tocar los paquetes 01–06 para fines docentes: los experimentos de laboratorio se
  hacen editando constantes localmente o con los scripts de `docs/curso/labNN/`, y se
  revierten con `git checkout`.

## Honestidad sobre lo que hace la "fase 2"

La fase 2 **no demuestra teoremas**: el grafo de lemas está escrito a mano y el agente
aprende un orden topológico válido y a descartar distractores. Lo matemáticamente
sustancioso es escribir el grafo (qué depende de qué, qué errores son tentadores).
Esto hay que decirlo así a los estudiantes. El salto a "el agente encuentra la
demostración" solo ocurre en el 06, donde el verificador es independiente (un SDP).
