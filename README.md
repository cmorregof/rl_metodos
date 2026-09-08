# rl_metodos

**Una serie de agentes de Reinforcement Learning que aprenden, en vivo en la terminal, métodos numéricos cada vez más sofisticados: primero el método, luego la demostración de su convergencia.**

Cada carpeta es un proyecto autocontenido con el mismo formato, al estilo de los vídeos de *AI Warehouse*:

1. **Fase 1 · el método.** Un agente tabular (Q-learning) que no conoce el algoritmo aprende, a base de recompensas, a ejecutarlo: dónde cortar, qué conservar, cómo ajustar un parámetro. Lo que emerge es el método del libro de texto.
2. **Fase 2 · la demostración.** El teorema de convergencia se modela como un grafo de lemas con dependencias, mezclado con distractores (lemas falsos, irrelevantes, non sequiturs clásicos). El agente aprende a ordenar los pasos válidos y a rechazar los distractores.
3. **Interfaz en vivo** con `rich`: gráficas ASCII de la iteración, tablas de la política aprendida con ✔/✘ frente a la teoría, la demostración escribiéndose con saltos lógicos tachados, cronómetro, marcador y bitácora de hitos.
4. **Síntesis final**: un informe en Markdown con prosa generada a partir de los datos (cuándo descubrió cada cosa, en qué orden se asentó cada lema, qué errores clásicos intentó), la demostración final, las tablas y un JSON con toda la historia.

## Proyectos

| # | método | qué aprende el agente | teorema demostrado |
|---|---|---|---|
| [01](01_biseccion/) | **Bisección** | cortar en λ = 1/2 y conservar el subintervalo con cambio de signo | convergencia de la bisección: mₙ → c con f(c) = 0 y \|mₙ − c\| ≤ (b − a)/2ⁿ⁺¹ |
| [02](02_punto_fijo/) | **Punto fijo** | ajustar α en x ← x − α·f(x) a partir de la razón de residuos ρ hasta que el mapa sea contractivo (ρ → 0 ≈ Newton) | punto fijo de Banach en [a, b]: existencia, unicidad, convergencia y cotas a priori / a posteriori |
| [03](03_taylor/) | **Series de Taylor** | sumar la serie de f(x*) con criterio: cuántos términos, cuándo parar (criterio del último término) y cuándo la serie no converge (criterio del cociente) | teorema de Taylor con resto de Lagrange (vía Rolle iterado) y convergencia de la serie |
| [04](04_newton/) | **Newton–Raphson** | cuándo dar el paso completo y cuándo amortiguar o retroceder con búsqueda lineal (Newton con salvaguardas); observa el orden ≈ 2 | convergencia cuadrática local, usando el resto de Lagrange de 03 |
| 05 | Secante | *(próximo)* | orden de convergencia φ = (1 + √5)/2 |
| 06 | … | | |

## Ejecutar un proyecto

```bash
cd 01_biseccion            # o 02_punto_fijo, 03_taylor, 04_newton
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
python -m bisectrl         # o fixpointrl, taylorrl, newtonrl
pytest -q
```

Los proyectos se encadenan: el punto fijo óptimo de 02 (α = 1/f') es el paso de Newton; el resto de Lagrange de 03 es la herramienta con la que 04 demuestra la convergencia cuadrática; y Bolzano (01) reaparece en 02 para la existencia del punto fijo.

Cada proyecto acepta `--speed`, `--fast`, `--no-tui`, `--seed`, `--episodes1`, `--episodes2` y `--tol`. Requiere Python ≥ 3.10 y una terminal de al menos 120 × 40.

## Ideas de diseño que se repiten

* **Shaping basado en potencial** (Ng, Harada y Russell, 1999): la recompensa por iteración es `−1 + (progreso medido con un potencial que el agente no puede manipular)`. Así la señal es densa sin cambiar la política óptima.
* **Estados observables, no privilegiados**: signos de f, razones de residuos, oscilación. Nada que el método real no pueda medir.
* **Distractores en la demostración** que son errores reales de estudiante: «Bolzano garantiza raíz, luego el método converge», «|g'(p)| < 1 basta para cualquier x₀», «una sucesión acotada converge».
* **Hitos comparados con la teoría**: el agente no sabe cuál es la respuesta correcta, pero el informe sí, y marca el momento exacto en que la política coincide con ella. Cuando el agente discrepa de la regla de libro (Newton: Armijo es conservador si el paso propuesto ya decrece), el informe lo señala como «matiz» en vez de esconderlo.
* **El potencial del shaping no debe depender de lo que el agente controla**: en punto fijo se usa |f(x)| y no el tamaño del paso (reducir α "fingiría" progreso); en Newton se usa el error real y no |f| (si no, "converger" a f → 0 en el infinito parecería un éxito).

## Licencia

MIT.
