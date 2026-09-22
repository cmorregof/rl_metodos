# 07 · InterpRL — interpolación polinómica

**Un agente de Reinforcement Learning aprende, en vivo en la terminal, dónde poner los nodos de interpolación (Chebyshev emerge frente a equiespaciados) y cuántos usar, demuestra el teorema del error de interpolación, y en la fase 3 busca nodos con constante de Lebesgue menor que la de Chebyshev usando un verificador independiente.**

Séptimo proyecto de [rl_metodos](../README.md) y el primero pensado para el curso: sale de la búsqueda de raíces y estrena el patrón de **tres fases** que sigue el resto de la serie docente. La fase 3 es el [proyecto 06](../06_stepsizes/) en pequeño: no hay grafo de lemas, hay un verificador y una búsqueda, y el resultado es un número que cualquiera puede recomputar.

```
python -m interprl                 # fases 1 y 2, en vivo
python -m interprl search --n 8    # fase 3
```

## Fase 1 · dónde y cuántos nodos

Objetivo: aproximar `f` en `[−1, 1]` por un polinomio interpolante con `máx|f − pₙ| < tol` (relativa, `10⁻⁵`) usando el menor grado. Funciones: tipo Runge `1/(1 + a x²)` con `a ∈ [4, 8]` (polo dentro de la región de Runge: con nodos equiespaciados **añadir nodos empeora**), `1/(x − c)` con el polo cerca del intervalo, `√(x + c)`, y `e^{kx}`, `sin(kx + c)` (enteras: cualquier familia converge). Dos decisiones:

* **Familia de nodos**, una vez por episodio (cabeza bandido sobre `γ ∈ {0, ¼, ½, ¾, 1}`): `xᵢ(γ) = (1 − γ)·equiespaciadoᵢ + γ·Chebyshev–Lobattoᵢ`. El agente no sabe qué es Chebyshev; solo ve el retorno del episodio.
* **Cuántos nodos**, un paso por nodo (tabla Q): observa solo el **estimador** `máx|pₙ − pₙ₋₁|` frente a la tolerancia (4 cubos), si ese estimador **bajó o subió**, y si lleva pocos (`n ≤ 6`) o muchos nodos. Acciones: **añadir** un nodo (`−1 + log₁₀(error anterior / error nuevo)`, shaping por potencial con el error real, que el entorno conoce y el agente no ve) o **entregar** (`+10` si el error real es `< tol`, `−10` si no). Presupuesto: 40 nodos.

Lo que emerge, medido con semilla 1 y 3000 generaciones:

| qué | generación | valor final |
|---|---|---|
| criterio de parada: entregar cuando `est ≪ tol` y va bajando | ~80 | coincide con la teoría en los 8 estados bien visitados |
| prefiere `γ = 1` (Chebyshev) de forma estable | 500–800 según semilla | `Q(γ=1) ≈ −3` frente a `Q(γ=0) ≈ −12` |
| éxito en las últimas 100 generaciones | | 80–90 %, con ~19 nodos de media |

Por qué gana Chebyshev, sin agente: con equiespaciados la constante de Lebesgue crece como `2ⁿ` (n = 16: 934) y con Chebyshev como `log n` (n = 16: 2.7); y `máx|w|` de los nodos de Chebyshev es `2⁻ⁿ`, el mínimo posible. El informe imprime esa tabla.

Simplificación deliberada: al añadir un nodo se recalculan todos (la familia cambia con `n`) y eso cuesta 1, no `n`. El coste mide el grado, no las evaluaciones de `f`.

## Fase 2 · demostrar el teorema del error

**Teorema.** Sea `f ∈ Cⁿ⁺¹[a, b]`, `x₀ < … < xₙ` nodos distintos y `pₙ` el interpolante. Para cada `x` existe `ξ` con
`f(x) − pₙ(x) = f⁽ⁿ⁺¹⁾(ξ)·w(x)/(n+1)!`, `w(t) = ∏(t − xᵢ)`. Por tanto `máx|f − pₙ| ≤ M·máx|w|/(n+1)!`, y en `[−1, 1]` los nodos de Chebyshev minimizan `máx|w| = 2⁻ⁿ`.

Grafo de 13 pasos (hipótesis, existencia y unicidad, polinomio nodal, fijar `x`, función auxiliar, `n + 2` ceros, Rolle iterado, derivada `n + 1`, fórmula, cota, minimalidad de Chebyshev, ∎) más 8 **distractores** con los errores clásicos: «más nodos ⇒ menor error» (Runge), «`f` continua ⇒ `pₙ → f`» (Faber), el non sequitur de Weierstrass, «el error es 0 en los nodos luego es pequeño», «los equiespaciados minimizan `máx|w|`», la confusión con el resto de Taylor, «único ⇒ mejor aproximación», y Newton.

Mismo entorno y agente que en 01–05: `−0.5` por línea válida, `−2` por salto lógico, `+20` en ∎; Q-learning tabular con ε-greedy y bono UCB. Con semilla 1: primera demostración completa en la generación ~190, primera mínima en la ~460.

## Fase 3 · lo que la teoría no da: nodos con Λ mínima

El teorema dice qué nodos minimizan `máx|w|`. No dice qué nodos minimizan la **constante de Lebesgue** `Λ = máx Σ|ℓᵢ(x)|`, que es lo que de verdad amplifica el error de los datos (`‖f − pₙ‖ ≤ (1 + Λ)·‖f − p*‖`). Eso solo se conoce numéricamente (Brutman 1978, 1997): el Chebyshev extendido está a menos de 0.02 del óptimo, y los óptimos no tienen forma cerrada.

Aquí no hay grafo. Hay:

* un **verificador** (`interp.lebesgue_constant`): dados nodos cualesquiera, calcula `Λ` localizando el único máximo local de la función de Lebesgue en cada tramo entre nodos (malla + sección áurea; dos mallas distintas coinciden a `10⁻¹²`);
* una **búsqueda** (`search.py`): entropía cruzada sobre los nodos interiores, simétricos, con `±1` fijos, más refinamiento por coordenadas, todo escrito a mano para que se lea entero;
* **referencias**: Chebyshev–Lobatto y Chebyshev extendido.

```bash
python -m interprl search --n 8
#   chebyshev_lobatto      Λ = 2.27473107
#   chebyshev_extendido    Λ = 1.94157316
#   encontrado             Λ = 1.93872335  ratio = 0.998532  bate la referencia ✔
python -m interprl verify runs/<fecha>/busqueda_n8.json   # recalcula Λ con otra malla
python -m interprl lebesgue --nodes=-1,-0.5,0,0.5,1     # con "=" porque el primer nodo empieza por "-"
```

| n | Λ Chebyshev–Lobatto | Λ Chebyshev extendido | Λ encontrado | tiempo |
|---|---|---|---|---|
| 4 | 1.7988 | 1.5702 | **1.5595** | 0.6 s |
| 8 | 2.2747 | 1.9416 | **1.9387** | 1.8 s |
| 12 | 2.5393 | 2.1747 | **2.1574** | 5.4 s |

El "teorema" que sale de aquí es del tipo «estos 9 nodos tienen `Λ = 1.93872`, un 0.15 % menos que el Chebyshev extendido», y su demostración es que `verify` lo reproduce. Es pequeño, es nuevo para quien lo encuentra, y es exactamente la forma que tendrá el proyecto final del curso. Lo honesto: el verificador es numérico (doble precisión, no racionales como en el 06), y los óptimos de Lebesgue ya están tabulados en la literatura, así que "nuevo" aquí significa *reproducido y superada la referencia de libro*, no *inédito*. Para un resultado inédito hay que cambiar el problema: nodos no simétricos, otro intervalo o peso, o interpolación de Hermite, y ahí el verificador sigue sirviendo tal cual.

## Interfaz en vivo

| panel | qué muestra |
|---|---|
| **Fase 1** | `f` (cian) frente a `pₙ` (amarillo), los nodos ▲ en la línea base, `↑↓` rojos cuando el interpolante se sale del recuadro (Runge); debajo, el error real y el estimador de los últimos nodos |
| **Políticas aprendidas** | `Q(γ)` de cada familia con sus usos (★ la preferida) y la regla de parada estado a estado con ✔/✘ frente a la teoría |
| **Fase 2 / Lo que el agente cree ahora** | el intento de demostración en curso y la demostración de la política voraz |
| **Marcador / Bitácora** | cronómetro, generación, ε, aciertos, demostraciones e hitos |

Al terminar guarda `informe.md`, `demostracion.md` e `historia.json` en `runs/<fecha>/`.

## Uso

```bash
cd 07_interpolacion
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\python en vez de python
pip install -e ".[dev]"
python -m interprl                 # ~2 min con animación
python -m interprl --no-tui        # ~15 s
python -m interprl search --n 8
pytest -q
```

## Referencias

* Burden y Faires. *Análisis numérico*, §3.1 (interpolación de Lagrange y error).
* Trefethen (2013). *Approximation Theory and Approximation Practice*, caps. 5, 13, 15 (Runge, Lebesgue, potencial).
* Berrut y Trefethen (2004). *Barycentric Lagrange interpolation.* SIAM Review.
* Brutman (1978). *On the Lebesgue function for polynomial interpolation.* SIAM J. Numer. Anal.; (1997) *Lebesgue functions for polynomial interpolation: a survey.* Ann. Numer. Math.
