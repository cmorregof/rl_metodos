# Laboratorio 3 · notas del docente

Resultados medidos el 6 de octubre de 2026 en un MacBook Air (Apple M5, 24 GB) con macOS 26.5.2 y
Python 3.13.13, semilla 1 salvo que se diga otra cosa. Las cifras son idénticas a las de la primera
medición (Linux, Python 3.12); los tiempos son los de este Mac. Todo se reproduce con
`docs/curso/lab03/experimentos.py` y `graficar.py`. Este laboratorio usa el proyecto 07
(interpolación); en el arco del curso estaba como lab 5 y se adelantó porque la clase del 1 de
octubre ya cubrió Taylor y Lagrange. El laboratorio de Taylor (proyecto 03) sigue pendiente.

## Plan de la sesión (90 min)

| min | bloque | modo |
|---|---|---|
| 0–10 | descargar el ZIP de hoy, instalar 07 con matplotlib, `--no-tui` (8 s) | guiado |
| 10–25 | basales a mano (a)–(d), sin IA; revisar con `basales` | parejas, papel |
| 25–40 | `graficar.py`: ajustar `PUNTOS`, `RANGO`, `NODOS`; PIB (e)–(g) | parejas |
| 40–55 | agente con animación, dos semillas, informe (h)–(i) | parejas |
| 55–70 | `familias`, `solo_equi`, `sin_runge`, `recompensa` (j)–(l) | parejas + plenaria corta |
| 70–85 | `dibujar`, `grafo` (m)–(n); `mi_grafo` (o) | parejas + plenaria |
| 85–90 | cierre y entrega | expositivo |

Las ideas de la sesión son tres: la suma de valores absolutos de los basales es lo que amplifica
los errores (de (c) a (g) y a (h)); lo que el agente aprende depende de los problemas que ve ((k));
y el verificador solo sabe lo que dicen las flechas ((n) y (o)). Si se va el tiempo, cortar (i),
(k) y la tarea opcional, en ese orden. No cortar (c), (g) ni `mi_grafo`.

## Tiempos medidos

| orden | tiempo |
|---|---|
| `python -m interprl --no-tui --seed 1` | 8 s (fase 1: 7.5–8.2 s, fase 2: 0.1–0.2 s) |
| `python -m interprl --seed 1` (animación) | 2.5 min (2 min 28 s y 2 min 39 s) |
| `basales`, `dibujar`, `grafo`, `mi_grafo` | < 0.1 s |
| `familias` | 0.8 s |
| `hitos --seed N` | 8 s (fases 1 y 2 completas) |
| `solo_equi`, `sin_runge` | 6.6 s (dos entrenamientos de 2000 generaciones) |
| `recompensa` | 10 s (tres entrenamientos) |
| `graficar.py` | 0.4 s; la primera vez, 6 s (matplotlib arma su caché de fuentes) |
| `search --n 4` | 1 s |

Son dos tandas seguidas que coinciden entre sí. La primera tanda, justo después de instalar, fue más
lenta: `familias` 12 s y los entrenamientos de `experimentos.py` alrededor de 1.5 veces más. De los
2.5 min de la animación, el cálculo son unos 8 s; el resto son las pausas fijas de la animación y el
dibujo. En la primera medición (Linux, Python 3.12) los tiempos eran algo más del doble que estos
(17 s para `--no-tui`). En los PCs del laboratorio no se ha medido: contar con que sea más lento que
aquí. Avisar que `--no-tui` no está colgado.

## Respuestas y cifras

**(a)** L₀ = t(t − 1)/2, L₁ = 1 − t², L₂ = t(t + 1)/2.

**(b)** L₀ = (t − 1)(t − 3)/3, L₁ = −t(t − 3)/2, L₂ = t(t − 1)/6. El error típico es el signo de
L₁: el denominador es (1 − 0)(1 − 3) = −2.

**(c)** Lo que imprime `basales`:

| nodos | t | L₀ | L₁ | L₂ | suma | suma de valores absolutos |
|---|---|---|---|---|---|---|
| −1, 0, 1 | −1/2 | 3/8 | 3/4 | −1/8 | 1 | 5/4 |
| −1, 0, 1 | 1/2 | −1/8 | 3/4 | 3/8 | 1 | 5/4 |
| 0, 1, 3 | 2 | −1/3 | 1 | 1/3 | 1 | 5/3 |
| −1, 0, 1 | 2 | 1 | −3 | 3 | 1 | 7 |

La constante de Lebesgue de −1, 0, 1 en [−1, 1] es exactamente 5/4, alcanzada en t = ±1/2: el
número que calcularon a mano. Vale la pena decirlo en voz alta.

**(d)** Una versión aceptable, que es también la solución de `mi_grafo` (7 pasos, el agente la
ordena en la generación 5):

| clave | afirmación | deps |
|---|---|---|
| H1 | los nodos son distintos | — |
| DEF | Lₖ(t) = ∏ (t − xⱼ)/(xₖ − xⱼ), grado n | H1 |
| DELTA | Lₖ(xⱼ) = 1 si j = k, 0 si no | DEF |
| S | S(t) = ΣLₖ(t) tiene grado ≤ n | DEF |
| SNODOS | S(xⱼ) = 1 para todo j | S, DELTA |
| UNIQ | un polinomio de grado ≤ n con n + 1 ceros es nulo | H1 |
| QED | S − 1 tiene grado ≤ n y n + 1 ceros, luego S ≡ 1 | S, SNODOS, UNIQ |

Distractor natural: «S vale 1 en los n + 1 nodos, luego S ≡ 1» (le falta el grado ≤ n: la flecha
de S a QED es la que lo descarta).

**(e)** Vietnam: P(2013) = 216.3 frente a 213.7 real (+1.2 %); P(2015) = 247.1 frente a 239.3
(+3.3 %). Funciona porque la serie es suave entre los nodos y los basales en ±1/2 están entre
−1/8 y 3/4: no amplifican. 2015 es peor porque el crecimiento en dólares casi se detiene ese año
(233.5 → 239.3) y la parábola no lo sabe.

**(f)** Colombia: P(2020) = 317.8 frente a 270.3 real (+47.5, +17.6 %). El polinomio solo sabe
lo que dicen los nodos, y la caída de 2020 no dejó rastro en 2018, 2019 ni 2021. La fórmula del
error necesita f‴ acotada y conocida; con datos no hay f, y un choque de un año hace que cualquier
f razonable tenga f‴ enorme. La fórmula dice cuándo confiar (f suave), no cuánto se equivoca uno
con datos reales.

**(g)** China: P(2022) = 20 179.9 frente a 17 881.8 (+12.9 %); P(2023) = 22 932.0 frente a
17 794.8 (+28.9 %). Basales en t = 3/2: 3/8, −5/4, 15/8 (suma de absolutos 3.5); en t = 2:
1, −3, 3 (suma 7). Un 1 % en el dato de 2021 (178.2) se multiplica por L₂(2) = 3: la estimación
de 2023 se mueve 534.6, de 22 932.0 a 23 466.6. Fuera de los nodos los basales crecen y el
polinomio amplifica. Interpolar no es pronosticar.

**Hitos** (`experimentos.py hitos --seed N`, idénticos a los de la animación con la misma semilla):

| hito | seed 1 | seed 2 |
|---|---|---|
| primera aproximación correcta | 80 | 12 |
| descubrió el criterio de parada | 80 | 111 |
| su regla coincide con la teoría (8 estados, estable 100 gen.) | 526 | 542 |
| prefiere γ = 1 de forma estable | 726 | 572 |
| primera demostración completa | 186 | 215 |
| primera demostración mínima | 463 | 368 |

Con seed 2 aparece además «≥95 % de aciertos» en la 2490; con seed 1 no aparece. Al final: seed 1,
89 % de éxito en las últimas 100, 2187 correctas, 792 prematuras, 21 agotadas; seed 2, 80 %,
2186, 782, 32.

**(h)** Seed 1: Q(γ = 1) = −2.8 (2584 usos) frente a Q(γ = 0) = −13.4 (110 usos). Q(γ) es el
retorno medio de una partida con esa familia; con equiespaciados las funciones tipo Runge acaban
en entrega prematura o agotadas. La tabla de Λ del informe: con n = 16, equiespaciados 934.5,
Chebyshev 2.7. Es el mismo fenómeno de la última fila de (c): basales grandes amplifican, y con
equiespaciados crecen como 2ⁿ.

**(i)** Cuando los interpolantes convergen geométricamente, |pₙ − pₙ₋₁| es del orden del error
de pₙ₋₁ y acota bien el de pₙ. Engaña cuando todavía no hay convergencia (dos polinomios malos
pueden parecerse) o cuando el error se estanca. Costo: 792 entregas prematuras de 3000 con seed 1.
Es el mismo problema que Colombia: lo que no se ve en las muestras no existe para el método.

**(j)** `familias` (120 problemas, 40 nodos máximo):

| tipo de f | cuántos | equiespaciados | Chebyshev |
|---|---|---|---|
| Runge 1/(1 + ax²) | 31 | 0/31 | 31/31, 29.2 nodos |
| polo 1/(x − c) | 29 | 27/29, 22.5 nodos | 29/29, 16.5 nodos |
| raíz √(x + c) | 17 | 17/17, 14.9 nodos | 17/17, 11.2 nodos |
| entera (e^{kx}, sin) | 43 | 43/43, 10.4 nodos | 43/43, 9.5 nodos |

`solo_equi` (2000 generaciones): éxito 85 % → 25 %, entregas prematuras 653 → 1505, agotadas
14 → 7. Con Runge, cada nodo equiespaciado sube el error: el progreso es negativo en cada paso y
seguir cuesta más que el −10 de entregar mal. El agente aprende a rendirse pronto, que es lo
racional con esos nodos.

**(k)** `sin_runge`: éxito 94 %, Q(γ) entre +0.7 y +7.5, preferida γ = 0.75. Ya no hay una familia
que gane con claridad (las diferencias están dentro del ruido del bandido, que usa α = 0.05). Lo
que el agente aprendió en la parte 4 no es «Chebyshev es mejor» como teorema: es «con estos
problemas, Chebyshev da más puntos». El teorema (Chebyshev minimiza máx|w|) es sobre los nodos;
la preferencia del agente aparece solo cuando los problemas la hacen importar. La distribución de
problemas es parte de la especificación, igual que la recompensa.

**(l)** `recompensa`:

| caso | éxito últimas 100 | correctas / prematuras / agotadas | familia |
|---|---|---|---|
| original (−10, −1, +10) | 85 % | 1333 / 653 / 14 | γ = 1 |
| entregar mal cuesta 0 | 0 % | 1 / 1999 / 0 | ninguna (imprime γ = 0 por empate: todas las Q valen −0.0) |
| cada nodo es gratis | 87 % | 1353 / 632 / 15 | γ = 1 |

Con «entregar mal cuesta 0» entrega siempre en el primer paso. Una partida que trabaja hace unos
18 nodos (−18), el progreso telescopa a log₁₀(error inicial/error final) ≈ 5, y gana +10: unos
−3 en total. Entregar ya da 0. Rendirse es óptimo. Es el mismo fenómeno que (f) del lab 1. Con
«cada nodo gratis» casi nada cambia: el descuento γ = 0.95 hace que el +10 valga más cuanto antes,
y pasado el error de máquina añadir nodos no da progreso. Probado también `--entregar-mal -1
--nodo 0`: 85 %, γ = 1.

**(m)** Si los n + 1 nodos se juntan en x₀, las condiciones de interpolación pasan a ser
f(x₀), f′(x₀), …, f⁽ⁿ⁾(x₀) (interpolación de Hermite en el límite), el interpolante es el
polinomio de Taylor y w(x) → (x − x₀)ⁿ⁺¹. Taylor es el caso en que toda la información está en
un punto: la idea con la que se abrió la clase del 1 de octubre. Fuera de ese límite, el distractor
confunde los dos polinomios.

**(n)** Si x = xᵢ, w(x) = 0 y g ni siquiera está definida. FIX trata ese caso aparte (ambos lados
son 0). Sin la flecha FIX → AUX, la demostración de 12 pasos que acepta el verificador divide por
cero sin decirlo. Responsable: quien escribió las `deps`.

**(o)** Libre. Buenos distractores que han salido al pensarlo: el del grado (arriba); «cada Lₖ está
entre 0 y 1, luego la suma es ≤ n + 1» (falso: la fila t = 2 de (c)); «S(t) = 1 porque los
basales son una partición de la unidad» (circular).

**Párrafo final.** Que mencionen: rellenar datos entre muestras de algo suave (sirve), y que no
sirve para choques que no tocan los nodos (Colombia) ni para extrapolar (China), y que más nodos
equiespaciados no arreglan nada (Runge). Bonus si conectan con Λ.

**Tarea opcional.** Equiespaciados con 5 nodos: Λ = 2.2078. Con nodos interiores en ±0.7:
Λ = 1.7727. `search --n 4` (1 s) encuentra ±0.620911 con Λ = 1.55949, por debajo del Chebyshev
extendido (1.57017).

## Trampas

* **El ZIP tiene que ser de hoy.** Si lo descargaron antes, no trae `docs/curso/lab03`.
* `matplotlib` va en la misma línea de instalación. Si falta, `graficar.py` dice cómo instalarlo.
* En `basales` los nodos van con `=`: `--nodos=-1,0,1`. Sin `=` el `-1` parece una opción.
* `graficar.py` se edita. Si alguien lo rompe (comillas, comas), se recupera del ZIP o con
  `git checkout docs/curso/lab03/graficar.py`.
* Con `PUNTOS = [..., 2]` y `RANGO = None`, la línea en t = 2 existe pero cae fuera del eje:
  es intencional, es el ajuste 2.
* El PIB está en dólares corrientes: mezcla producción, inflación y tasa de cambio. La caída de
  Colombia entre 2014 y 2015 (381.2 → 293.5) es sobre todo devaluación del peso; si alguien la
  usa como nodo, es una buena discusión.
* `pib_tres_paises.csv` está congelado (Banco Mundial vía `datasets/gdp`, CC BY 4.0, hasta 2023).
  `graficar.py --reconstruir` lo regenera desde la fuente.
* `mi_grafo.py` sin tocar da un error a propósito: ∎ sin deps. Es el primer mensaje que ven.
