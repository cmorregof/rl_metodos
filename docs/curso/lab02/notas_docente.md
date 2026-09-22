# Laboratorio 2 · notas del docente

Resultados medidos el 22 de septiembre de 2026 (MacBook, Python 3.13, semilla 1). Todo se
reproduce con `docs/curso/lab02/experimentos.py`. Cada experimento tarda menos de 1 s.

## Plan de la sesión (90 min)

| min | bloque | modo |
|---|---|---|
| 0–10 | instalar 02 (venv nuevo) y comprobar con `--no-tui` | guiado |
| 10–20 | el problema en la pizarra: g(x) = x − αf(x), |1 − αf′| < 1, ρ como contracción empírica. (a)–(c) en papel | expositivo + parejas |
| 20–30 | `alfa_fijo`; plenaria corta sobre (d) | parejas + plenaria |
| 30–45 | correr con animación, hitos, tabla de la ley | parejas |
| 45–60 | `ciego`; plenaria sobre (g): el estado importa | parejas + plenaria |
| 60–72 | `potencial`; (i) | parejas |
| 72–85 | `dibujar`, `grafo`; plenaria sobre (l) | parejas + plenaria |
| 85–90 | cierre y entrega | expositivo |

Las ideas de la sesión son (g) y (i): el estado y el potencial son decisiones de diseño, y cada
una mal tomada produce un agente que aprende otra cosa. Si se va el tiempo, cortar (h) y (k).

## Respuestas y cifras

**(a)** g′ = 1 − 2α; converge si |1 − 2α| < 1, o sea 0 < α < 1. Con α = 1/2 = 1/f′, g′ = 0 y converge en un paso (es Newton).

**(b)** ρ > 1 monótona: αf′ < 0, el signo de α está mal → invertir. ρ > 1 oscila: αf′ > 2 → α demasiado grande → reducir. ρ < 1 oscila: 1 < αf′ < 2, contractivo pero pasado → reducir. ρ ∈ (0.5, 1) monótona: αf′ pequeño → doblar. ρ < 0.25: casi Newton → mantener.

**(c)** Los cubos intermedios (0.25–0.5), donde doblar o mantener dan casi lo mismo. El informe los marca con dos acciones válidas.

**(d)** Medido (300 problemas por α): α = −0.5 → 57 %, α = +0.5 → 53 %, α = −1 → 46 %, α = +1 → 30 %, α = +0.1 → 23 %, α = −0.1 → 7 %. El agente: 92 % en las últimas 100 generaciones, 10.9 iteraciones de media. Ningún α fijo puede servir porque el banco mezcla f′(r) > 0 y f′(r) < 0 (la mitad de los problemas tiene el signo invertido a propósito), y además f′ varía de 0.4 a 5 en magnitud.

**Hitos con semilla 1**: "signo de α mal → invertir" en la gen 1 (es el estado más visitado al principio), primera convergencia 27, "oscila → reducir" 59, ley = Banach en los 12 estados en la 680, ≥95 % en la 681. Fase 2: primera completa 198, voraz mínima 225, primera mínima 421.

**(e)** Con 3000 generaciones coincide en 12/12. La tabla del informe es la misma que imprime `hitos`.

**(f)** Con ρ ≥ 1.5 monótona, αf′ es muy negativo. Invertir el signo lo arregla de golpe; reducir a la mitad también reduce |αf′| y en dos o tres pasos se llega a lo mismo. Las dos son "aceptables" en el código de la teoría (`theoretical_action`).

**(g)** Medido: éxito 92 % → 74 %; divergencias 255 → 708. En ρ > 1 el agente ciego aprende **invertir** en ambos cubos. Es lo racional: no puede distinguir "signo mal" (invertir) de "α grande" (reducir), y de los dos errores, invertir cuando había que reducir suele salvarse en el paso siguiente (el nuevo estado vuelve a ser ρ > 1 y vuelve a invertir, oscilando entre signos hasta que otro estado reduce α), mientras que reducir cuando había que invertir nunca converge. Es un ejemplo limpio de que la política óptima depende de lo que el estado permite ver.

**(h)** Buenas respuestas: el signo de f(xₙ)·(xₙ − xₙ₋₁) (dice si el paso va hacia la raíz), o el cociente (xₙ₊₁ − xₙ)/(xₙ − xₙ₋₁) (estima g′ con signo, que es mejor que ρ porque distingue directamente g′ > 1, g′ < −1 y |g′| < 1). Esta última es de hecho lo que hace el método de Steffensen/Aitken. Cualquier observable que se calcule con valores de f y de x ya vistos vale.

**(i)** Medido: éxito 92 % → 59 %; ley coincide en 10/12. Las desviaciones: en `ρ < 0.25, oscila` reduce α en vez de mantener (reducir α hace el paso más corto, y eso ahora da puntos), y en `ρ ≥ 1.5, oscila` mantiene en vez de reducir. "Moverse menos" se consigue reduciendo α, que es una acción del agente; "reducir el residuo" no se consigue con ninguna acción directa, solo acercándose a la raíz. Por eso el potencial debe ser algo que el agente no controle. (Es la regla del README raíz: "el potencial no debe depender de lo que el agente controla".) Nota honesta: el efecto es una caída del éxito y dos estados torcidos, no un colapso total como en el lab 1; la razón es que el episodio solo termina con |f| < tol, así que el agente sigue teniendo incentivo a converger.

**(j)** Profundidad 5 (QED → POST → LIM → GEO → ERR → DEF/EXIST/H2). DEF y H2 se usan en varios sitios. `dibujar` lo muestra como árbol; el grafo en papel debe fusionar los nodos repetidos.

**(k)** El teorema pide contracción **en todo [a, b]** y que g lleve [a, b] en sí mismo (H1): eso garantiza convergencia global en el intervalo. El distractor solo tiene |g′(p)| < 1, que da contracción en un entorno de p: convergencia local, y solo si x₀ cae en ese entorno. Sin H1 la iteración puede salirse de la zona contractiva. Conecta con la fase 1: el agente a veces diverge (255 de 3000) precisamente cuando el x₀ inicial cae lejos y α mal elegido lo saca.

**(l)** La k de ERR es la constante de contracción de H2. Sin H2 no hay k, o solo se tiene k = 1 (Lipschitz con constante 1, que es lo que da |g′| ≤ 1). Con k = 1, GEO da |xₙ − p| ≤ |x₀ − p|: acotado, no convergente; LIM se cae. Es exactamente el distractor D_K1 colado por la puerta de atrás. Ejemplo para la pizarra: g(x) = x + 1 en ℝ, o g(x) = −x en [−1, 1], que cumple |g(x) − g(y)| = |x − y| y no converge.

**(m)** Quien escribió las `deps`. El verificador solo mira flechas.

**Tarea opcional.** Con `RATIO_EDGES = (0.5, 1.0)` (3 cubos × 2 = 6 estados) el agente pierde la distinción "casi Newton → mantener" y dobla α de más; el éxito baja unos puntos, no se hunde. Medirlo antes de darla como tarea si se va a calificar.

## Trampas

* Cada proyecto tiene su propio `.venv`. Si intentan usar el de 01 desde 02, `fixpointrl` no existe.
* `RATIO_EDGES` y `theoretical_action` están acoplados: si cambian los cubos, la tabla "teoría" del informe deja de tener sentido. Solo para la tarea opcional, y revertir.
* En `ciego` y `potencial` el trainer se reentrena con un entorno distinto pero la misma semilla; las cifras del caso [A] deben coincidir con `hitos`.
