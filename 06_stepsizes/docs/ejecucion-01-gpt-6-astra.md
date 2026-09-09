# Ejecución 01 · `gpt-6-astra`, 2026-09-09 21:41–23:58

`python -m steprl evolve --provider openai --model gpt-6-astra --effort high --generations 20 --anytime-n 31` (métrica de la época: garantía anclada en ½; límite de salida 16k).

## Qué pasó

* 20 generaciones pedidas, 17 registradas, **6 válidas**. Las otras 11 se perdieron porque la respuesta se cortó por longitud: en la API de OpenAI los tokens de razonamiento cuentan dentro de `max_completion_tokens`, y con esfuerzo alto 16k no bastaban. Corregido: 64k y tolerancia a cortes con bloque completo.
* Varias llamadas tardaron 15–30 minutos (el cliente reintenta con 10 min de espera). Dos horas en total, de las que 7.5 minutos fueron CPU (los SDP).
* El programa de la generación 17 lleva `_HORIZON = 31` escrito a mano: **sobreajuste al horizonte de evaluación**. A N=63 su exponente por duplicación es −0.51. Corregido: la puntuación anytime es ahora el peor exponente entre varios horizontes (31 y 63 por defecto) y el prompt lo advierte.

## Resultados con la métrica corregida

Paso constante como referencia: exponente por duplicación 0.92 a N=31 y N=63 (tiende a 1); τ₆₃ = 0.00394.

| gen | idea del modelo (su comentario) | fijo (×óptimo) | dup p N=31 | dup p N=63 | τ₆₃ | const/τ₆₃ |
|---|---|---|---|---|---|---|
| 1 | silver con picos amortiguados ρ^(5/6) | 1.278 | 0.265 | — | — | — |
| 2 | √2 alternado con pasos largos que saturan en 1+ρ | 1.119 | 1.047 | 1.047 | 0.00176 | 2.23 |
| 3 | bloques silver de 3 unidos por «puentes» con presupuesto | 1.325 | 0.664 | — | — | — |
| 7 | composiciones asimétricas por programación dinámica | 1.218 | 0.934 | 0.934 | 0.00155 | 2.54 |
| 10 | árboles mixtos s/f-componibles (Grimmer–Shu–Wang) | 1.218 | 0.929 | 0.692 | 0.00199 | 1.97 |
| 17 | como 10, con enumeración exhaustiva y `_HORIZON = 31` | 1.218 | 0.932 | −0.513 | 0.00157 | 2.51 |

## Lectura

1. **Nadie acelera.** Todos los programas tienen pasos acotados (o picos), así que su orden es 1 y solo mejoran la constante: entre 2 y 2.5 veces mejor que el paso constante a N=63. Es exactamente lo que dice la teoría para pasos acotados.
2. **El modelo converge a la familia conocida.** Desde la generación 7 propone variantes de las composiciones s/f de Grimmer–Shu–Wang y aterriza en la misma nota (1.218× a horizonte fijo). No sale de esa familia porque el prompt no le decía que con pasos acotados no hay aceleración; ahora sí.
3. **La métrica anclada en ½ engañaba**: daba 1.46 a N=31, por encima de la cota inferior asintótica 1.334. Con el exponente por duplicación el mejor programa (gen 2) da 1.047, honesto y comparable con el 1.119 publicado.
4. **Sobreajuste al evaluador**: la generación 17 optimiza N=31 y se rompe después. Cualquier bucle LLM → verificador necesita horizontes que el programa no conozca.

## Siguiente ejecución

Misma orden con la métrica corregida, dos horizontes y 64k de salida. Objetivo concreto: un programa sin horizonte escrito con exponente por duplicación > 1.05 en N=31 y N=63 a la vez, y luego confirmarlo a N=127.
