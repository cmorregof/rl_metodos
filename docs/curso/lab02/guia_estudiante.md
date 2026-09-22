# Laboratorio 2 · Aprender a controlar una iteración

**Métodos numéricos · Matemáticas · 90 minutos · en parejas**

En el laboratorio 1 el agente aprendió *dónde cortar*. Hoy aprende algo más difícil: **ajustar un
parámetro sobre la marcha** mirando solo lo que la iteración le deja ver. Y el teorema es el de
Banach, el que está detrás de casi todos los métodos iterativos del curso.

Todas las órdenes se escriben dentro de `02_punto_fijo` y están escritas para Windows. Mac/Linux:
`.venv\Scripts\python` → `.venv/bin/python` y `..\docs\curso\lab02\experimentos.py` →
`../docs/curso/lab02/experimentos.py`.

## Qué vas a hacer

1. Ver que un parámetro fijo no sirve, y que un agente que solo mira dos números lo ajusta bien.
2. Quitarle al agente información y ver qué deja de poder aprender (**el estado importa**).
3. Darle una recompensa de progreso que él mismo puede manipular (**el potencial importa**).
4. Dibujar el grafo del teorema de Banach y encontrar el hueco cuando falta la contracción.

## 0 · Instalación (0–10 min)

```
cd 02_punto_fijo
python -m venv .venv
.venv\Scripts\python -m pip install -e ".[dev]"
.venv\Scripts\python -m fixpointrl --no-tui --seed 1
```

Si el ZIP es de hoy, la carpeta `.venv` de `01_biseccion` no sirve aquí: cada proyecto tiene la suya.

## 1 · El problema (10–20 min, en papel)

Para resolver `f(x) = 0` iteramos `x ← g(x)` con `g(x) = x − α·f(x)`. La raíz `r` es punto fijo
de `g` **para cualquier α**. Pero la iteración solo converge si `g` es contracción cerca de `r`:
`|g′(r)| = |1 − α·f′(r)| < 1`.

El agente **no conoce f′ ni r**. Tras cada iteración solo ve dos cosas medibles:

* `ρ = |f(xₙ₊₁)| / |f(xₙ)|`, la razón de residuos consecutivos (en 6 cubos);
* si el residuo **cambió de signo** (oscila) o no (monótona).

Y puede: mantener α, doblarlo, reducirlo a la mitad o invertir su signo.

- (a) Para `f(x) = 2(x − 1)`, ¿para qué valores de α converge la iteración desde cualquier `x₀`?
  ¿Y qué α la hace converger en un solo paso?
- (b) Cerca de la raíz, `ρ ≈ |1 − α·f′(r)|`. Rellena con la acción que tú harías: si `ρ > 1` y la
  iteración es monótona, ¿qué está mal? ¿Y si `ρ > 1` y oscila? ¿Y si `ρ < 1` pero oscila?
- (c) Predice: ¿en cuál de los 12 estados (6 cubos de ρ × monótona/oscila) es más difícil decidir?

## 2 · Sin agente: α fijo (20–30 min)

```
.venv\Scripts\python ..\docs\curso\lab02\experimentos.py alfa_fijo
```

- (d) ¿Cuál es el mejor α fijo y en qué porcentaje de problemas converge? ¿Por qué ningún α fijo
  puede funcionar siempre? (Pista: el signo de f′(r) cambia de un problema a otro.)

## 3 · Mira cómo aprende (30–45 min)

```
.venv\Scripts\python -m fixpointrl --seed 1
```

Anota la generación de cada hito. Luego corre con `--seed 2` y completa la segunda columna.

| hito | gen. (seed 1) | gen. (seed 2) |
|---|---|---|
| "si diverge monótonamente, el signo de α está mal" | | |
| primera convergencia | | |
| "si oscila, α es demasiado grande" | | |
| ley coincide con Banach en los 12 estados | | |
| primera demostración mínima | | |

Abre `runs\<fecha>\informe.md` y mira la tabla **"ley de control aprendida"**.

- (e) Compárala con tu tabla de (b). ¿En qué estados coincide con lo que tú habrías hecho?
- (f) En el estado `ρ ≥ 1.5, monótona` la teoría admite dos acciones (invertir o reducir).
  ¿Por qué las dos son razonables ahí?

## 4 · El estado importa (45–60 min)

Le quitamos al agente el bit de oscilación: ahora solo ve ρ.

```
.venv\Scripts\python ..\docs\curso\lab02\experimentos.py ciego
```

- (g) ¿Cuánto cae el éxito? En los estados con `ρ > 1`, ¿qué acción aprende y por qué es la
  mejor que puede tomar **sin saber** si oscila?
- (h) Propón otro observable (algo que se pueda medir sin conocer f′ ni r) que le ayudaría más
  que el bit de oscilación, o argumenta que no hace falta.

## 5 · El potencial importa (60–72 min)

La recompensa de progreso es `log₁₀(|f(xₙ)| / |f(xₙ₊₁)|)`: premia reducir el residuo, que el agente
no controla directamente. Vamos a cambiarla por `log₁₀(|paso anterior| / |paso nuevo|)`: premia
**moverse cada vez menos**.

```
.venv\Scripts\python ..\docs\curso\lab02\experimentos.py potencial
```

- (i) ¿Cuánto cae el éxito? Mira los estados marcados ✘: ¿qué hace ahora el agente cuando
  oscila cerca de la raíz (`ρ < 0.25`)? ¿Por qué "moverse menos" le da puntos aunque no le
  acerque a la raíz? ¿Con qué acción se consigue moverse menos sin resolver nada?

## 6 · El teorema de Banach como grafo (72–85 min)

Abre `fixpointrl\proof_kb.py`: 13 pasos con dependencias y 7 distractores.

- (j) Dibuja el grafo en papel (nodos, flechas, hipótesis, ∎). Compáralo con:

```
.venv\Scripts\python ..\docs\curso\lab02\experimentos.py dibujar
```

- (k) El distractor `D_LOCAL` dice "`|g′(p)| < 1`, luego converge desde cualquier `x₀`". ¿Qué
  diferencia hay entre lo que dice el teorema y lo que dice el distractor? (Pista: local vs global;
  ¿qué hipótesis garantiza que la iteración no se sale de `[a, b]`?)

Ahora el hueco:

```
.venv\Scripts\python ..\docs\curso\lab02\experimentos.py grafo
```

Quitamos H2 (la contracción) de las dependencias de `ERR`. El verificador acepta una demostración
donde la contracción **se declara después de haberla usado**.

- (l) `ERR` dice `|xₙ₊₁ − p| ≤ k·|xₙ − p|`. ¿De dónde sale esa `k`? ¿Qué le pasa a la convergencia
  (`LIM`) si solo sabemos `k ≤ 1`? Relaciónalo con el distractor `D_K1`.
- (m) ¿Quién es responsable de que la demostración sea correcta: el agente, el verificador o
  quien escribió las `deps`?

## 7 · Cierre (85–90 min)

Entrega (una por pareja): la tabla de hitos, las respuestas (a)–(m) en la hoja de entrega, y un
párrafo: *¿qué observable elegirías para controlar el paso de Newton (laboratorio 4) sin conocer
f″?*

**Tarea opcional.** El estado tiene 12 combinaciones. Cambia `RATIO_EDGES` en `env_fixpoint.py`
a solo dos cortes, `(0.5, 1.0)`, corre `--no-tui` y compara el éxito. ¿Cuánta resolución en ρ hace
falta? Revierte el cambio después.
