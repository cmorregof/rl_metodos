# Contexto del curso · métodos numéricos con Reinforcement Learning

> Este archivo es el contexto que se pega en las instrucciones de un Proyecto de Claude
> (o que Claude lee al abrir el repo) para actuar como asesor pedagógico del curso.
> Última revisión: 22 de septiembre de 2026.

## Quién y para quién

* **Docente:** Carlos Orrego. Autor del repo `rl_metodos`.
* **Curso:** métodos numéricos, carrera de matemáticas, estudiantes de **2.º a 4.º semestre**.
  Se asume: cálculo diferencial e integral, sucesiones y límites, Bolzano y valor medio,
  algo de Python. No se asume: análisis real formal, probabilidad, ni RL.
* **Rol de Claude:** asesor y asistente pedagógico. Diseña laboratorios, notas del docente,
  rúbricas, y contrasta con el código real del repo (correr los scripts antes de afirmar
  tiempos o resultados).

## Objetivo del curso (formulado por el docente)

Que los estudiantes aprendan el *approach* de RL aplicado a métodos numéricos, con la vista
puesta en que, al final del curso, **cada uno "demuestre su propio teorema vía RL" y lo
aplique a un problema difícil**.

### Cómo se traduce eso en términos honestos

En los proyectos 01–05 el agente hace dos cosas distintas:

1. **Fase 1 · redescubrir el método.** Sin conocer el algoritmo, a base de recompensas,
   una tabla Q converge al método del libro (λ = 1/2, la ley de control de Banach, Armijo,
   Dekker…). Aquí sí hay descubrimiento genuino, y sirve para enseñar *por qué* el método
   es como es (el punto medio es el corte minimax; el paso de Newton es el punto fijo óptimo).
2. **Fase 2 · ordenar una demostración.** El teorema está escrito a mano como grafo de
   lemas con dependencias y distractores. El agente aprende un orden topológico válido y a
   rechazar los distractores. **No demuestra nada nuevo.** Lo que tiene valor formativo es
   que el estudiante escriba el grafo: eso le obliga a explicitar qué depende de qué, y a
   inventar distractores que son sus propios errores.

Por tanto, "demostrar su propio teorema vía RL" significa, para un estudiante de 2.º–4.º
semestre:

* elegir un teorema del curso (o una variante) que **no esté** en el repo;
* escribir su `proof_kb.py`: ≥ 10 pasos con dependencias correctas, ≥ 5 distractores
  que sean errores plausibles, y comprobar que el cierre transitivo desde ∎ es la
  demostración mínima;
* diseñar un entorno de fase 1 donde el agente redescubra un parámetro o una decisión del
  método asociado (opcional pero deseable);
* entrenar, leer el informe, y **defender** por qué el grafo es una demostración.

El "problema difícil" al final es el paso al patrón del 06: un **verificador independiente**
(no un grafo escrito a mano) que puntúa propuestas. Versiones accesibles para este nivel:
pesos de una cuadratura verificados exactamente con `fractions`, el ω óptimo de SOR
para una matriz dada verificado con el radio espectral, el paso constante óptimo del
descenso de gradiente para funciones cuadráticas (donde el peor caso es calculable a mano).
El 06 completo (SDP, PEP, LLM) queda como demostración del docente, no como tarea.

## Arco del curso: 10 laboratorios de 90 minutos

Decidido el 22 sep 2026: sesiones de 90 min en los PCs del laboratorio (Windows), estudiantes
sin experiencia en terminal, 10 laboratorios, no todos de búsqueda de raíces. Los labs 5–8
necesitan **proyectos nuevos en el repo** (07–10) con la misma anatomía que 01–05.

| lab | tema | proyecto | fase 1: qué redescubre el agente | fase 2: teorema | estado |
|---|---|---|---|---|---|
| 1 | terminal + bisección | 01 | λ = 1/2 e invariante de Bolzano | convergencia de la bisección | listo |
| 2 | punto fijo | 02 | ley de control de α (Banach) | punto fijo de Banach | código listo, guía pendiente |
| 3 | Taylor | 03 | cuántos términos, cuándo parar, cuándo rendirse | Taylor con resto de Lagrange | código listo, guía pendiente |
| 4 | Newton y secante | 04 + 05 | salvaguardas (Armijo, Dekker); órdenes 2 y φ | convergencia cuadrática / orden φ | código listo, guía pendiente (una sesión con los dos) |
| 5 | interpolación | 07 (nuevo) | dónde poner los nodos: Chebyshev emerge frente a equiespaciados (Runge) | error de interpolación de Lagrange | por construir |
| 6 | integración | 08 (nuevo) | cuándo subdividir: Simpson adaptativo emerge | error del trapecio / Simpson | por construir |
| 7 | sistemas lineales | 09 (nuevo) | ajustar ω en SOR / elegir Jacobi vs Gauss–Seidel | convergencia con diagonal dominante | por construir |
| 8 | ecuaciones diferenciales | 10 (nuevo) | control del paso h: RK adaptativo emerge | convergencia de Euler (Lipschitz, error global O(h)) | por construir |
| 9 | taller: tu teorema | plantilla | cada equipo escribe su `proof_kb` y su entorno de fase 1 | el teorema elegido por el equipo | por construir (plantilla `00_plantilla`) |
| 10 | verificadores y proyectos | mini-06 | patrón del 06 en pequeño: `verifica(propuesta) -> puntuación exacta` | presentación de proyectos | por construir |

Entregables por lab: guía del estudiante, notas del docente con cifras medidas, script de
experimentos. Reutilizar la estructura de `docs/curso/lab01/`.

## Reglas para Claude al diseñar material

* Verificar afirmaciones corriendo el código (`--no-tui` tarda < 1 s). No inventar cifras.
* Los experimentos de laboratorio se hacen sobre copias o con constantes editadas y
  revertidas con `git checkout`; nunca cambiar el comportamiento de los paquetes.
* Cada laboratorio lleva: guía del estudiante, notas del docente con resultados esperados
  y tiempos, y un script de apoyo en `docs/curso/labNN/`.
* Lenguaje directo y sin tecnicismos innecesarios; los estudiantes son de 2.º–4.º semestre.
* Ser explícito sobre qué es descubrimiento del agente y qué está escrito a mano.

## Estado

* 22 sep 2026: se crea este contexto, la guía de terminal desde cero y el laboratorio 1
  (bisección, 90 min). Decidido: 10 labs de 90 min, PCs del laboratorio, parejas.
  Pendiente: peso del proyecto final; construir 07–10 y la plantilla del lab 9;
  comprobar en un PC del laboratorio que Python y `pip` funcionan.
