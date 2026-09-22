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

Aclaración del docente (22 sep 2026): "demostrar su propio teorema" significa **algo nuevo en
métodos numéricos**, no un teorema del libro reescrito como grafo. Eso fija el patrón del
final del curso: el del proyecto 06, no el de la fase 2 de 01–05.

En los proyectos 01–05 el agente hace dos cosas distintas:

1. **Fase 1 · redescubrir el método.** Sin conocer el algoritmo, a base de recompensas,
   una tabla Q converge al método del libro (λ = 1/2, la ley de control de Banach, Armijo,
   Dekker…). Sirve para enseñar *por qué* el método es como es.
2. **Fase 2 · ordenar una demostración.** El teorema está escrito a mano como grafo de
   lemas con dependencias y distractores. El agente aprende un orden válido y a rechazar
   distractores. **No demuestra nada nuevo.** Su valor formativo es que el estudiante escriba
   el grafo: explicitar qué depende de qué e inventar distractores que son sus propios errores.

Para que al final haya **algo nuevo**, hace falta el tercer ingrediente, el del 06: un
**verificador independiente** que certifica cada propuesta, y una búsqueda (RL, entropía
cruzada, o un LLM) que propone. Un "teorema nuevo" a este nivel es una afirmación concreta y
certificada del tipo "estos 9 nodos tienen constante de Lebesgue 2.31, menor que la de
Chebyshev (2.45)", o "estos pesos de cuadratura integran exactamente hasta grado 7 con esta
función peso", con certificado exacto (racionales, o cota de error verificable). Pequeño, pero
nuevo y verificable. Eso es alcanzable en un semestre si desde el lab 5 cada proyecto trae su
**fase 3: verificador + búsqueda**.

Por tanto, a partir del proyecto 07 cada proyecto tiene tres fases:

| fase | qué hay | quién demuestra |
|---|---|---|
| 1 · método | entorno de RL, el agente redescubre el método del libro | nadie: es descubrimiento del método |
| 2 · demostración | grafo de lemas del teorema clásico + distractores | el autor del grafo |
| 3 · frontera | `verify.py`: certifica una propuesta; `search`: busca propuestas mejores que la referencia | el verificador; el resultado es nuevo si supera lo tabulado |

El proyecto final del estudiante: elegir un problema con verificador accesible, correr la
búsqueda, y presentar el mejor resultado certificado con su certificado. El 06 completo
(SDP, PEP, LLM) queda como demostración del docente.

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
| 5 | interpolación | 07 (nuevo) | dónde poner los nodos (Chebyshev emerge frente a equiespaciados, Runge) y cuántos | error de interpolación de Lagrange; fase 3: constante de Lebesgue como verificador, búsqueda de nodos mejores que Chebyshev | en construcción (22 sep 2026) |
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
