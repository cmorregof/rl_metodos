r"""Laboratorio 3 · tu primera entrada en la biblioteca de demostraciones del curso.

Aquí escribes, como grafo, la demostración que hiciste en papel en la parte 2 (d):
«para n + 1 nodos distintos, la suma de los basales de Lagrange es 1 en todo t».

Cada paso es una línea Step(clave, título, texto, deps). En deps van las claves de los
pasos que ese paso USA. Las hipótesis no usan nada: deps=(). El paso final lleva is_qed=True.
Un distractor es un paso que suena bien pero está mal: distractor=True y sin deps.

Comprueba tu grafo con (desde 07_interpolacion):

    .venv\Scripts\python ..\docs\curso\lab03\experimentos.py mi_grafo

El programa revisa que el grafo esté bien formado, lo dibuja y pone al agente de la
fase 2 a buscar un orden válido. El agente no sabe matemáticas: solo respeta tus flechas.
Si una flecha falta, el hueco es tuyo.

Cuando termines, cambia el nombre del archivo a mi_grafo_<apellido1>_<apellido2>.py y
entrégalo. Los grafos buenos entran a la biblioteca del curso con sus autores.
"""

from interprl.proof_kb import Step

TEOREMA = "Si x₀, …, xₙ son distintos, entonces L₀(t) + L₁(t) + … + Lₙ(t) = 1 para todo t."
AUTORES = "escribe aquí sus nombres"

STEPS = [
    # --- hipótesis: no dependen de nada ---
    Step("H1", "Hipótesis", "Los nodos x₀, …, xₙ son distintos.", deps=()),

    # --- un paso ya escrito como ejemplo: la definición necesita H1 para poder dividir ---
    Step("DEF", "Definición", "Lₖ(t) = ∏_{j≠k} (t − xⱼ)/(xₖ − xⱼ): un polinomio de grado n.", deps=("H1",)),

    # --- escribe aquí tus pasos, uno por línea, en el orden que quieras ---
    # Step("CLAVE", "Título corto", "Lo que afirma el paso.", deps=("CLAVE_DE_LO_QUE_USA", ...)),

    # --- tu distractor: un error que tú podrías haber escrito en un examen ---
    # Step("D_MIO", "Distractor", "Un paso tentador que está mal.", distractor=True),

    # --- la conclusión: ¿qué pasos usa? ---
    Step("QED", "Conclusión", "L₀(t) + … + Lₙ(t) = 1 para todo t. ∎", deps=(), is_qed=True),
]
