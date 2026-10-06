# Biblioteca de demostraciones del curso

Aquí viven los teoremas escritos como grafo por los estudiantes del curso de métodos numéricos.
Cada entrada es un archivo `.py` con el mismo formato que los `proof_kb.py` de los proyectos:
una lista `STEPS` de pasos `Step(clave, título, texto, deps, distractor, is_qed)`, más
`TEOREMA` (el enunciado) y `AUTORES`.

## Qué es y qué no es

Un grafo dice **qué paso usa qué**. El verificador comprueba que cada paso se afirme después de
sus dependencias, que no se repita y que no sea un distractor. No comprueba que cada afirmación sea
verdadera ni que las flechas estén completas: eso es responsabilidad de quien escribe el grafo.
Por eso cada entrada lleva los nombres de sus autores y pasa por revisión del docente antes de
entrar. Los distractores son la parte más valiosa: son errores reales que alguien podría cometer.

## Cómo se revisa una entrada

Desde `07_interpolacion` (cualquier proyecto con `interprl` instalado sirve):

```
.venv\Scripts\python ..\docs\curso\lab03\experimentos.py mi_grafo ..\docs\curso\biblioteca\<archivo>.py
```

Revisa la estructura (claves repetidas, dependencias inexistentes, ciclos, ∎ sin dependencias),
dibuja el grafo desde ∎ y pone al agente de la fase 2 a buscar un orden válido.

## Nombres

`<lab>_<tema>_<apellido1>_<apellido2>.py`, por ejemplo `lab03_suma_basales_perez_gomez.py`.

## Entradas

| archivo | teorema | autores | laboratorio |
|---|---|---|---|
| (todavía ninguna: las primeras llegan con el laboratorio 3) | | | |
