# Fuentes LaTeX del material docente

Se compilan con XeLaTeX (TeX Live 2026 basic) y las fuentes del sistema de macOS
(Avenir Next, Menlo, Apple Symbols). `preambulo.tex` es común a todos los documentos.

```bash
cd docs/curso/tex
xelatex 00_terminal_desde_cero.tex
xelatex lab01_guia_estudiante.tex
xelatex lab01_notas_docente.tex
cp *.pdf ../pdf/
```

Los `.md` de `docs/curso/` siguen siendo la versión de referencia para leer en GitHub;
los `.tex` son la versión maquetada para imprimir o repartir. Al cambiar un `.md`, actualizar
el `.tex` correspondiente y recompilar.
