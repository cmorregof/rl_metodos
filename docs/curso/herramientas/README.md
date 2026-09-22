# Herramientas del material docente

## Markdown → PDF

Sin pandoc. Vía puramente Python (`markdown` + `fpdf2`), con Arial Unicode y Menlo del sistema (macOS).

```bash
python3 -m venv /tmp/pdfenv && /tmp/pdfenv/bin/pip install markdown fpdf2
cd docs/curso
/tmp/pdfenv/bin/python -W ignore herramientas/md2pdf.py 00_terminal_desde_cero.md pdf/00_terminal_desde_cero.pdf
/tmp/pdfenv/bin/python -W ignore herramientas/md2pdf.py lab01/guia_estudiante.md pdf/lab01_guia_estudiante.pdf
/tmp/pdfenv/bin/python -W ignore herramientas/md2pdf.py lab01/notas_docente.md pdf/lab01_notas_docente.pdf
```

Limitaciones: dentro de las celdas de tabla se pierde el formato (negrita, código); los enlaces
relativos quedan como texto. Los PDF de `pdf/` se regeneran a mano cuando cambian los `.md`.
