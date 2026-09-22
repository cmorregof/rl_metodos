import re, sys
from pathlib import Path
import markdown
from fpdf import FPDF

src, dst = Path(sys.argv[1]), Path(sys.argv[2])
md = src.read_text(encoding="utf-8")
md = re.sub(r"\]\((?!http)[^)]+\)", "]", md)  # enlaces relativos: solo el texto
md = re.sub(r"\[([^\]]+)\]\b", r"\1", md)
html = markdown.markdown(md, extensions=["tables", "fenced_code"])
# fpdf2 write_html: normalizar bloques de código y tablas
html = html.replace("<pre><code>", "<pre>").replace("</code></pre>", "</pre>")
html = re.sub(r'<pre class="[^"]*">', "<pre>", html)
html = html.replace("<table>", '<table border="1" width="100%">')
html = re.sub(r"<th>", '<th width="25%">', html)
def _strip(m):
    inner = re.sub(r"<[^>]+>", "", m.group(2))
    return m.group(1) + inner + m.group(3)
html = re.sub(r"(<t[dh](?:\s[^>]*)?>)(.*?)(</t[dh]>)", _strip, html, flags=re.S)

class PDF(FPDF):
    def footer(self):
        self.set_y(-12); self.set_font("Arial", size=8); self.set_text_color(120)
        self.cell(0, 8, f"{src.stem}  ·  página {self.page_no()}", align="C")

pdf = PDF(format="A4")
pdf.set_margins(18, 16, 18)
pdf.set_auto_page_break(True, 18)
pdf.add_font("Arial", "", "/System/Library/Fonts/Supplemental/Arial Unicode.ttf")
pdf.add_font("Arial", "B", "/System/Library/Fonts/Supplemental/Arial Unicode.ttf")
pdf.add_font("Arial", "I", "/System/Library/Fonts/Supplemental/Arial Unicode.ttf")
pdf.add_font("Arial", "BI", "/System/Library/Fonts/Supplemental/Arial Unicode.ttf")
for st in ("", "B", "I", "BI"):
    pdf.add_font("Menlo", st, "/System/Library/Fonts/Menlo.ttc")
pdf.set_font("Arial", size=10.5)
pdf.add_page()
pdf.write_html(html, font_family="Arial", pre_code_font="Menlo"  # fpdf2 avisa que está obsoleto; funciona)
pdf.output(str(dst))
print("ok", dst, pdf.page_no(), "páginas")
