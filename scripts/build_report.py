#!/usr/bin/env python3
"""Genera un PDF A4 profesional del informe español de Evaluación 03.

Uso: python scripts/build_report.py
Dependencias: markdown, weasyprint, pypdf, Pillow.
Se valida que sean como máximo 10 páginas sin contar la portada.
"""
from __future__ import annotations

from pathlib import Path
import html
import logging
import re

from markdown import markdown
from PIL import Image
from pypdf import PdfReader
from weasyprint import HTML

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "report" / "INFORME_FINAL_E03.md"
OUTPUT = ROOT / "report" / "Informe_Tecnico_E03_YOLO.pdf"
ASSET = ROOT / "assets" / "figures"


def crop_gallery() -> None:
    """Exporta paneles 1 y 4 de la galería real-vs-predicho ejecutada en Test."""
    path = ASSET / "test_gallery.png"
    if not path.exists():
        raise FileNotFoundError(path)
    with Image.open(path) as im:
        w, h = im.size
        for row, label in [(0, "confusion"), (3, "false_positive"), (5, "correct")]:
            upper = int(h * row / 6)
            lower = int(h * (row + 1) / 6)
            image = im.crop((0, upper, w, lower)).convert("RGB")
            if image.width > 1100:
                image.thumbnail((1100, 600), Image.Resampling.LANCZOS)
            image.save(ASSET / f"test_{label}.jpg", quality=82, optimize=True)


def make_html(font_size=10.0) -> str:
    raw = SOURCE.read_text(encoding="utf-8")
    title_part, contents = raw.split("\n---\n", 1)
    heading = "Detección automática de maquinaria de construcción"
    title_html = markdown(title_part, extensions=["tables", "fenced_code"])
    main_html = markdown(contents, extensions=["tables", "fenced_code"])
    figure_pattern = r'<img[^>]*test_gallery[.]png[^>]*>'
    replacement = """
        <div class="gallery">
          <div><img alt="Confusión clase" src="../assets/figures/test_confusion.jpg">
          <small>Vista cenital: cargadora y etiqueta adicional de excavadora.</small></div>
          <div><img alt="Detección falsa" src="../assets/figures/test_false_positive.jpg">
          <small>Cargadora: detección espuria de camión de volteo.</small></div>
          <div><img alt="Acierto de camion" src="../assets/figures/test_correct.jpg">
          <small>Ejemplo representativo de detección correcta de camión.</small></div>
        </div>
    """
    main_html, count = re.subn(figure_pattern, lambda m: replacement, main_html, flags=re.S)
    if count != 1:
        raise RuntimeError(f"Se esperaban 1 galería, se encontraron {count}")

    css = f"""
    @page {{
      size: A4; margin: 16mm 17mm 19mm 17mm;
      @top-left {{content:"UNIVERSIDAD BERNARDO O'HIGGINS  /  MACHINE LEARNING";
                  font-family:DejaVu Sans, sans-serif; font-size:7.2pt;
                  font-weight:600; color:#64748b}}
      @top-right {{content:"EVALUACIÓN 03  ·  YOLO26n";
                  font-family:DejaVu Sans, sans-serif; font-size:7.2pt; color:#64748b}}
      @bottom-left {{content:"M. Marsault · J. Martinat · L. Maillet";
                     font-family:DejaVu Sans, sans-serif; font-size:7.2pt; color:#64748b}}
      @bottom-right {{content:"PÁG. " counter(page) "  /  " counter(pages);
                      font-family:DejaVu Sans, sans-serif; font-size:7.2pt; color:#64748b}}
    }}
    @page portada {{margin:0; @top-left{{content:none}} @top-right{{content:none}}
                   @bottom-left{{content:none}} @bottom-right{{content:none}}}}
    * {{ box-sizing: border-box; }}
    html {{ font-family: DejaVu Sans, Arial, sans-serif; color:#1d2939; }}
    body {{ margin:0; font-size:{font_size}pt; line-height:1.42; }}
    section.cover {{page:portada; break-after:page; width:210mm; height:297mm;
                   background:linear-gradient(145deg,#09192e 0%,#132a47 60%,#0b4a5b 100%);
                   color:#f8fafc; padding:28mm 25mm; position:relative;}}
    .cover .eyebrow {{font-size:10pt; color:#6ee7ce; text-transform:uppercase;
                      letter-spacing:2.0pt; margin-top:21mm; font-weight:700;}}
    .cover .cover-title {{font-size:31pt; line-height:1.16; letter-spacing:-0.9pt;
                         color:white; font-weight:800; margin:12mm 0 5mm 0;}}
    .cover .subtitle {{font-size:13pt; line-height:1.5; color:#bfd1df; max-width:160mm;}}
    .cover .stripe {{width:20mm; height:2.2mm; background:#5eead4; margin:12mm 0;}}
    .cover .metrics {{margin-top:16mm; display:flex; gap:4mm;}}
    .cover .metric {{background:rgba(255,255,255,.10); width:46mm;
                     border:0.5pt solid rgba(255,255,255,.22);
                     border-radius:8pt; padding:5mm 4mm;}}
    .cover .metric strong {{display:block; font-size:21pt; color:#f9fafb;}}
    .cover .metric small {{font-size:9pt; color:#c7d6e4;}}
    .cover .byline {{margin-top:19mm; padding-top:7mm;
                    border-top:.5pt solid #527088; font-size:10pt; color:#dce8f3;}}
    .cover .byline p {{margin:2mm 0;}}
    main {{ margin:0; }}
    h2 {{font-size:17pt; margin:8mm 0 3mm 0; padding-bottom:2mm;
         color:#142b49; border-bottom:1.3pt solid #cde2e6;
         break-after:avoid;}}
    h3 {{font-size:11.5pt; color:#15556c; margin:4.5mm 0 1.5mm;
         break-after:avoid;}}
    p {{margin:2.2mm 0 3mm; text-align:justify; orphans:3; widows:3;}}
    strong {{color:#182e47;}}
    a {{color:#087a85; text-decoration:none; overflow-wrap:anywhere;}}
    code {{font-family:DejaVu Sans Mono, monospace; color:#15556c;
           font-size:0.86em; background:#eaf4f5; border-radius:2pt; padding:0 1.5pt;}}
    table {{border-collapse:collapse; width:100%; margin:3mm 0 4mm; font-size:8.1pt;
            table-layout:auto; break-inside:avoid;}}
    th {{background:#173a59; color:white; text-align:left; font-weight:700;}}
    td,th {{border:.5pt solid #d9e4ec; padding:4pt 5pt; vertical-align:top;}}
    tr:nth-child(even) td {{background:#f4f8fa;}}
    table td:not(:first-child), table th:not(:first-child) {{text-align:center;}}
    img {{display:block; margin:1.5mm auto 1.5mm; max-width:100%; max-height:64mm;
          object-fit:contain;}}
    .gallery {{display:flex; flex-wrap:wrap; justify-content:space-between;
              gap:3mm; margin:2mm 0;}}
    .gallery div {{width:48%; break-inside:avoid;}}
    .gallery div img {{width:100%; max-height:49mm; object-fit:contain;}}
    .gallery small {{font-size:7.3pt; color:#556579; display:block; margin-top:1mm;}}
    p:has(>em:only-child) {{font-size:8.2pt; color:#64748b; font-style:normal;
                            margin:1mm 0 4mm;}}
    hr {{border:0; border-top:1pt solid #e2e8f0; margin:5mm 0;}}
    ol li {{margin-bottom:2mm; overflow-wrap:anywhere; font-size:8.6pt;}}
    blockquote {{margin:3mm 0; padding:2mm 4mm; background:#f3f9fa;
                border-left:3pt solid #0f8490;}}
    """
    title_block = f"""
    <section class="cover">
      <div class="eyebrow">UBO · Evaluación 03 · Machine Learning</div>
      <div class="cover-title">Detección automática<br>
        de maquinaria de<br> construcción</div>
      <div class="subtitle">YOLO26n · Transfer learning · Fine tuning<br>
        Evidencia experimental reproducible, evaluación y análisis crítico</div>
      <div class="stripe"></div>
      <div class="metrics">
        <div class="metric"><strong>86,59 %</strong><small>mAP50 · Test</small></div>
        <div class="metric"><strong>71,66 %</strong><small>mAP50-95 · Test</small></div>
        <div class="metric"><strong>90,13 %</strong><small>Precision · Test</small></div>
      </div>
      <div class="byline">
        <p><b>Autores</b> · Mathis Marsault · Julien Martinat · Louis Maillet</p>
        <p><b>Docente</b> · Dr. Eliecer Peña Ancavil</p>
        <p><b>Universidad</b> · Universidad Bernardo O'Higgins, Santiago</p>
        <p><b>Entrega</b> · 7 de noviembre de 2026</p>
        <p><b>Grupo de trabajo</b> · Marsault · Martinat · Maillet (3 integrantes)</p>
        <p><b>Módulo académico</b> · Machine Learning</p>
      </div>
    </section>
    """
    return f'<!doctype html><html lang="es"><head><meta charset="utf-8"><title>{html.escape(heading)} — UBO</title><style>{css}</style></head><body>{title_block}<main>{main_html}</main></body></html>'


def main():
    logging.getLogger("weasyprint").setLevel(logging.WARNING)
    crop_gallery()
    for font in (11.0, 10.8, 10.6, 10.4, 10.2, 10.0):
        document = HTML(string=make_html(font), base_url=str(ROOT / "report")).render()
        pages = len(document.pages)
        print(f"Prueba de maqueta: {pages} páginas con fuente {font} pt")
        if pages <= 11:
            document.write_pdf(str(OUTPUT))
            final_pages = len(PdfReader(str(OUTPUT)).pages)
            if final_pages != pages:
                raise RuntimeError(f"Conteo discrepante: {final_pages} / {pages}")
            print(f"PDF creado: {OUTPUT}, {pages} páginas (portada + {pages - 1} del informe).")
            return
    raise RuntimeError("El informe supera 10 páginas sin contar portada. Revisar diseño o texto.")


if __name__ == "__main__":
    main()
