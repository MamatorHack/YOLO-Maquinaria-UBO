"""Genera informe técnico PDF maquetado desde el informe verificado de GitHub.
Ejecución EN COLAB, después de los entrenamientos y la generación de evidencia:
  %pip install -q markdown weasyprint pypdf
  !python scripts/generar_informe_pdf.py
No evalúa Test, no vuelve a entrenar y no modifica los pesos.
"""
from __future__ import annotations
import html
import json
import re
from pathlib import Path
import markdown
from PIL import Image, ImageOps, ImageDraw

ROOT = Path("/content/YOLO-Maquinaria-UBO")
DOC = ROOT / "docs/INFORME_TECNICO_E03_PARA_PDF.md"
OUT = ROOT / "report"
OUT.mkdir(parents=True, exist_ok=True)
ASSETS = OUT / "figures"
ASSETS.mkdir(exist_ok=True)

def ref(p):
    return p.resolve().as_uri()

def plot_counts(path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    names=["EXCAVATORS","dump truck","wheel loader"]
    v=[[1477,1067,973],[35,120,55],[17,80,30]]
    fig,axes=plt.subplots(1,3,figsize=(11.5,2.7),sharey=True)
    for ax,label,values in zip(axes,("Train","Validation","Test"),v):
        bars=ax.bar(names,values,color=["#3756aa","#159681","#d7a04d"])
        ax.set_title(label)
        for bar in bars:ax.text(bar.get_x()+bar.get_width()/2,bar.get_height()+max(values)*.025,str(int(bar.get_height())),ha="center",fontsize=8)
        ax.tick_params(axis="x",labelrotation=22,labelsize=9)
        ax.spines[["top","right"]].set_visible(False)
        ax.set_ylim(0,1600)
    fig.tight_layout()
    fig.savefig(path,dpi=180,facecolor="white")
    plt.close(fig)

def plot_metrics(path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    labels=["Precision","Recall","mAP50","mAP50-95"]
    tr=[.886345,.800567,.870787,.712923]
    ft=[.831851,.834979,.874891,.707573]
    x=np.arange(len(labels))
    fig,ax=plt.subplots(figsize=(9.2,2.7))
    b1=ax.bar(x-.17,tr,.33,color="#244f9a",label="Transfer learning")
    b2=ax.bar(x+.17,ft,.33,color="#17a18e",label="Fine tuning")
    ax.set_xticks(x,labels)
    ax.set_ylim(0,1)
    ax.legend(loc="lower left",ncol=2,fontsize=9)
    ax.spines[["top","right"]].set_visible(False)
    ax.grid(axis="y",alpha=.12)
    for b in [*b1,*b2]:
        ax.text(b.get_x()+b.get_width()/2,b.get_height()+.013,f"{b.get_height():.3f}",ha="center",fontsize=8)
    fig.tight_layout()
    fig.savefig(path,dpi=175,facecolor="white")
    plt.close(fig)

def montage(images, output):
    loaded=[]
    for p in images:
        try:
            im=Image.open(p).convert("RGB")
            im.thumbnail((850,550))
            loaded.append(im)
        except Exception:
            pass
    if not loaded:
        return None
    cell_w,cell_h=850,550
    frame=Image.new("RGB",(cell_w*len(loaded),cell_h),"white")
    for i,im in enumerate(loaded):
        frame.paste(im,(i*cell_w+(cell_w-im.width)//2,(cell_h-im.height)//2))
    frame.save(output,quality=88)
    return output

def video_stills():
    try:
        import cv2
        photos=[]
        for folder in ("video_1_streaming","video_2_streaming"):
            root=ROOT/"results"/folder
            found=next((p for p in root.rglob("*.mp4") if p.is_file()),None) if root.exists() else None
            if not found:
                continue
            cap=cv2.VideoCapture(str(found))
            n=cap.get(cv2.CAP_PROP_FRAME_COUNT)
            cap.set(cv2.CAP_PROP_POS_FRAMES,int(n*.45))
            ok,frame=cap.read()
            cap.release()
            if ok:
                p=ASSETS/(folder+".jpg")
                cv2.imwrite(str(p),frame)
                photos.append(p)
        if not photos:
            return None
        return montage(photos,ASSETS/"videos_stills.jpg")
    except Exception as err:
        print("No se pudieron extraer fotogramas:",err)
        return None

plot_counts(ASSETS/"class_counts.png")
plot_metrics(ASSETS/"validation_metrics.png")
figs={
    1:ROOT/"results/annotations_train.png",
    2:ASSETS/"class_counts.png",
    4:ROOT/"results/transfer_curves.png",
    5:ROOT/"results/fine_curves.png",
    6:ASSETS/"validation_metrics.png",
}
gallery=ROOT/"results/test_gallery.png"
if gallery.exists():
    with Image.open(gallery) as img:
        # Primeros dos pares de cajas, para mantener legibilidad en A4.
        excerpt=img.crop((0,0,img.width,max(1,int(img.height/3))))
        excerpt.save(ASSETS/"test_gallery_extract.png")
    figs[7]=ASSETS/"test_gallery_extract.png"

ex_imgs=ROOT/"results/external_images"
if ex_imgs.is_dir():
    candidates=[p for p in sorted(ex_imgs.rglob("*")) if p.suffix.lower() in (".jpg",".png",".jpeg")]
    combined=montage(candidates[:2],ASSETS/"external_images.jpg")
    if combined:figs[8]=combined
video_image=video_stills()
if video_image:figs[9]=video_image

text=DOC.read_text(encoding="utf-8")
errors_path=ROOT/"results/error_analysis/casos.json"
error_figures=[]
if errors_path.exists():
    errors=json.loads(errors_path.read_text(encoding="utf-8"))
    if "objeto_perdido" in errors:
        e=errors["objeto_perdido"]
        # Sustituir el campo pendiente por la evidencia real encontrada visualmente.
        text=re.sub(
            r'- \*\*Objeto perdido \(FN\):\*\*.*',
            "- **Objeto perdido (FN):** en la imagen de Test `"+e["archivo"]+
            "` al menos una caja real no recibió una detección emparejada de la misma clase con confianza "
            "≥0,25 e IoU ≥0,50. La categoría omitida es "+
            ", ".join(e["explicacion"].get("clases",[]))+
            ". Este ejemplo no se utilizó para seleccionar o ajustar el modelo.",
            text
        )
    for key in ("objeto_perdido","falsa_deteccion","confusion_de_clases"):
        if key in errors and Path(errors[key]["imagen"]).exists():
            error_figures.append((key,Path(errors[key]["imagen"])))

def figure_tag(num, name, path):
    return ('<figure class="evidence"><img src="'+html.escape(ref(path))+
            '"><figcaption><b>Figura '+str(num)+'.</b> '+
            html.escape(name)+'</figcaption></figure>')

fig_names={
    1:"Ejemplos reales de entrenamiento y cajas etiquetadas.",
    2:"Distribución de cajas por clase y partición tras la limpieza.",
    4:"Evolución de las pérdidas y mAP durante el transfer learning.",
    5:"Evolución de las pérdidas y mAP durante el fine tuning.",
    6:"Métricas de Validation usadas para la selección del modelo.",
    7:"Comparación de etiquetas reales y detecciones en imágenes de Test.",
    8:"Resultados del modelo sobre dos fotografías externas.",
    9:"Fotogramas representativos de las dos demostraciones en video.",
}
# Sustituir líneas Figura N por la figura real. No reproducir placeholders de evidencia faltante.
for number in range(1,10):
    regex=r"(?m)^\*Figura\s+"+str(number)+r"[^\n]*$"
    if number in figs and figs[number].exists():
        text=re.sub(regex,figure_tag(number,fig_names[number],figs[number]),text)
    else:
        text=re.sub(regex,"",text)

if error_figures:
    extra="<div class='error-plates'>"
    for key,path in error_figures:
        extra+=figure_tag("7-"+key.replace("_","-"),"Inspección cualitativa: "+key.replace("_"," "),path)
    extra+="</div>"
    text=text.replace("## 6. Uso con imágenes y video externos",extra+"\n\n## 6. Uso con imágenes y video externos")

# La portada queda separada de la numeración del texto técnico.
parts=text.split("## 1. Problema y dataset",1)
if len(parts)!=2:
    raise ValueError("Falta la sección 1 del informe")
body="## 1. Problema y dataset"+parts[1]
body=re.sub(r'^# Detección.*?\n','',body)
has_pending=("PENDIENTE" in body)
if has_pending:
    print("AVISO: el informe contiene campos PENDIENTE. PDF preliminar, NO entregar sin revisar.")
html_report=markdown.markdown(body,extensions=["tables","fenced_code","attr_list"])
css="""
@page { size: A4; margin: 17mm 17mm 18mm 17mm;
  @bottom-right {content: "UBO  |  " counter(page);font:8pt 'DejaVu Sans';color:#596b84;}
  @top-right {content: "MACHINE LEARNING  /  EVALUACIÓN 03";font:7.5pt 'DejaVu Sans';color:#7788a1;}
}
@page:first{margin:0;@bottom-right {content:"";}@top-right {content:"";}}
body{font-family:'DejaVu Sans',Arial,sans-serif;color:#20334c;font-size:11pt;line-height:1.29;}
.cover {page-break-after:always;position:relative;height:280mm;box-sizing:border-box;
  background:#112949;color:white;padding:35mm 27mm 25mm 27mm;}
.cover .eyebrow{color:#91d9db;letter-spacing:3px;font-size:10pt;font-weight:700;}
.cover h1{font-size:29pt;line-height:1.12;color:white;max-width:145mm;margin:23mm 0 11mm;}
.cover .subtitle{font-size:13pt;color:#cddbee;max-width:145mm;margin-bottom:28mm;}
.cover .line{height:2px;width:33mm;background:#42b9ab;margin:15mm 0;}
.cover .details{font-size:11pt;line-height:2.05;}
.cover .footer{position:absolute;bottom:24mm;left:27mm;color:#abc3df;font-size:10pt;}
.content{padding:0;}
h2{font-size:15pt;color:#153f75;border-bottom:1px solid #c4d9e7;padding-bottom:4pt;margin:16pt 0 7pt;}
h2:first-child{margin-top:0;}
p{margin:5pt 0 8pt;orphans:2;widows:2;text-align:justify;}
strong{color:#193f6b;}
table{border-collapse:collapse;width:100%;margin:9pt 0 10pt;font-size:9.0pt;line-height:1.25;table-layout:auto;}
th{background:#174676;color:white;text-align:left;font-weight:bold;}
th,td{border:0.4pt solid #c7d4df;padding:5pt 6pt;vertical-align:top;}
tr:nth-child(even){background:#f0f6f9;}
tr{break-inside:avoid;}
figure{margin:7pt 0 12pt;text-align:center;break-inside:avoid;}
figure img{display:block;margin:0 auto;max-width:100%;max-height:98mm;object-fit:contain;}
figcaption{font-size:9pt;text-align:center;color:#496179;margin-top:4pt;}
.error-plates img{max-height:78mm;}
.error-plates figure{margin-bottom:7pt;}
ul,ol{padding-left:19pt;margin:4pt 0 9pt;}
li{margin-bottom:4pt;}
code{background:#eff3f8;padding:1pt 2pt;color:#14496a;font-size:9.3pt;overflow-wrap:anywhere;}
a{color:#12699e;overflow-wrap:anywhere;}
blockquote{border-left:3px solid #1da59d;padding-left:10pt;color:#4e6078;}
"""
cover="""<section class="cover">
<div class="eyebrow">UNIVERSIDAD BERNARDO O'HIGGINS</div>
<div class="line"></div>
<h1>Detección automática de maquinaria de construcción mediante YOLO26n</h1>
<div class="subtitle">Evaluación 03 · Proyecto de detección de objetos</div>
<div class="details"><b>Mathis Marsault</b><br><b>Julien Martinat</b><br><b>Louis Maillet</b><br>
Machine Learning<br>Profesor: Dr. Eliecer Peña Ancavil<br>7 de noviembre de 2026</div>
<div class="footer">Documento técnico  |  Transfer learning · Fine tuning · Validation · Test</div>
</section>"""
document="<html><head><meta charset='utf-8'><style>"+css+"</style></head><body>"+cover+"<div class='content'>"+html_report+"</div></body></html>"
from weasyprint import HTML
pdf_name="Informe_E03_YOLO_BORRADOR.pdf" if has_pending else "Informe_E03_YOLO_FINAL.pdf"
pdf=OUT/pdf_name
HTML(string=document,base_url=ROOT.as_uri()+"/").write_pdf(str(pdf))
try:
    from pypdf import PdfReader
    pages=len(PdfReader(str(pdf)).pages)
    print(f"PDF generado: {pdf} | {pages} páginas totales (incluida la portada)")
    if pages>11:
        print("ATENCIÓN: excede las 10 páginas permitidas sin portada; reducir figuras o texto antes de entregar.")
except Exception as err:
    print("No se pudo verificar el número de páginas:",err)
print("Figuras insertadas:",sorted(set(figs)))
print("Casos de errores:",[key for key,_ in error_figures])
print("Estado:","BORRADOR (completar pendientes)" if has_pending else "CANDIDATO A ENTREGA (revisión humana obligatoria)")
