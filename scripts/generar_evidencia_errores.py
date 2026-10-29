"""Construye evidencia cualitativa de tres errores en Test SIN reevaluar métricas.
El modelo ya está seleccionado en Validation. Esta inspección de predicciones
sobre fotografías de Test se utiliza únicamente para visualizar los errores
exigidos en la pauta; NO cambia pesos, umbrales, selección ni métricas Test.
Uso en Colab:
    %run scripts/generar_evidencia_errores.py
"""
from __future__ import annotations
import json
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image
from ultralytics import YOLO

ROOT = Path("/content/YOLO-Maquinaria-UBO")
DATASET = Path("/content/yolo_processed")
MODEL = ROOT / "models/modelo_elegido.pt"
IMG_DIR = DATASET / "test/images"
LBL_DIR = DATASET / "test/labels"
OUT = ROOT / "results/error_analysis"
CLASSES = ["EXCAVATORS", "dump truck", "wheel loader"]
CONFIANZA = 0.25
IOU_MIN = 0.50

def iou(a, b):
    x1,y1 = max(a[0],b[0]),max(a[1],b[1])
    x2,y2 = min(a[2],b[2]),min(a[3],b[3])
    inter=max(0.,x2-x1)*max(0.,y2-y1)
    area=lambda z:max(0.,z[2]-z[0])*max(0.,z[3]-z[1])
    den=area(a)+area(b)-inter
    return inter/den if den else 0.

def ground_truth(p, width, height):
    label=LBL_DIR/p.relative_to(IMG_DIR).with_suffix(".txt")
    boxes=[]
    for line in label.read_text().splitlines():
        c,x,y,w,h=map(float,line.split())
        boxes.append((int(c),[(x-w/2)*width,(y-h/2)*height,
                             (x+w/2)*width,(y+h/2)*height]))
    return boxes

def match(gt, preds):
    used=set(); fp=[]
    for index,(cls,conf,box) in sorted(enumerate(preds),key=lambda z:-z[1][1]):
        candidates=[(iou(box, gbox),j) for j,(gcls,gbox) in enumerate(gt)
                    if j not in used and gcls==cls]
        best,j=max(candidates,default=(0.,-1))
        if best>=IOU_MIN:
            used.add(j)
        else:
            fp.append(index)
    missing=[j for j in range(len(gt)) if j not in used]
    return missing,fp

def plot_case(p, gt, preds, target, path):
    fig, ax=plt.subplots(1,2,figsize=(11,5.5))
    im=Image.open(p).convert("RGB")
    for a in ax:
        a.imshow(im)
        a.axis("off")
    ax[0].set_title("Etiquetas reales (ground truth)")
    ax[1].set_title("Predicciones YOLO, conf. >= 0.25")
    for cls,box in gt:
        x1,y1,x2,y2=box
        ax[0].add_patch(Rectangle((x1,y1),x2-x1,y2-y1,fill=False,edgecolor="lime",linewidth=2))
        ax[0].text(x1,y1,CLASSES[cls],fontsize=8,color="black",backgroundcolor="lime")
    for cls,conf,box in preds:
        x1,y1,x2,y2=box
        ax[1].add_patch(Rectangle((x1,y1),x2-x1,y2-y1,fill=False,edgecolor="cyan",linewidth=2))
        ax[1].text(x1,y1,f"{CLASSES[cls]} {conf:.2f}",fontsize=8,color="black",backgroundcolor="cyan")
    fig.suptitle(f"{target}: {p.name}",fontsize=10)
    fig.tight_layout()
    fig.savefig(path,dpi=155,bbox_inches="tight")
    plt.close(fig)

def run():
    if not MODEL.exists():
        raise FileNotFoundError("Falta models/modelo_elegido.pt en la sesión Colab")
    if not IMG_DIR.is_dir():
        raise FileNotFoundError("Falta el dataset preparado /content/yolo_processed")
    OUT.mkdir(parents=True,exist_ok=True)
    model=YOLO(str(MODEL))
    found={}
    files=sorted(p for p in IMG_DIR.rglob("*") if p.suffix.lower() in {".jpg",".jpeg",".png"})
    for p in files:
        with Image.open(p) as image:
            w,h=image.size
        gt=ground_truth(p,w,h)
        pred=model.predict(str(p),imgsz=416,conf=CONFIANZA,verbose=False)[0]
        boxes=[(int(c),float(conf),box.tolist()) for c,conf,box
               in zip(pred.boxes.cls.cpu(),pred.boxes.conf.cpu(),pred.boxes.xyxy.cpu())]
        fn,fp=match(gt,boxes)
        candidate={}
        if fn:
            candidate["objeto_perdido"]={
                "detalle":f"Hay {len(fn)} objeto(s) reales sin predicción de la misma clase con IoU >=0.50.",
                "clases": [CLASSES[gt[i][0]] for i in fn],
            }
        if fp:
            candidate["falsa_deteccion"]={
                "detalle":f"Hay {len(fp)} deteccion(es) sin caja real emparejada (conf >=0.25).",
                "clases": [CLASSES[boxes[i][0]] for i in fp],
            }
        confused=[]
        for j in fp:
            cls,conf,box=boxes[j]
            for gt_cls,gt_box in gt:
                if cls!=gt_cls and iou(box,gt_box)>=0.3:
                    confused.append((CLASSES[cls],CLASSES[gt_cls],round(float(conf),2)))
        if confused:
            candidate["confusion_de_clases"]={
                "detalle":"Predicción de clase diferente que se superpone a una caja real.",
                "pares_pred_real":confused[:3],
            }
        for typ,info in candidate.items():
            if typ not in found and not any(e["archivo"]==p.name for e in found.values()):
                dest=OUT/f"{typ}.png"
                plot_case(p,gt,boxes,typ,dest)
                found[typ]={"archivo":p.name,"imagen":str(dest),"explicacion":info}
        if all(t in found for t in ("objeto_perdido","falsa_deteccion","confusion_de_clases")):
            break
    (OUT/"casos.json").write_text(json.dumps(found,indent=2,ensure_ascii=False),encoding="utf-8")
    print(json.dumps(found,indent=2,ensure_ascii=False))
    if "objeto_perdido" not in found:
        print("No se halló objeto perdido bajo estos umbrales; revisar manualmente y no inventar.")
    print("Las métricas de Test guardadas no se vuelven a calcular.")
    return found

if __name__=="__main__":
    run()
