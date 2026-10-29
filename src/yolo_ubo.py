"""Herramientas reproducibles para Evaluación 03 — YOLO maquinaria."""
from __future__ import annotations

import hashlib
import json
import shutil
from collections import Counter
from pathlib import Path
from time import perf_counter

import numpy as np
import yaml

CLASSES = ("EXCAVATORS", "dump truck", "wheel loader")
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SPLITS = {"train": "train", "valid": "val", "val": "val", "test": "test"}


def find_dataset(root: str | Path) -> Path:
    """Encuentra un solo data.yaml dentro del ZIP extraído."""
    root = Path(root)
    found = list(root.rglob("data.yaml"))
    if len(found) != 1:
        raise ValueError(f"Se esperaba un data.yaml; encontrados: {len(found)}")
    return found[0]


def prepare_dataset(yaml_path: str | Path, output_dir: str | Path, selected: tuple[str, ...] = CLASSES) -> Path:
    """Filtra clases y copia cada split SIN remezclar ni mover sus imágenes."""
    yaml_path, output_dir = Path(yaml_path).resolve(), Path(output_dir).resolve()
    data = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    names = data["names"]
    names = {int(k): v for k, v in names.items()} if isinstance(names, dict) else dict(enumerate(names))
    missing = set(selected) - set(names.values())
    if missing:
        raise ValueError(f"Clases ausentes del YAML: {missing}; disponibles: {names}")
    indexes = {old: selected.index(name) for old, name in names.items() if name in selected}
    base = Path(data.get("path") or yaml_path.parent)
    if not base.is_absolute():
        base = (yaml_path.parent / base).resolve()
    if not base.exists():
        base = yaml_path.parent

    for dst_split, variants in (("train", ("train",)), ("val", ("val", "valid")), ("test", ("test",))):
        key = next((k for k in variants if k in data), None)
        if key is None:
            raise ValueError(f"Split obligatorio no encontrado: {dst_split}")
        image_dir = Path(data[key])
        if not image_dir.is_absolute():
            image_dir = (base / image_dir).resolve()
        if not image_dir.exists():
            # Algunos exports Roboflow proporcionan 'train/images' con path heredado.
            image_dir = (yaml_path.parent / data[key]).resolve()
        if not image_dir.is_dir():
            raise FileNotFoundError(image_dir)
        dest_images = output_dir / dst_split / "images"
        dest_labels = output_dir / dst_split / "labels"
        dest_images.mkdir(parents=True, exist_ok=True)
        dest_labels.mkdir(parents=True, exist_ok=True)
        images = [p for p in sorted(image_dir.rglob("*")) if p.suffix.lower() in IMAGE_EXTS]
        if not images:
            raise ValueError(f"Split vacío: {dst_split}")
        label_root = image_dir.parent / "labels"
        for img in images:
            relative = img.relative_to(image_dir)
            dest_img = dest_images / relative
            dest_img.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(img, dest_img)
            original_label = label_root / relative.with_suffix(".txt")
            target_label = dest_labels / relative.with_suffix(".txt")
            target_label.parent.mkdir(parents=True, exist_ok=True)
            converted = []
            if original_label.exists():
                for ln in original_label.read_text(encoding="utf-8").splitlines():
                    parts = ln.split()
                    if len(parts) != 5:
                        raise ValueError(f"Etiqueta inválida en {original_label}: {ln}")
                    old = int(parts[0])
                    coords = [float(v) for v in parts[1:]]
                    x, y, w, h = coords
                    if not (0 <= x <= 1 and 0 <= y <= 1 and 0 < w <= 1 and 0 < h <= 1):
                        raise ValueError(f"Bounding box fuera de rango en {original_label}: {ln}")
                    if old in indexes:
                        converted.append(" ".join([str(indexes[old])] + parts[1:]))
            target_label.write_text("\n".join(converted) + ("\n" if converted else ""), encoding="utf-8")
    config = {"path": str(output_dir), "train": "train/images", "val": "val/images",
              "test": "test/images", "nc": len(selected), "names": list(selected)}
    final_yaml = output_dir / "data.yaml"
    final_yaml.write_text(yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return final_yaml


def count_dataset(data_yaml: str | Path) -> dict:
    data = yaml.safe_load(Path(data_yaml).read_text(encoding="utf-8"))
    root = Path(data["path"])
    summary = {}
    for split in ("train", "val", "test"):
        images = [p for p in (root / data[split]).rglob("*") if p.suffix.lower() in IMAGE_EXTS]
        labels_dir = root / split / "labels"
        counts = Counter()
        boxes = 0
        for p in labels_dir.rglob("*.txt"):
            for line in p.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    counts[int(line.split()[0])] += 1
                    boxes += 1
        summary[split] = {"images": len(images), "boxes": boxes,
                          "per_class_boxes": {data["names"][i]: counts[i] for i in range(len(data["names"]))}}
    return summary


def check_exact_leakage(data_yaml: str | Path) -> dict:
    """SHA256 de imágenes de distintos splits. NO detecta automáticamente augmentations/escenas."""
    data = yaml.safe_load(Path(data_yaml).read_text(encoding="utf-8"))
    root = Path(data["path"])
    seen = {}
    overlaps = []
    for split in ("train", "val", "test"):
        for p in (root / data[split]).rglob("*"):
            if p.suffix.lower() not in IMAGE_EXTS:
                continue
            digest = hashlib.sha256(p.read_bytes()).hexdigest()
            if digest in seen and seen[digest][0] != split:
                overlaps.append({"first": str(seen[digest][1]), "second": str(p)})
            seen.setdefault(digest, (split, p))
    return {"exact_duplicates_across_splits": overlaps,
            "warning": "Revisar manualmente imágenes visualmente similares y escenas de un mismo video."}


def plot_samples(data_yaml: str | Path, split="train", n=6, output="results/samples.png"):
    import matplotlib.pyplot as plt
    from PIL import Image
    data = yaml.safe_load(Path(data_yaml).read_text(encoding="utf-8"))
    root = Path(data["path"])
    images = sorted(p for p in (root / data[split]).rglob("*") if p.suffix.lower() in IMAGE_EXTS)
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    for ax, p in zip(axes.flat, images[:n]):
        im = np.asarray(Image.open(p).convert("RGB"))
        height, width = im.shape[:2]
        ax.imshow(im)
        for ln in (root / split / "labels" / p.relative_to(root / data[split]).with_suffix(".txt")).read_text().splitlines():
            cls, x, y, w, h = map(float, ln.split())
            from matplotlib.patches import Rectangle
            ax.add_patch(Rectangle(((x-w/2)*width, (y-h/2)*height), w*width, h*height,
                                   fill=False, color="lime", linewidth=2))
            ax.text((x-w/2)*width, (y-h/2)*height, data["names"][int(cls)],
                    color="black", backgroundcolor="lime", fontsize=8)
        ax.set_title(p.name[:35])
        ax.axis("off")
    for ax in list(axes.flat)[min(n, len(images)):]:
        ax.axis("off")
    fig.tight_layout()
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=140)
    plt.close(fig)
    return output


def train_stage(model_weights, data_yaml, name, epochs, imgsz, batch, freeze, lr0,
                optimizer="auto", fraction=1.0, project="runs_ubo", seed=42):
    from ultralytics import YOLO
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("El proyecto requiere entrenamiento en Colab con GPU. Activar T4.")
    gpu = torch.cuda.get_device_name(0)
    model = YOLO(str(model_weights))
    start = perf_counter()
    model.train(data=str(data_yaml), epochs=epochs, imgsz=imgsz, batch=batch,
                freeze=freeze, optimizer=optimizer, lr0=lr0, fraction=fraction,
                project=project, name=name, seed=seed, device=0,
                exist_ok=False, plots=True)
    seconds = perf_counter() - start
    best = Path(model.trainer.best)
    if not best.exists():
        raise FileNotFoundError(f"Pesos entrenados no encontrados: {best}")
    return {"best": str(best), "gpu": gpu, "seconds": round(seconds, 2),
            "epochs": epochs, "imgsz": imgsz, "batch": batch, "freeze": freeze,
            "optimizer": optimizer, "lr0": lr0, "fraction": fraction,
            "results_csv": str(Path(model.trainer.save_dir) / "results.csv")}


def evaluate(weights, data_yaml, split="val", imgsz=416, conf=0.25):
    from ultralytics import YOLO
    metrics = YOLO(str(weights)).val(data=str(data_yaml), split=split,
                                   imgsz=imgsz, conf=conf, device=0, plots=True)
    return {"precision": float(metrics.box.mp), "recall": float(metrics.box.mr),
            "mAP50": float(metrics.box.map50), "mAP50_95": float(metrics.box.map)}


def select_model(transfer_metrics, fine_metrics, output="results/selection.json"):
    """Selección determinista, exclusivamente por VALIDATION."""
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"Ya existe elección. No sobrescribir: {output}")
    score_t = (transfer_metrics["mAP50_95"], transfer_metrics["mAP50"])
    score_f = (fine_metrics["mAP50_95"], fine_metrics["mAP50"])
    selected = "fine_tuning" if score_f > score_t else "transfer_learning"
    output.parent.mkdir(parents=True, exist_ok=True)
    result = {"selected": selected, "rule": "max Validation mAP50-95; empate mAP50; empate transfer",
              "transfer_validation": transfer_metrics, "fine_validation": fine_metrics}
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def test_once(weights, data_yaml, imgsz=416, output="results/test_metrics.json"):
    """No calcula Test si ya existe el archivo marcador."""
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"TEST ya fue ejecutado: {output}")
    if not (output.parent / "selection.json").exists():
        raise RuntimeError("Primero seleccionar el modelo utilizando solo Validation.")
    output.parent.mkdir(parents=True, exist_ok=True)
    # Marcador previo: no repetir Test después de un fallo parcial sin documentarlo.
    output.write_text(json.dumps({"status": "started", "weights": str(weights), "imgsz": imgsz}))
    metrics = evaluate(weights, data_yaml, split="test", imgsz=imgsz)
    output.write_text(json.dumps({"status": "finished", **metrics}, indent=2))
    return metrics


def box_iou(a, b):
    """IoU para cajas xyxy."""
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2-x1)*max(0, y2-y1)
    aa = max(0, a[2]-a[0])*max(0, a[3]-a[1])
    bb = max(0, b[2]-b[0])*max(0, b[3]-b[1])
    return inter/(aa+bb-inter) if aa+bb-inter else 0.0


def match_boxes(gt_boxes, predictions, iou_threshold=0.5):
    """Matching voraz por confianza y misma clase, TP/FP/FN con IoU mínimo."""
    used = set()
    tp = fp = 0
    for cls, conf, pred_box in sorted(predictions, key=lambda p: -p[1]):
        candidates = [(box_iou(pred_box, b), j) for j, (c, b) in enumerate(gt_boxes)
                      if j not in used and c == cls]
        best_iou, best_idx = max(candidates, default=(0., -1))
        if best_iou >= iou_threshold:
            used.add(best_idx)
            tp += 1
        else:
            fp += 1
    return {"TP": tp, "FP": fp, "FN": len(gt_boxes)-len(used)}


def count_detections(weights, data_yaml, split="val", imgsz=416, conf=0.25, iou=0.5):
    """Recuento educativo de TP/FP/FN; NO sustituye las métricas Ultralytics."""
    from ultralytics import YOLO
    from PIL import Image
    data = yaml.safe_load(Path(data_yaml).read_text())
    root = Path(data["path"])
    images_dir = root / data[split]
    model = YOLO(str(weights))
    counts = Counter()
    examples = {"false_positive": [], "missed": []}
    for p in sorted(images_dir.rglob("*")):
        if p.suffix.lower() not in IMAGE_EXTS:
            continue
        width, height = Image.open(p).size
        label_path = root / split / "labels" / p.relative_to(images_dir).with_suffix(".txt")
        gt = []
        for ln in label_path.read_text().splitlines():
            c, x, y, w, h = map(float, ln.split())
            gt.append((int(c), [(x-w/2)*width, (y-h/2)*height,
                                (x+w/2)*width, (y+h/2)*height]))
        result = model.predict(str(p), imgsz=imgsz, conf=conf, verbose=False)[0]
        preds = [(int(c), float(q), xy.tolist()) for c, q, xy in
                 zip(result.boxes.cls.cpu(), result.boxes.conf.cpu(), result.boxes.xyxy.cpu())]
        hit = match_boxes(gt, preds, iou)
        counts.update(hit)
        if hit["FP"] and len(examples["false_positive"]) < 5:
            examples["false_positive"].append(str(p))
        if hit["FN"] and len(examples["missed"]) < 5:
            examples["missed"].append(str(p))
    return {"counts": dict(counts), "confidence_threshold": conf, "iou_threshold": iou,
            "examples": examples}


def plot_training_curves(csv_path, output):
    """Gráficas por época: loss y mAP50-95."""
    import pandas as pd
    import matplotlib.pyplot as plt
    df = pd.read_csv(csv_path)
    df.columns = [c.strip() for c in df.columns]
    cols = [c for c in df if ("loss" in c.lower() or "map50-95" in c.lower())]
    if not cols:
        raise ValueError("No hay columnas de loss/mAP en results.csv")
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    for col in cols:
        ax = axes[0] if "loss" in col.lower() else axes[1]
        ax.plot(df["epoch"], df[col], label=col)
    for ax in axes:
        ax.set_xlabel("Época")
        ax.legend(fontsize=7)
        ax.grid(alpha=0.2)
    axes[0].set_title("Pérdidas"); axes[1].set_title("mAP50-95 (Validation)")
    fig.tight_layout()
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150)
    plt.close(fig)
    return str(output)
