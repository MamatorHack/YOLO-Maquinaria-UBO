"""Pruebas locales: no requieren GPU, descargar modelos ni datos reales."""
from pathlib import Path

import pytest
import yaml

from src.yolo_ubo import box_iou, match_boxes, prepare_dataset, count_dataset, check_exact_leakage, select_model, test_once as run_test_once


def test_iou():
    assert box_iou((0, 0, 10, 10), (0, 0, 10, 10)) == 1
    assert box_iou((0, 0, 1, 1), (2, 2, 3, 3)) == 0


def test_matches_no_reuse():
    gt = [(0, [0, 0, 10, 10])]
    pred = [(0, .9, [0, 0, 10, 10]), (0, .8, [0, 0, 10, 10])]
    assert match_boxes(gt, pred) == {"TP": 1, "FP": 1, "FN": 0}
    assert match_boxes(gt, [(1, .9, [0, 0, 10, 10])]) == {"TP": 0, "FP": 1, "FN": 1}


def synthetic_dataset(tmp_path):
    from PIL import Image
    base = tmp_path / "source"
    for split in ("train", "valid", "test"):
        (base / split / "images").mkdir(parents=True)
        (base / split / "labels").mkdir(parents=True)
        Image.new("RGB", (16, 16), color={"train":"red","valid":"green","test":"blue"}[split]).save(base / split / "images" / "image.png")
        (base / split / "labels" / "image.txt").write_text("2 0.5 0.5 0.4 0.4\n")
    y = {"train": "train/images", "val":"valid/images","test":"test/images",
         "names": ["other","dump truck","EXCAVATORS","wheel loader"]}
    (base/"data.yaml").write_text(yaml.safe_dump(y))
    return base/"data.yaml"


def test_prepare_counts_and_leakage(tmp_path):
    dataset = prepare_dataset(synthetic_dataset(tmp_path), tmp_path/"working",
                              ("EXCAVATORS","dump truck","wheel loader"))
    totals = count_dataset(dataset)
    assert all(totals[s]["images"] == 1 for s in ("train","val","test"))
    assert all(totals[s]["boxes"] == 1 for s in ("train","val","test"))
    assert (tmp_path/"working/train/labels/image.txt").read_text().startswith("0 ")
    assert not check_exact_leakage(dataset)["exact_duplicates_across_splits"]


def test_selection_locked(tmp_path):
    path = tmp_path/"results/selection.json"
    a = {"mAP50_95": .3, "mAP50": .7}
    b = {"mAP50_95": .4, "mAP50": .6}
    assert select_model(a,b,path)["selected"] == "fine_tuning"
    with pytest.raises(FileExistsError):
        select_model(b,a,path)


def test_test_requires_selection(tmp_path):
    with pytest.raises(RuntimeError):
        run_test_once("nonexistent.pt","nonexistent.yaml",output=tmp_path/"results/test_metrics.json")
