"""Prueba de preparación de datos sin depender del dataset real."""
import io
import json
import zipfile

import numpy as np
from PIL import Image

from scripts.preparar_dataset_sin_fugas import prepare


def image_bytes(seed):
    array = np.random.default_rng(seed).integers(0, 255, size=(64,64,3), dtype=np.uint8)
    image = Image.fromarray(array, mode="RGB")
    output = io.BytesIO()
    image.save(output, format="JPEG", quality=92)
    return output.getvalue()


def test_clean_dataset_groups_video_and_drops_duplicate(tmp_path):
    original = tmp_path / "source.zip"
    corrected = tmp_path / "clean.zip"
    base1 = image_bytes(1)
    base2 = image_bytes(2)
    base3 = image_bytes(3)
    with zipfile.ZipFile(original, "w") as z:
        z.writestr("data.yaml", "train: ../train/images\nval: ../valid/images\ntest: ../test/images\nnames: ['EXCAVATORS', 'dump truck', 'wheel loader']\n")
        for split, name, raw in [
            ("train","construction--1-_mp4-1.jpg",base1),
            ("valid","construction--1-_mp4-2.jpg",base2),
            ("test","construction--1-_mp4-3.jpg",base3),
            ("train","construction--100-_jpg.jpg",base1),
            ("valid","construction--dup-_jpg.jpg",base1),
            ("valid","construction--unico-_jpg.jpg",base2),
            ("test","construction--otro-_jpg.jpg",base3),
        ]:
            z.writestr(f"{split}/images/{name}", raw)
            z.writestr(f"{split}/labels/{name[:-4]}.txt", "0 0.5 0.5 0.4 0.4\n")
    meta = prepare(original, corrected)
    assert meta["moved_video_frames"] == 2
    assert len(meta["removed"]) == 1
    with zipfile.ZipFile(corrected) as z:
        assert "LIMPIEZA_DATOS.json" in z.namelist()
        assert "train/images/construction--1-_mp4-2.jpg" in z.namelist()
        assert "train/images/construction--1-_mp4-3.jpg" in z.namelist()
        assert "valid/images/construction--dup-_jpg.jpg" not in z.namelist()
        assert z.testzip() is None
        counts = json.loads(z.read("LIMPIEZA_DATOS.json"))["counts"]
        assert counts["train"]["images"] == 4
        assert counts["valid"]["images"] == 1
        assert counts["test"]["images"] == 1
