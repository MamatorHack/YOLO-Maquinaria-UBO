#!/usr/bin/env python3
"""Limpieza reproducible del dataset Excavators v2. Uso:
python scripts/preparar_dataset_sin_fugas.py original.zip corregido.zip
"""
import hashlib
import io
import json
import re
import sys
import zipfile
from collections import Counter
from pathlib import Path
import numpy as np
from PIL import Image

CLASSES = ['EXCAVATORS', 'dump truck', 'wheel loader']
IMAGES_RE = re.compile(r'^(train|valid|test)/images/([^/]+)\.(jpg|jpeg|png)$', re.I)


def fingerprint(raw):
    with Image.open(io.BytesIO(raw)) as image:
        im = image.convert('L')
        arr = np.asarray(im.resize((9, 8)), dtype=np.int16)
        dh = 0
        for bit in (arr[:, :-1] > arr[:, 1:]).reshape(-1):
            dh = (dh << 1) | int(bit)
        small = np.asarray(im.resize((64, 64)), dtype=np.float32).reshape(-1)
        small -= small.mean()
        norm = np.linalg.norm(small)
        if norm:
            small /= norm
        return dh, small


def near(a, b):
    return (a[0] ^ b[0]).bit_count() <= 6 and float(np.dot(a[1], b[1])) >= .985


def prepare(source, output):
    source, output = Path(source), Path(output)
    if source.resolve() == output.resolve():
        raise ValueError('No sobrescribir el ZIP original')
    with zipfile.ZipFile(source) as z:
        yaml_raw = z.read('data.yaml').decode()
        if not all(v in yaml_raw for v in CLASSES):
            raise ValueError('El YAML no contiene las tres clases previstas')
        members = set(z.namelist())
        entries = []
        for name in sorted(members):
            m = IMAGES_RE.fullmatch(name)
            if not m:
                continue
            split, file_stem, ext = m.groups()
            label = name.replace('/images/', '/labels/').rsplit('.', 1)[0] + '.txt'
            if label not in members:
                raise ValueError('Anotación ausente: ' + label)
            entries.append({'name': name, 'label': label, 'split': split,
                            'video': '_mp4-' in file_stem})
        train = [x for x in entries if x['split'] == 'train' or x['video']]
        still_refs = [(x['name'], fingerprint(z.read(x['name']))) for x in train if not x['video']]
        remaining = [x for x in entries if x['split'] != 'train' and not x['video']]
        kept = list(train)
        removed = []
        val_refs = []
        for split in ('valid', 'test'):
            for x in (item for item in remaining if item['split'] == split):
                fp = fingerprint(z.read(x['name']))
                duplicate = next((name for name, known in still_refs if near(fp, known)), None)
                if duplicate is None and split == 'test':
                    duplicate = next((name for name, known in val_refs if near(fp, known)), None)
                if duplicate:
                    removed.append({'excluded': x['name'], 'similar_to': duplicate})
                else:
                    kept.append(x)
                    if split == 'valid':
                        val_refs.append((x['name'], fp))
        assert len(entries) == len(kept) + len(removed)
        counts = {s: {'images': 0, 'boxes': Counter()} for s in ('train','valid','test')}
        seen = set()
        with zipfile.ZipFile(output, 'w', zipfile.ZIP_DEFLATED, compresslevel=5) as out:
            for x in kept:
                target_split = 'train' if x['video'] else x['split']
                img = target_split + '/images/' + Path(x['name']).name
                lbl = target_split + '/labels/' + Path(x['label']).name
                if img in seen or lbl in seen:
                    raise ValueError('Nombre duplicado: ' + img)
                seen.update((img, lbl))
                label_bytes = z.read(x['label'])
                out.writestr(img, z.read(x['name']))
                out.writestr(lbl, label_bytes)
                counts[target_split]['images'] += 1
                for line in label_bytes.decode().splitlines():
                    if not line.strip():
                        continue
                    vals = line.split()
                    if len(vals) != 5 or int(vals[0]) not in range(3):
                        raise ValueError('Anotación inválida: ' + x['label'])
                    coords = [float(v) for v in vals[1:]]
                    if not (all(0 <= v <= 1 for v in coords) and coords[2] > 0 and coords[3] > 0):
                        raise ValueError('Caja fuera de rango: ' + x['label'])
                    counts[target_split]['boxes'][int(vals[0])] += 1
            out.writestr('data.yaml', "train: train/images\nval: valid/images\ntest: test/images\nnc: 3\nnames: ['EXCAVATORS', 'dump truck', 'wheel loader']\n")
            for filename in ('README.dataset.txt', 'README.roboflow.txt'):
                if filename in members:
                    out.writestr(filename, z.read(filename))
            meta = {
                'original': source.name, 'sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
                'method': 'move all *_mp4-* video frames to Train; remove visually near-identical stills from Validation/Test',
                'criterion': 'dHash <=6/64, Pearson grayscale 64x64 >=0.985',
                'moved_video_frames': sum(x['split']!='train' and x['video'] for x in train),
                'removed': removed,
                'counts': {s: {'images': d['images'], 'boxes': sum(d['boxes'].values()),
                               'per_class': dict(d['boxes'])} for s,d in counts.items()},
                'limits': 'Near duplicate matching is not exhaustive; manual source inspection is still required',
            }
            out.writestr('LIMPIEZA_DATOS.json', json.dumps(meta, indent=2, ensure_ascii=False))
    print(json.dumps({k: v for k,v in meta.items() if k not in ('removed','sha256')}, indent=2))
    return meta


if __name__ == '__main__':
    if len(sys.argv) != 3:
        raise SystemExit('python preparar_dataset_sin_fugas.py original.zip corregido.zip')
    prepare(sys.argv[1], sys.argv[2])
