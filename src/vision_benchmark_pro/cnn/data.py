from __future__ import annotations
import os, shutil, random
from pathlib import Path
import tensorflow as tf

def split_folders(src: Path, out: Path, train=0.7, val=0.15, test=0.15, ext=('.png','.jpg','.jpeg')):
    src, out = Path(src), Path(out)
    for split in ("train","val","test"):
        for cls in sorted([d.name for d in src.iterdir() if d.is_dir()]):
            (out/split/cls).mkdir(parents=True, exist_ok=True)

    for cls in sorted([d.name for d in src.iterdir() if d.is_dir()]):
        files = [p.name for p in (src/cls).iterdir() if p.suffix.lower() in ext]
        random.shuffle(files)
        n = len(files); ntr = int(train*n); nv = int(val*n)
        tr = files[:ntr]; va = files[ntr:ntr+nv]; te = files[ntr+nv:]
        for fname in tr: shutil.copy2(src/cls/fname, out/"train"/cls/fname)
        for fname in va: shutil.copy2(src/cls/fname, out/"val"/cls/fname)
        for fname in te: shutil.copy2(src/cls/fname, out/"test"/cls/fname)

def flow_from_dir(dir_path: Path, img_size=128, batch=32, shuffle=True):
    datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1.0/255)
    return datagen.flow_from_directory(
        str(dir_path), target_size=(img_size,img_size),
        batch_size=batch, class_mode='categorical', shuffle=shuffle
    )
