from __future__ import annotations
import os, re, shutil, random
from pathlib import Path
from typing import Dict, Tuple

# Default class map (extend/trim as needed)
DEFAULT_CLASS_MAP = {
    "PAScarRear": 0,
    "PASmotorbikeSide": 1,
    "PAScarSide": 2,
    "PASbicycle": 3,
    "PAScar": 4,
    "PASperson": 5,
    "PASpersonWalking": 6,
    "PASpersonStanding": 7,
    "PASpersonSitting": 8,
    "PAScarFrontal": 9,
    "PASstreetSign": 10,
    "PASstreet": 11,
    "PASbuildingPart": 12,
    "PASbuildingWhole": 13,
    "PASbuildingRegion": 14,
    "PASstreetPart": 15,
    "PAScarPart": 16,
    "PAStreePart": 17,
    "PAStreeWhole": 18,
    "PASwindow": 19,
    "PASstreetlight": 20,
    "PASsky": 21,
    "PAStreeRegion": 22,
    "PASbuilding": 23,
    "PASdoor": 24,
    "PAStrafficlight": 25,
    "PAStree": 26,
    "PASposter": 27,
    "PASposterClutter": 28,
    "PASskyRegion": 29,
    "PASbicycleSide": 30,
}

def convert_one_pascal(txt: str, img_w: int, img_h: int, class_map: Dict[str,int]) -> str:
    objects = re.findall(r'Details for object \d+ \("([^"]+)"\).*?Bounding box.*?: \((\d+), (\d+)\) - \((\d+), (\d+)\)',
                         txt, re.DOTALL)
    lines = []
    for cls, xmin, ymin, xmax, ymax in objects:
        if cls not in class_map:  # skip unknown labels
            continue
        xmin, ymin, xmax, ymax = map(int, (xmin, ymin, xmax, ymax))
        cx = (xmin + xmax) / 2 / img_w
        cy = (ymin + ymax) / 2 / img_h
        w  = (xmax - xmin) / img_w
        h  = (ymax - ymin) / img_h
        cx = max(0.0, min(1.0, cx)); cy = max(0.0, min(1.0, cy))
        w  = max(0.0, min(1.0, w));  h  = max(0.0, min(1.0, h))
        lines.append(f"{class_map[cls]} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
    return "\n".join(lines)

def build_yolo_dataset(images_dir: Path, annos_dir: Path, out_dir: Path,
                       img_w: int, img_h: int, class_map: Dict[str,int] | None = None,
                       splits: Tuple[float,float,float] = (0.8, 0.1, 0.1)) -> None:
    class_map = class_map or DEFAULT_CLASS_MAP
    out_dir = Path(out_dir)
    (out_dir/"train/images").mkdir(parents=True, exist_ok=True)
    (out_dir/"val/images").mkdir(parents=True, exist_ok=True)
    (out_dir/"test/images").mkdir(parents=True, exist_ok=True)
    (out_dir/"train/labels").mkdir(parents=True, exist_ok=True)
    (out_dir/"val/labels").mkdir(parents=True, exist_ok=True)
    (out_dir/"test/labels").mkdir(parents=True, exist_ok=True)

    # collect pairs
    pairs = []
    for class_folder in sorted(p for p in Path(images_dir).iterdir() if p.is_dir()):
        ann_folder = Path(annos_dir)/class_folder.name
        for img in sorted(class_folder.glob("*.*")):
            stem = img.stem
            ann = ann_folder/(stem + ".txt")
            if ann.exists():
                pairs.append((img, ann))

    random.shuffle(pairs)
    n = len(pairs)
    ntr = int(splits[0] * n); nv = int(splits[1] * n)
    tr, va, te = pairs[:ntr], pairs[ntr:ntr+nv], pairs[ntr+nv:]

    def handle_split(split_pairs, split_name):
        for img, ann in split_pairs:
            ytxt = convert_one_pascal(Path(ann).read_text(encoding="utf-8", errors="ignore"), img_w, img_h, class_map)
            if not ytxt:
                continue
            shutil.copy2(img, out_dir/f"{split_name}/images"/img.name)
            (out_dir/f"{split_name}/labels"/(img.stem + ".txt")).write_text(ytxt, encoding="utf-8")

    handle_split(tr, "train"); handle_split(va, "val"); handle_split(te, "test")
