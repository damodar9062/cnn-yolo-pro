# Vision Benchmark Pro — CNN (TF/Keras) vs YOLO (Object Detection)

A professionalized version of your **EAS595 HW02** project with clean packaging, parameterized scripts, CI, and docs.
It includes:
- **Dataset conversion** from VOC-style text to **YOLO** format
- **CNN training/evaluation** (TF/Keras) for 9-way classification
- **YOLO training/evaluation** (YOLOv5-style, via Python API if available)
- **Benchmark** script to compare metrics and timings

> This repo is a production-style reframe of your homework — not coursework-looking.

## Quickstart
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -U pip && pip install -e .[dev]
pre-commit install
```

### 1) Convert dataset to YOLO format
```bash
python scripts/convert_voc_to_yolo.py   --images data/PNGImages   --annos  data/Annotations   --out    data/yolo    --imgw 640 --imgh 480
```
This creates `data/yolo/train|val|test/(images|labels)`.

### 2) Train CNN (classification)
```bash
# Split raw class folders into split/train|val|test, then train
python scripts/cnn_split.py --src data/PNGImages --out data/split --train 0.7 --val 0.15 --test 0.15

python scripts/cnn_train.py   --train data/split/train   --val   data/split/val     --epochs 200 --batch 32 --img 128 --ckpt runs/cnn/checkpoints
```

### 3) Evaluate CNN checkpoints
```bash
python scripts/cnn_eval.py   --checkpoints runs/cnn/checkpoints   --test data/split/test   --img 128   --report runs/cnn/eval_report.txt
```

### 4) Train YOLO (detection) — if YOLOv5 Python API is installed
```bash
# Ensure your YAML points to data/yolo folders
python scripts/yolo_train.py --data configs/train.yaml --epochs 200 --img 128 --batch 8 --name custom_yolo_model
```

### 5) Evaluate YOLO
```bash
python scripts/yolo_eval.py   --weights runs/yolo/custom_yolo_model/weights/best.pt   --images  data/yolo/test/images   --labels  data/yolo/test/labels   --img 128   --out runs/yolo/predictions
```

### 6) Benchmark (CNN vs YOLO)
```bash
python scripts/benchmark.py   --cnn_report runs/cnn/eval_report.txt   --yolo_report runs/yolo/metrics.json
```

## Dependencies
Base: `numpy, pandas, scikit-learn, opencv-python, matplotlib, seaborn`  
CNN: **TensorFlow** (`pip install tensorflow` or `tensorflow-macos` depending on your OS)  
YOLO: Python API for YOLOv5 (`pip install yolov5`) or clone Ultralytics repo and add to PYTHONPATH.

## License
MIT © 2025 Dhamodar Burla
