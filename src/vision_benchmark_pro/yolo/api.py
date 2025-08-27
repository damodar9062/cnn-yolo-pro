from __future__ import annotations
import json, time, os
from pathlib import Path

def train_with_yolov5(data_yaml: str, imgsz=128, epochs=200, batch_size=8, name='custom_yolo_model'):
    try:
        from yolov5 import train
    except Exception as e:
        raise RuntimeError("yolov5 Python API not found. Install with 'pip install yolov5' or clone ultralytics/yolov5.") from e
    t0 = time.time()
    train.run(data=data_yaml, imgsz=imgsz, epochs=epochs, batch_size=batch_size, weights='yolov5s.pt', name=name)
    return time.time() - t0

def eval_with_yolov5(weights: str, source_images: str, imgsz=128, project='runs/yolo', name='pred', save_txt=True, save_conf=True):
    try:
        from yolov5 import detect
    except Exception as e:
        raise RuntimeError("yolov5 Python API not found. Install with 'pip install yolov5' or clone ultralytics/yolov5.") from e
    t0 = time.time()
    detect.run(weights=weights, source=source_images, imgsz=(imgsz, imgsz), save_txt=save_txt, save_conf=save_conf, project=project, name=name)
    return time.time() - t0
