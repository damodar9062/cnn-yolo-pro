import argparse, os, json
from pathlib import Path
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
from vision_benchmark_pro.yolo.api import eval_with_yolov5

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--weights", required=True)
    p.add_argument("--images", required=True)
    p.add_argument("--labels", required=True)
    p.add_argument("--img", type=int, default=128)
    p.add_argument("--out", default="runs/yolo/predictions")
    args = p.parse_args()

    secs = eval_with_yolov5(args.weights, args.images, imgsz=args.img, project=args.out, name="pred")
    print(f"Evaluation completed in {secs:.2f} seconds.")

    # Aggregate per-label files to y_pred/y_true (best-effort simple mapping)
    labels_dir = Path(args.labels)
    preds_dir = Path(args.out)/"pred"/"labels"
    y_true, y_pred = [], []
    if preds_dir.exists():
        pred_files = sorted([p for p in preds_dir.iterdir() if p.suffix=='.txt'])
        gt_files = sorted([p for p in labels_dir.iterdir() if p.suffix=='.txt'])
        n = min(len(pred_files), len(gt_files))
        for i in range(n):
            gt = [int(l.strip().split()[0]) for l in gt_files[i].read_text().splitlines() if l.strip()]
            pr = [int(l.strip().split()[0]) for l in pred_files[i].read_text().splitlines() if l.strip()]
            # naive pairing: compare counts; truncate to shortest
            m = min(len(gt), len(pr))
            y_true.extend(gt[:m]); y_pred.extend(pr[:m])

    if y_true and y_pred:
        metrics = {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "precision": float(precision_score(y_true, y_pred, average='weighted', zero_division=0)),
            "recall": float(recall_score(y_true, y_pred, average='weighted', zero_division=0)),
            "f1_score": float(f1_score(y_true, y_pred, average='weighted', zero_division=0)),
            "count": len(y_true),
            "seconds": secs
        }
        Path(args.out).mkdir(parents=True, exist_ok=True)
        (Path(args.out)/"metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        print(json.dumps(metrics, indent=2))
    else:
        print("Warning: Could not align predictions with labels to compute metrics.")

if __name__ == "__main__":
    main()
