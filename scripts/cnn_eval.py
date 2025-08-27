import argparse, os, time, numpy as np
from pathlib import Path
from vision_benchmark_pro.cnn.model import create_model
from vision_benchmark_pro.cnn.data import flow_from_dir
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoints", required=True)
    p.add_argument("--test", required=True)
    p.add_argument("--img", type=int, default=128)
    p.add_argument("--classes", type=int, default=9)
    p.add_argument("--report", default="runs/cnn/eval_report.txt")
    args = p.parse_args()

    test_gen = flow_from_dir(Path(args.test), img_size=args.img, batch=32, shuffle=False)
    class_labels = list(test_gen.class_indices.keys())

    ckpt_dir = Path(args.checkpoints)
    ckpts = sorted([p for p in ckpt_dir.iterdir() if p.suffix == ".ckpt" or p.suffixes == [".ckpt",".index"] or p.name.endswith(".ckpt")])
    if not ckpts:
        # allow tensorflow's separate .index naming
        ckpts = sorted([p for p in ckpt_dir.iterdir() if p.name.endswith(".ckpt.index")])
    selected = []
    if ckpts:
        first = ckpts[0]
        last = ckpts[-1]
        selected = [first, last]

    model = create_model(input_shape=(args.img,args.img,3), num_classes=args.classes, dropout=0.3)

    lines = []
    for ck in selected:
        ck_path = str(ck).replace(".index","")
        model.load_weights(ck_path)
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        t0 = time.time()
        loss, acc = model.evaluate(test_gen, verbose=0)
        dt = time.time() - t0

        preds = model.predict(test_gen, verbose=0)
        y_pred = np.argmax(preds, axis=1)
        y_true = test_gen.classes

        acc_m = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        rec = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        f1  = f1_score(y_true, y_pred, average='weighted', zero_division=0)

        lines.append(f"Checkpoint: {ck_path}\nTest Acc: {acc_m:.4f} Time: {dt:.2f}s\n")
        lines.append(classification_report(y_true, y_pred, target_names=class_labels))
        lines.append("\n")

    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote report to {args.report}")

if __name__ == "__main__":
    main()
