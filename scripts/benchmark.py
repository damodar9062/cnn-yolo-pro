import argparse, json, pandas as pd

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cnn_report", required=True, help="Path to cnn eval report text")
    p.add_argument("--yolo_report", required=True, help="Path to YOLO metrics.json")
    args = p.parse_args()

    # Parse CNN accuracy from report last 'Test Acc' line
    cnn_acc = None
    with open(args.cnn_report, 'r', encoding='utf-8') as f:
        for line in f:
            if 'Test Acc:' in line:
                try:
                    cnn_acc = float(line.split('Test Acc:')[1].split()[0])
                except Exception:
                    pass
    yolo = json.loads(open(args.yolo_report, 'r', encoding='utf-8').read())
    df = pd.DataFrame([
        {"model": "CNN", "accuracy": cnn_acc, "precision": None, "recall": None, "f1": None},
        {"model": "YOLO", **{k: yolo.get(k) for k in ("accuracy","precision","recall","f1_score")}}
    ])
    print(df)

if __name__ == "__main__":
    main()
