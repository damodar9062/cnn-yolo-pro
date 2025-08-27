import argparse
from pathlib import Path
from vision_benchmark_pro.cnn.data import split_folders

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--src", required=True, help="Root folder with class subfolders")
    p.add_argument("--out", required=True, help="Output split folder")
    p.add_argument("--train", type=float, default=0.7)
    p.add_argument("--val", type=float, default=0.15)
    p.add_argument("--test", type=float, default=0.15)
    args = p.parse_args()

    split_folders(Path(args.src), Path(args.out), args.train, args.val, args.test)
    print(f"Split created under {args.out}")

if __name__ == "__main__":
    main()
