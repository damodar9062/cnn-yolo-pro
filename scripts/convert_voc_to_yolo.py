import argparse
from pathlib import Path
from vision_benchmark_pro.data.voc_to_yolo import build_yolo_dataset

def main():
    p = argparse.ArgumentParser(description="Convert VOC-like text annos to YOLO format and split dataset")
    p.add_argument("--images", required=True, help="Root images folder with class subfolders")
    p.add_argument("--annos", required=True, help="Root annotations (txt) with class subfolders")
    p.add_argument("--out", required=True, help="Output root for YOLO dataset")
    p.add_argument("--imgw", type=int, default=640)
    p.add_argument("--imgh", type=int, default=480)
    p.add_argument("--train", type=float, default=0.8)
    p.add_argument("--val", type=float, default=0.1)
    p.add_argument("--test", type=float, default=0.1)
    args = p.parse_args()

    build_yolo_dataset(Path(args.images), Path(args.annos), Path(args.out), args.imgw, args.imgh, splits=(args.train,args.val,args.test))
    print(f"YOLO dataset created at: {args.out}")

if __name__ == "__main__":
    main()
