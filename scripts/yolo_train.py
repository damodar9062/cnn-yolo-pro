import argparse
from vision_benchmark_pro.yolo.api import train_with_yolov5

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True, help="train.yaml path")
    p.add_argument("--img", type=int, default=128)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--name", default="custom_yolo_model")
    args = p.parse_args()

    secs = train_with_yolov5(args.data, imgsz=args.img, epochs=args.epochs, batch_size=args.batch, name=args.name)
    print(f"Training completed in {secs:.2f} seconds. Checkpoints under runs/yolo/{args.name}/")

if __name__ == "__main__":
    main()
