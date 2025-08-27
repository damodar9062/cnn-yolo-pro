import argparse, os
from pathlib import Path
from vision_benchmark_pro.cnn.model import create_model
from vision_benchmark_pro.cnn.data import flow_from_dir
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train", required=True)
    p.add_argument("--val", required=True)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--img", type=int, default=128)
    p.add_argument("--classes", type=int, default=9)
    p.add_argument("--ckpt", default="runs/cnn/checkpoints")
    args = p.parse_args()

    train_gen = flow_from_dir(Path(args.train), img_size=args.img, batch=args.batch, shuffle=True)
    val_gen = flow_from_dir(Path(args.val), img_size=args.img, batch=args.batch, shuffle=False)
    model = create_model(input_shape=(args.img,args.img,3), num_classes=args.classes, dropout=0.2)

    ckpt_dir = Path(args.ckpt); ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir/"cp-{epoch:04d}.ckpt"
    callbacks=[
        ModelCheckpoint(filepath=str(ckpt_path), save_weights_only=True, verbose=1, save_freq='epoch'),
        CSVLogger(str(Path(args.ckpt).parent/"training_log.csv"), append=True)
    ]

    model.fit(train_gen, epochs=args.epochs, validation_data=val_gen, callbacks=callbacks)

if __name__ == "__main__":
    main()
