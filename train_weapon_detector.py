"""Train a custom YOLO weapon detector.

The repository cannot invent firearm training data. Supply a YOLO-format
dataset YAML containing weapon classes (for example gun, pistol, rifle):

    python train_weapon_detector.py --data data/weapon_dataset/data.yaml

The trained weights are written to models/weapon_yolov8.pt by default. Set
WEAPON_MODEL_PATH to that file before starting the dashboard.
"""

import argparse
import shutil
from pathlib import Path

try:
    from ultralytics import YOLO
except ImportError as e:
    raise ImportError(
        "Failed to import ultralytics. Please install it with: pip install ultralytics"
    ) from e

BASE_DIR = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a custom YOLO weapon detector")
    parser.add_argument("--data", required=True, help="YOLO dataset YAML path")
    parser.add_argument("--base-model", default="yolov8n.pt")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--output", default=str(BASE_DIR / "models" / "weapon_yolov8.pt"))
    args = parser.parse_args()

    data_path = Path(args.data).resolve()
    if not data_path.is_file():
        raise FileNotFoundError(f"Dataset YAML not found: {data_path}")
    if args.epochs < 1 or args.imgsz < 32 or args.batch < 1:
        raise ValueError("epochs, imgsz, and batch must be positive")

    project_dir = BASE_DIR / "runs" / "weapon"
    project_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(args.base_model)
    results = model.train(
        data=str(data_path),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        project=str(project_dir),
        name="train",
        exist_ok=True,
    )
    best = Path(results.save_dir) / "weights" / "best.pt"
    if not best.is_file():
        raise FileNotFoundError(f"Training completed without best.pt: {best}")

    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best, output)
    print(f"Saved weapon detector to {output}")


if __name__ == "__main__":
    main()
