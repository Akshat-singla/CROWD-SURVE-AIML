"""Prepare the weapon dataset in YOLO format with train/val split."""

import os
import random
import shutil
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
SRC_IMAGES = BASE_DIR / "Dataset" / "images"
SRC_LABELS = BASE_DIR / "Dataset" / "labels"

DEST_BASE = BASE_DIR / "data" / "weapon_dataset"
DEST_IMG_TRAIN = DEST_BASE / "images" / "train"
DEST_IMG_VAL = DEST_BASE / "images" / "val"
DEST_LBL_TRAIN = DEST_BASE / "labels" / "train"
DEST_LBL_VAL = DEST_BASE / "labels" / "val"

TRAIN_RATIO = 0.8
random.seed(42)


def main() -> None:
    for d in (DEST_IMG_TRAIN, DEST_IMG_VAL, DEST_LBL_TRAIN, DEST_LBL_VAL):
        d.mkdir(parents=True, exist_ok=True)

    pairs = []
    for img_name in sorted(os.listdir(SRC_IMAGES)):
        if not img_name.lower().endswith((".png", ".jpg", ".jpeg")):
            continue
        stem = Path(img_name).stem
        lbl_name = stem + ".txt"
        src_img = SRC_IMAGES / img_name
        src_lbl = SRC_LABELS / lbl_name
        if not src_lbl.is_file():
            print(f"[SKIP] Missing label for {img_name}")
            continue
        pairs.append((src_img, src_lbl, img_name, lbl_name))

    random.shuffle(pairs)
    split_idx = int(len(pairs) * TRAIN_RATIO)
    train_pairs = pairs[:split_idx]
    val_pairs = pairs[split_idx:]

    print(f"Total pairs: {len(pairs)}")
    print(f"Train: {len(train_pairs)}, Val: {len(val_pairs)}")

    for src_img, src_lbl, img_name, lbl_name in train_pairs:
        shutil.copy2(src_img, DEST_IMG_TRAIN / img_name)
        shutil.copy2(src_lbl, DEST_LBL_TRAIN / lbl_name)

    for src_img, src_lbl, img_name, lbl_name in val_pairs:
        shutil.copy2(src_img, DEST_IMG_VAL / img_name)
        shutil.copy2(src_lbl, DEST_LBL_VAL / lbl_name)

    yaml_path = DEST_BASE / "data.yaml"
    yaml_path.write_text(
        f"path: {DEST_BASE.resolve()}\n"
        "train: images/train\n"
        "val: images/val\n"
        "\n"
        "names:\n"
        "  0: person\n"
        "  1: weapon\n",
        encoding="utf-8",
    )
    print(f"[OK] Wrote {yaml_path}")
    print(f"[OK] Dataset prepared at {DEST_BASE.resolve()}")


if __name__ == "__main__":
    main()
