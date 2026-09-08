"""Dataset exporter: generates YOLO directory structures, train/val splits, and data.yaml."""

import os
import shutil
try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False
from pathlib import Path
from typing import List, Dict, Any, Tuple


def export_yolo_dataset(
    processed_images_dir: str,
    labels_dir: str,
    output_yolo_dir: str,
    val_split: float = 0.2,
    mode: str = "2-class",
    seed: int = 42
) -> Dict[str, Any]:
    """
    Splits images and corresponding .txt labels into train and val folders,
    and generates the standard data.yaml needed for YOLOv8 model.train().
    """
    random.seed(seed)
    processed_path = Path(processed_images_dir)
    labels_path = Path(labels_dir)
    output_path = Path(output_yolo_dir)

    # Prepare directory tree
    train_images = output_path / "images" / "train"
    val_images = output_path / "images" / "val"
    train_labels = output_path / "labels" / "train"
    val_labels = output_path / "labels" / "val"

    for d in [train_images, val_images, train_labels, val_labels]:
        os.makedirs(d, exist_ok=True)

    # Find paired images and labels
    label_files = list(labels_path.glob("*.txt"))
    paired_samples = []

    for lf in label_files:
        stem = lf.stem
        # Search for corresponding image
        for ext in [".jpg", ".jpeg", ".png", ".webp"]:
            candidate_img = processed_path / f"{stem}{ext}"
            if candidate_img.exists():
                paired_samples.append((candidate_img, lf))
                break

    if not paired_samples:
        raise ValueError(f"No matching image-label pairs found between '{processed_images_dir}' and '{labels_dir}'.")

    random.shuffle(paired_samples)
    val_count = max(1, int(len(paired_samples) * val_split))
    val_samples = paired_samples[:val_count]
    train_samples = paired_samples[val_count:]

    # Copy files
    for img, lbl in train_samples:
        shutil.copy2(img, train_images / img.name)
        shutil.copy2(lbl, train_labels / lbl.name)

    for img, lbl in val_samples:
        shutil.copy2(img, val_images / img.name)
        shutil.copy2(lbl, val_labels / lbl.name)

    # Build data.yaml
    if mode == "3-class":
        class_names = ["closed_door", "partially_open_door", "wide_open_door"]
    else:
        class_names = ["open_door", "closed_door"]

    data_yaml_content = {
        "path": str(output_path.resolve()),
        "train": "images/train",
        "val": "images/val",
        "names": {i: name for i, name in enumerate(class_names)}
    }

    yaml_file_path = output_path / "data.yaml"
    if HAS_YAML:
        with open(yaml_file_path, "w") as yf:
            yaml.dump(data_yaml_content, yf, default_flow_style=False, sort_keys=False)
    else:
        names_str = "\n".join([f"  {i}: {name}" for i, name in enumerate(class_names)])
        manual_yaml = f"path: {output_path.resolve()}\ntrain: images/train\nval: images/val\nnames:\n{names_str}\n"
        with open(yaml_file_path, "w") as yf:
            yf.write(manual_yaml)

    summary = {
        "total_pairs": len(paired_samples),
        "train_count": len(train_samples),
        "val_count": len(val_samples),
        "data_yaml": str(yaml_file_path.resolve()),
        "classes": class_names
    }

    print(f"📦 YOLO Dataset Export Complete!")
    print(f"   ├─ Train set: {len(train_samples)} pairs")
    print(f"   ├─ Val set:   {len(val_samples)} pairs")
    print(f"   └─ Config:    {yaml_file_path}")

    return summary
