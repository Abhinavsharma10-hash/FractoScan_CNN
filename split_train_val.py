import os, shutil, random
from pathlib import Path

# Paths
TRAIN_FOLDER = Path("data/train")   # Original images are here now
VAL_FOLDER = Path("data/val")       # Will create validation images here

CLASSES = ["fractured", "non_fractured"]
SPLIT_RATIO = 0.8  # 80% train, 20% val

# Create val folders if not exist
for cls in CLASSES:
    (VAL_FOLDER / cls).mkdir(parents=True, exist_ok=True)

# Split each class
for cls in CLASSES:
    class_train_path = TRAIN_FOLDER / cls
    class_val_path = VAL_FOLDER / cls

    # Get all images
    images = [f for f in os.listdir(class_train_path) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
    random.shuffle(images)

    split_idx = int(len(images) * SPLIT_RATIO)
    train_imgs = images[:split_idx]
    val_imgs = images[split_idx:]

    # Move val images to val folder
    for fname in val_imgs:
        shutil.move(class_train_path / fname, class_val_path / fname)

    print(f"{cls}: {len(train_imgs)} train, {len(val_imgs)} val")
