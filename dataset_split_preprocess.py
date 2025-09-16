import os
import shutil
import random

# --- Paths ---
DATA_DIR = r"C:\Users\ASUS\Desktop\smart_waste_management\archive (1)\dataset-resized"  # original dataset with class folders
OUTPUT_DIR = r"C:\Users\ASUS\Desktop\smart_waste_management\data"  # where train/val/test will be created
split_ratio = (0.7, 0.15, 0.15)  # train, val, test

# --- Create folder
for split in ["train", "val", "test"]:
    for class_name in os.listdir(DATA_DIR):
        os.makedirs(os.path.join(OUTPUT_DIR, split, class_name), exist_ok=True)


for class_name in os.listdir(DATA_DIR):
    class_path = os.path.join(DATA_DIR, class_name)
    images = os.listdir(class_path)
    random.shuffle(images)

    n_total = len(images)
    n_train = int(split_ratio[0] * n_total)
    n_val = int(split_ratio[1] * n_total)

    train_imgs = images[:n_train]
    val_imgs = images[n_train:n_train + n_val]
    test_imgs = images[n_train + n_val:]

    for img in train_imgs:
        shutil.copy(os.path.join(class_path, img), os.path.join(OUTPUT_DIR, "train", class_name, img))
    for img in val_imgs:
        shutil.copy(os.path.join(class_path, img), os.path.join(OUTPUT_DIR, "val", class_name, img))
    for img in test_imgs:
        shutil.copy(os.path.join(class_path, img), os.path.join(OUTPUT_DIR, "test", class_name, img))

print("Dataset split completed!")
