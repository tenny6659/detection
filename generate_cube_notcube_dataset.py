import os
import random
import numpy as np
from PIL import Image, ImageDraw
from tqdm import tqdm

# ================= CONFIG =================
IMG_SIZE = 416
TOTAL_IMAGES = 60        # small but effective
TRAIN_SPLIT = 0.8
SAVE_DIR = "cube_dataset"

# Create folders
for p in [
    "images/train", "images/val",
    "labels/train", "labels/val"
]:
    os.makedirs(f"{SAVE_DIR}/{p}", exist_ok=True)

# ================= BACKGROUND =================
def random_background():
    top = np.random.randint(0, 255, (1, 1, 3), dtype=np.uint8)
    bottom = np.random.randint(0, 255, (1, 1, 3), dtype=np.uint8)
    arr = np.linspace(0, 1, IMG_SIZE).reshape(IMG_SIZE, 1, 1)
    bg = (top * (1 - arr) + bottom * arr).astype(np.uint8)
    bg = np.tile(bg, (1, IMG_SIZE, 1))
    return Image.fromarray(bg)

# ================= DRAW CUBE =================
def draw_cube(img):
    draw = ImageDraw.Draw(img)
    s = random.randint(80, 140)
    cx, cy = IMG_SIZE // 2, IMG_SIZE // 2

    x1, y1 = cx - s, cy - s
    x2, y2 = cx + s, cy + s

    draw.rectangle([x1, y1, x2, y2], fill=(255, 80, 80))
    return (x1, y1, x2, y2), 0   # class 0 → cube

# ================= DRAW NON-CUBE (CIRCLE) =================
def draw_circle(img):
    draw = ImageDraw.Draw(img)
    r = random.randint(60, 120)
    cx, cy = IMG_SIZE // 2, IMG_SIZE // 2

    x1, y1 = cx - r, cy - r
    x2, y2 = cx + r, cy + r

    draw.ellipse([x1, y1, x2, y2], fill=(80, 80, 255))
    return (x1, y1, x2, y2), 1   # class 1 → not_cube

# ================= YOLO FORMAT =================
def yolo_format(box):
    x1, y1, x2, y2 = box
    w = x2 - x1
    h = y2 - y1
    cx = x1 + w / 2
    cy = y1 + h / 2
    return cx / IMG_SIZE, cy / IMG_SIZE, w / IMG_SIZE, h / IMG_SIZE

# ================= GENERATE =================
train_count = int(TOTAL_IMAGES * TRAIN_SPLIT)
val_count = TOTAL_IMAGES - train_count

def generate(n, mode):
    for i in tqdm(range(n), desc=f"Generating {mode}"):
        img = random_background()

        if random.random() < 0.5:
            box, cls = draw_cube(img)
        else:
            box, cls = draw_circle(img)

        cx, cy, w, h = yolo_format(box)

        img.save(f"{SAVE_DIR}/images/{mode}/{i}.jpg")
        with open(f"{SAVE_DIR}/labels/{mode}/{i}.txt", "w") as f:
            f.write(f"{cls} {cx} {cy} {w} {h}")

generate(train_count, "train")
generate(val_count, "val")

print("✅ Cube + Not-Cube dataset generated!")
