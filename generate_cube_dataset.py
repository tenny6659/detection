import os
import random
import numpy as np
from PIL import Image, ImageDraw
from tqdm import tqdm

# ==========================
# CONFIG
# ==========================
IMG_SIZE = 416
TOTAL_IMAGES = 30     # ✅ VERY SMALL DATASET
TRAIN_SPLIT = 0.8
SAVE_DIR = "cube_dataset"

# Create folders
for p in [
    "images/train", "images/val",
    "labels/train", "labels/val"
]:
    os.makedirs(f"{SAVE_DIR}/{p}", exist_ok=True)

# ==========================
# RANDOM BACKGROUND
# ==========================
def random_background(size):
    top = np.random.randint(0, 255, (1, 1, 3), dtype=np.uint8)
    bottom = np.random.randint(0, 255, (1, 1, 3), dtype=np.uint8)
    arr = np.linspace(0, 1, size).reshape(size, 1, 1)
    bg = (top * (1 - arr) + bottom * arr).astype(np.uint8)
    bg = np.tile(bg, (1, size, 1))
    return Image.fromarray(bg)

# ==========================
# DRAW CUBE
# ==========================
def draw_cube(img, scale=120):
    draw = ImageDraw.Draw(img)
    cx, cy = IMG_SIZE // 2, IMG_SIZE // 2
    s = scale

    p1 = (cx - s, cy - s)
    p2 = (cx + s, cy - s)
    p3 = (cx + s, cy + s)
    p4 = (cx - s, cy + s)

    offset = int(s * 0.4)
    p5 = (p1[0] + offset, p1[1] - offset)
    p6 = (p2[0] + offset, p2[1] - offset)
    p7 = (p3[0] + offset, p3[1] - offset)

    draw.polygon([p1, p2, p3, p4], fill=(255, 80, 80))
    draw.polygon([p1, p2, p6, p5], fill=(200, 60, 60))
    draw.polygon([p2, p3, p7, p6], fill=(150, 40, 40))

    x_min = min(p1[0], p5[0])
    y_min = min(p5[1], p1[1])
    x_max = max(p3[0], p7[0])
    y_max = max(p3[1], p7[1])

    return img, (x_min, y_min, x_max, y_max)

# ==========================
# YOLO FORMAT
# ==========================
def yolo_format(box):
    x1, y1, x2, y2 = box
    w = x2 - x1
    h = y2 - y1
    cx = x1 + w / 2
    cy = y1 + h / 2
    return cx / IMG_SIZE, cy / IMG_SIZE, w / IMG_SIZE, h / IMG_SIZE

# ==========================
# GENERATE IMAGES
# ==========================
train_count = int(TOTAL_IMAGES * TRAIN_SPLIT)
val_count = TOTAL_IMAGES - train_count

def generate(n, mode):
    for i in tqdm(range(n), desc=f"Generating {mode}"):
        bg = random_background(IMG_SIZE)
        scale = random.randint(80, 140)
        img, box = draw_cube(bg, scale)
        cx, cy, w, h = yolo_format(box)

        img.save(f"{SAVE_DIR}/images/{mode}/{i}.jpg")
        with open(f"{SAVE_DIR}/labels/{mode}/{i}.txt", "w") as f:
            f.write(f"0 {cx} {cy} {w} {h}")

generate(train_count, "train")
generate(val_count, "val")

print("✅ Dataset generated successfully!")
