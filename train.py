import numpy as np
import yaml
import os
from PIL import Image
from ultralytics import YOLO

import ultralytics
print("Ultralytics from:", ultralytics.__file__)

# ==================== DUMMY DATASET ====================
DATASET_ROOT = 'dummy_dataset'
NUM_CLASS    = 3
IMG_SIZE     = 640
NUM_TRAIN    = 20
NUM_VAL      = 8

for split in ['train', 'val']:
    os.makedirs(f'{DATASET_ROOT}/images/{split}', exist_ok=True)
    os.makedirs(f'{DATASET_ROOT}/labels/{split}', exist_ok=True)

def create_dummy_image(path):
    arr = np.random.randint(0, 255, (IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
    Image.fromarray(arr).save(path)

def create_dummy_label(path, num_objects=3):
    with open(path, 'w') as f:
        for _ in range(num_objects):
            cls    = np.random.randint(0, NUM_CLASS)
            cx, cy = np.random.uniform(0.2, 0.8, 2)
            w,  h  = np.random.uniform(0.1, 0.4, 2)
            f.write(f"{cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")

for i in range(NUM_TRAIN):
    create_dummy_image(f'{DATASET_ROOT}/images/train/img_{i:04d}.jpg')
    create_dummy_label(f'{DATASET_ROOT}/labels/train/img_{i:04d}.txt')

for i in range(NUM_VAL):
    create_dummy_image(f'{DATASET_ROOT}/images/val/img_{i:04d}.jpg')
    create_dummy_label(f'{DATASET_ROOT}/labels/val/img_{i:04d}.txt')

print("Dummy dataset ready")

# ==================== DATASET YAML ====================
dataset_cfg = {
    'path'  : os.path.abspath(DATASET_ROOT),
    'train' : 'images/train',
    'val'   : 'images/val',
    'nc'    : NUM_CLASS,
    'names' : {i: f'class_{i}' for i in range(NUM_CLASS)},
}
dataset_yaml = f'{DATASET_ROOT}/dataset.yaml'
with open(dataset_yaml, 'w') as f:
    yaml.dump(dataset_cfg, f, default_flow_style=False)

# ==================== TRAIN ====================
model = YOLO('ultralytics/cfg/models/11/yolo11.yaml')

results = model.train(
    data     = dataset_yaml,
    epochs   = 3,
    imgsz    = 320,
    batch    = 4,
    device   = 0,
    project  = os.path.abspath('runs/dummy_train'),
    name     = 'test_run',
    exist_ok = True,
)

print("Done! Results saved to:", results.save_dir)
