import os
import random

import albumentations as A
import cv2
import numpy as np
import tqdm

from utils.dataset import get_sample

DATASET_PATH = os.path.join("..", "..", "datasets", "generated")
PNG_PATH = os.path.join(DATASET_PATH, "png")
BBOX_PATH = os.path.join(DATASET_PATH, "bbox")
OUT_PATH = os.path.join(DATASET_PATH, "staff-detection")
OUT_TRAIN_PATH = os.path.join(OUT_PATH, "train")
OUT_TEST_PATH = os.path.join(OUT_PATH, "test")
OUT_VALID_PATH = os.path.join(OUT_PATH, "valid")

OUT_PATHS = [OUT_TRAIN_PATH, OUT_TEST_PATH, OUT_VALID_PATH]
SPLITS = [20, 3, 2]
SPLIT_POINTS = np.cumsum(SPLITS)

INITIAL_SEED = 41
NUM_AUG_PER_SAMPLE = 12
MAX_SAMPLE_IDX = 25


# prepare augmentations
transform = A.Compose(
    [
        A.SafeRotate(limit=(-45, 45), always_apply=True, border_mode=cv2.BORDER_CONSTANT),
        A.Affine(scale=0.9, always_apply=True, mode=cv2.BORDER_CONSTANT),
        A.RandomScale(scale_limit=(0.3, 1)),
    ]
)

seed = INITIAL_SEED
out_idx = 1
group_idx = 0

for idx in range(1, MAX_SAMPLE_IDX + 1):
    
    if idx > SPLIT_POINTS[group_idx]:
        group_idx += 1
        out_idx = 1
    
    out_path = OUT_PATHS[group_idx]
    out_x_path = os.path.join(out_path, "x")
    out_y_path = os.path.join(out_path, "y")
    
    # ensure output directories
    os.makedirs(out_x_path, exist_ok=True)
    os.makedirs(out_y_path, exist_ok=True)
    
    sample_idx = f"{idx:03d}"
    img, comb_mask = get_sample(PNG_PATH, BBOX_PATH, sample_idx)
    
    print(f"Augmenting sample with idx: {idx}/{MAX_SAMPLE_IDX} to {out_path}", )
    
    for i in tqdm.tqdm(range(NUM_AUG_PER_SAMPLE)):
        random.seed(seed)
        seed += 1
        out_idx += 1
        
        aug_data = transform(image=img, mask=comb_mask)
        x, y = aug_data["image"], aug_data["mask"]
        x, y = cv2.resize(x, (1024, 1024)), cv2.resize(y, (1024, 1024))
        cv2.imwrite(os.path.join(out_x_path, f'{out_idx:03d}' + '.png'), 255 - x)
        cv2.imwrite(os.path.join(out_y_path, f'{out_idx:03d}' + '.png'), y)

