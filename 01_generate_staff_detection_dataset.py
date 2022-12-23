import json
import os
import random

import tqdm
import albumentations as A
import cv2
import numpy as np
from PIL import Image, ImageDraw

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


def load_mask(pnb_path, sample_idx, mask_idx, hull=False):
    mask = cv2.imread(
        f"{pnb_path}/{sample_idx}/out-{mask_idx}.png", cv2.IMREAD_GRAYSCALE
    )
    mask[-2:, :] = 0
    if not hull:
        return mask

    # create convex hull
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    hulls = []
    for contour in contours:
        hull = cv2.convexHull(contour)
        hulls.append(hull)
        cv2.drawContours(mask, [hull], 0, (255), -1)
    # mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    return mask


def get_sample(sample_idx):

    # load sample from dataset
    img = cv2.imread(f"{PNG_PATH}/{sample_idx}/out-0.png")
    img = 255 - cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # generate staff image from bounding boxes
    with open(f"{BBOX_PATH}/{sample_idx}.json", "r") as bboxfile:
        bboxes = json.load(bboxfile)
    np.unique([bbox["type"] for bbox in bboxes])
    staffs = [bbox for bbox in bboxes if bbox["type"] == "Staff"]
    img_h, img_w = img.shape[:2]
    staff_mask = Image.fromarray(np.zeros_like(img))
    draw = ImageDraw.Draw(staff_mask)
    for staff in staffs:
        x = staff["x"]
        y = staff["y"]
        w = staff["width"]
        h = staff["height"]
        draw.rectangle(
            (x * img_w, y * img_h, (x + w) * img_w, (y + h) * img_h), fill="#fff"
        )
    staff_mask = cv2.cvtColor(np.array(staff_mask), cv2.COLOR_BGR2GRAY)
    
    staff_mask = load_mask(PNG_PATH, sample_idx, 48)

    # generate brace image as convex hulls
    brace_mask = load_mask(PNG_PATH, sample_idx, 3, hull=True)

    # generate creshendo / diminuendo masks as convex hulls
    creshendo_mask = load_mask(PNG_PATH, sample_idx, 45, hull=True)
    diminuendo_mask = load_mask(PNG_PATH, sample_idx, 46, hull=True)

    # combine masks
    comb_mask = np.zeros_like(img)
    comb_mask[staff_mask != 0] = (255, 0, 0)
    comb_mask[brace_mask != 0] = (0, 255, 0)
    comb_mask[creshendo_mask != 0] = (0, 0, 255)
    comb_mask[diminuendo_mask != 0] = (0, 255, 255)

    return img, comb_mask


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
    img, comb_mask = get_sample(sample_idx)
    
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

