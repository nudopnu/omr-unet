import json

import cv2
import numpy as np
from PIL import Image, ImageDraw


def load_mask(png_path, sample_idx, mask_idx, hull=False):

    mask = cv2.imread(
        f"{png_path}/{sample_idx}/out-{mask_idx}.png", cv2.IMREAD_GRAYSCALE
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

def load_classIdxMap():
    with open('../../datasets/generated/classlist.json', "r") as jsonfile:
        classlist = json.load(jsonfile)
    return { k['class']:k['id'] for k in classlist}

def load_bboxes(bbox_path, sample_idx, name, img_shape=None):

    with open(f"{bbox_path}/{sample_idx}.json", "r") as bboxfile:
        bboxes = json.load(bboxfile)

    bboxes = [bbox for bbox in bboxes if bbox["type"] == name]
    if img_shape == None:
        return bboxes

    res = []
    for bbox in bboxes:
        x = int(bbox["x"] * img_shape[1])
        y = int(bbox["y"] * img_shape[0])
        w = int(bbox["width"] * img_shape[1])
        h = int(bbox["height"] * img_shape[0])
        res.append(np.array([x, y, w, h]))
    return res

def load_stafflines(png_path, bbox_path, sample_idx):
    img = cv2.imread("../../datasets/generated/png/001/out-0.png")
    staff_mask = load_mask(png_path, sample_idx, 48)
    staffs = load_bboxes(bbox_path, sample_idx, "Staff", img.shape)

    res = np.zeros_like(staffs)

    staff_ys = np.where(staff_mask[:, 200:201] > 0)[0]
    staff_lines = []

    y_old = -1
    cur_ys = []
    for y in staff_ys:
        if (y_old < y - 1 and y_old != -1) or y == staff_ys[-1]:
            y0 = cur_ys[0]
            y1 = cur_ys[-1]
            x_vals = np.where(staff_mask[y1:y1+1, :] > 0)[1]
            x0 = x_vals[0]
            x1 = x_vals[-1]
            staff_lines.append(np.array([x0, x1, y0, y1]))
            cur_ys = []
        cur_ys.append(y)
        y_old = y
    return np.array(staff_lines)

def get_sample(png_path, bbox_path, sample_idx):

    # load sample from dataset
    img = cv2.imread(f"{png_path}/{sample_idx}/out-0.png")
    img = 255 - cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # generate staff image from bounding boxes
    name = "Staff"
    staffs = load_bboxes(bbox_path, sample_idx, name)

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
    
    staff_mask = load_mask(png_path, sample_idx, 48)

    # generate brace image as convex hulls
    brace_mask = load_mask(png_path, sample_idx, 3, hull=True)

    # generate creshendo / diminuendo masks as convex hulls
    creshendo_mask = load_mask(png_path, sample_idx, 45, hull=True)
    diminuendo_mask = load_mask(png_path, sample_idx, 46, hull=True)

    # combine masks
    comb_mask = np.zeros_like(img)
    comb_mask[staff_mask != 0] = (255, 0, 0)
    comb_mask[brace_mask != 0] = (0, 255, 0)
    comb_mask[creshendo_mask != 0] = (0, 0, 255)
    comb_mask[diminuendo_mask != 0] = (0, 255, 255)

    return img, comb_mask

