import numpy as np


def area(bbox):
    x0, y0, x1, y1 = bbox
    return (x1 - x0) * (y1 - y0)


def intersection_over_union(a, b, eps=(1, 2)):
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ax0 -= eps[0]
    ay0 -= eps[1]
    bx0 -= eps[0]
    by0 -= eps[1]
    ax1 += eps[0]
    ay1 += eps[1]
    bx1 += eps[0]
    by1 += eps[1]
    if bx0 >= ax1 or by0 >= ay1:
        return 0
    if ax0 >= bx1 or ay0 >= by1:
        return 0
    x0, x1 = sorted([ax0, ax1, bx0, bx1])[1:3]
    y0, y1 = sorted([ay0, ay1, by0, by1])[1:3]
    intersection = (x1 - x0) * (y1 - y0)
    union = (ax1 - ax0) * (ay1 - ay0) + (bx1 - bx0) * (by1 - by0) - intersection
    return intersection / union


def within(a, b):
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    return bx0 >= ax0 and by0 >= ay0 and bx1 <= ax1 and by1 <= ay1


def h_dist(a, b):
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    if bx0 > ax1:
        return bx0 - ax1
    if ax0 > bx1:
        return ax1 - bx0
    return 0


def from_mask(mask):
    tmp = np.indices(mask.shape[:2])[:, mask > 0]
    y0, x0 = np.min(tmp, axis=1)
    y1, x1 = np.max(tmp, axis=1)
    return x0, y0, x1, y1
