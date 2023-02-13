import numpy as np
import cv2
from sklearn.cluster import DBSCAN
from sklearn.mixture import GaussianMixture

from omr.delta_space import (
    analyze,
    get_staff_line_points,
    get_curvature_from_points,
)
from omr.morphology import (
    single_channel,
    sparse,
    dilation,
    chain,
    closing,
    sharpen,
    dense,
)


def detect_staffs(img, top_k_gaps=2, y_scale_before_clustering=3):

    # # original
    # img = dense()(cv2.imread(img_path))

    # preprocessing
    preprocess = chain(sparse(), single_channel(), sharpen())
    src = preprocess(img)

    # get staffline points
    gap_space, gap_dict, staff_gap, staff_thickness = analyze(src)
    lines, points = get_staff_line_points(gap_space, gap_dict, src.shape, k=top_k_gaps)
    canvas, curvature = get_curvature_from_points(points, staff_gap, src.shape[:2])
    staff_height = (staff_gap + staff_thickness) * 4 + staff_thickness

    # cluster y-scaled staffline points
    X = np.argwhere(lines > 0)
    X[:, 0] *= y_scale_before_clustering
    db = DBSCAN(eps=y_scale_before_clustering * staff_gap, min_samples=2).fit(X)
    labels = db.labels_ + 1

    # filter labels with gaussian mixture model
    counts = np.bincount(labels * (labels > 0))
    important_labels = np.indices((len(counts),))[0]
    if len(counts) > 5:
        gmm = GaussianMixture(n_components=2, tol=1e-9, max_iter=0).fit(counts[:, None])
        gmm_labels = gmm.predict(counts[:, None])
        low_mean = np.mean(counts[gmm_labels == 0])
        high_mean = np.mean(counts[gmm_labels == 1])
        mid_mean = (low_mean + high_mean) / 2
        important_labels = important_labels[counts > mid_mean]

    # plot clustered points
    labeled_lines = np.zeros(lines.shape, np.uint8)
    for idx, (y, x) in enumerate(X):
        if labels[idx] in important_labels:
            labeled_lines[int(y / y_scale_before_clustering), x] = labels[idx]

    # get morphologically closed image
    closed = chain(
        single_channel(),
        sparse(),
        lambda img: (img > 0).astype(np.uint8) * 255,
        closing(2 * staff_gap),
    )(img)

    # get dilated labeled staff line points and mask with closed image
    staff_areas = dilation(y_scale_before_clustering * staff_gap)(labeled_lines)
    staff_areas[closed == 0] = 0

    # estimate staff boxes
    staff_bounding_boxes = []
    for label in important_labels:
        mask = (staff_areas == label).astype(np.uint8) * 255
        sums = np.sum(np.indices(mask.shape)[0] * (mask > 0), axis=0)
        sums = sums[sums > 0] / np.sum((mask > 0), axis=0)[sums > 0]
        sums[np.isnan(sums)] = 0
        med = np.median(sums)
        x0 = np.min(np.indices(mask.shape)[1][(mask > 0)])
        y0 = int(med - staff_height / 2)
        x1 = np.max(np.indices(mask.shape)[1][(mask > 0)])
        y1 = int(med + staff_height / 2)
        staff_bounding_boxes.append([x0, y0, x1, y1])

    # apply curvature to boxes if any present
    if np.sum(curvature) <= 10:
        curvature *= 0

    # print("staff_gap:", staff_gap)
    # print("staff_thickness:", staff_thickness)
    # print("num_staffs:", len(important_labels))
    return staff_areas, staff_bounding_boxes, staff_gap, staff_thickness, curvature
