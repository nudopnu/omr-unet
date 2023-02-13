import numpy as np
import scipy.ndimage
import cv2
from .morphology import getAreaAttributes
from omr.morphology import getLocalExtrema


def extractParts(img, staff_gap=10, preprocess=lambda img: img):

    # copy and apply preprocessing
    src = img.copy()
    src = preprocess(src)

    # get connected components
    connectivity = 4
    (
        num_labels,
        labels,
        stats,
        centroids,
    ) = cv2.connectedComponentsWithStatsWithAlgorithm(
        src.astype(np.int8), connectivity, cv2.CV_32S, ccltype=cv2.CCL_DEFAULT
    )

    # remove background
    num_labels = num_labels
    stats = stats[1:]
    centroids = centroids[1:]
    num_labels = num_labels - 1
    labels = labels - 1

    # threshold areas
    areas = stats[:, cv2.CC_STAT_AREA]
    threshold = np.mean(areas) + 2 * np.std(areas)

    # obtain max areas (above threshold)
    max_area_idcs = np.array(np.where(areas > threshold))[0]
    rem_area_idcs = np.array(np.where(areas <= threshold))[0]
    max_areas = {}

    for idx in max_area_idcs:
        x, y, w, h, area, cx, cy = getAreaAttributes(stats, centroids, idx)
        bbox = np.array([x, y, w, h])
        max_areas[idx] = {"bbox": bbox, "center": np.array([cx, cy])}

    # determine problem areas (too big ones)
    top_k_areas = areas[max_area_idcs]
    k = len(top_k_areas)
    problem_areas = max_area_idcs[
        top_k_areas > (np.median(top_k_areas) + 2 * np.std(top_k_areas))
    ]

    # fix problem areas
    for problem_area_index in problem_areas:

        # horizontal sum cropped
        problem = labels == problem_area_index
        sums = np.sum(problem, 1)
        roi = np.min(np.nonzero(sums)), np.max(np.nonzero(sums))

        # determine local extrema
        smooth = scipy.ndimage.gaussian_filter1d(sums, staff_gap)
        smooth = scipy.ndimage.median_filter(smooth, staff_gap * 2)
        minima, maxima = getLocalExtrema(smooth[roi[0] : roi[1]])
        minima = minima + roi[0]

        # if minima not found -> infeasable
        if len(minima) == 0:
            print("Could not fix area")
            continue

        # initial split (horizontal line)
        # todo: extend to multiple splits
        split = (
            minima[0]
            - staff_gap
            + np.argmin(sums[minima[0] - 10 : minima[0] + staff_gap])
        )

        # search for path within gaps if points are crossed
        refine_split = np.repeat(split, problem.shape[1]).T
        offset = 30
        for idx in np.nonzero(problem[split, :] > 0)[0]:
            excerpt = problem[
                split - offset : split + offset, idx - offset : idx + offset
            ]
            refine_split[idx] = split - offset + np.argmin(np.sum(excerpt, 1))

        # generate mask from refined split
        mask = np.zeros_like(problem)
        for x, y in enumerate(refine_split):
            mask[y:, x] = 1

        # apply mask to problem area and replace labels
        refine_resolve1 = problem * ((1 - mask) > 0)
        refine_resolve2 = problem * mask

        r1_idx = num_labels
        r2_idx = r1_idx + 1
        labels[refine_resolve1 == True] = r1_idx
        labels[refine_resolve2 == True] = r2_idx

        # create new bunding boxes
        x, y, w, h, _, cx, cy = getAreaAttributes(stats, centroids, problem_area_index)
        r1_bbox = np.array([x, y, w, split - y])
        r1_center = np.array([cx, int((y + split) / 2)], np.uint16)
        r2_bbox = np.array([x, split, w, y + h - split])
        r2_center = np.array([cx, int((y + h + split) / 2)], np.uint16)
        max_areas[r1_idx] = {"bbox": r1_bbox, "center": r1_center}
        max_areas[r2_idx] = {"bbox": r2_bbox, "center": r2_center}

        # remove old problem areas
        if problem_area_index in max_areas:
            del max_areas[problem_area_index]

    return labels, max_areas


def split_img(
    img, labels, areas, callback=lambda i, _: None, preprocess_mask=lambda mask: mask
):
    max_area_idcs = np.array([(k, areas[k]["center"][1]) for k in areas])
    max_area_idcs = sorted(max_area_idcs, key=lambda x: x[1])

    canvas = np.zeros_like(labels)
    splits = []

    for i, (idx, y) in enumerate(max_area_idcs):

        canvas[labels == idx] = idx
        mask = canvas == idx
        mask = preprocess_mask(mask)

        mask_idcs = np.where(mask == True)
        y0, x0 = np.min(mask_idcs[0]), np.min(mask_idcs[1])
        y1, x1 = np.max(mask_idcs[0]), np.max(mask_idcs[1])

        res = (mask[y0:y1, x0:x1] > 0)[..., None] * img[y0:y1, x0:x1]
        res = 255 - res

        splits.append(res)
        callback(i, res)
    return splits
