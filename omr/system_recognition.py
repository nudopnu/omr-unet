import cv2
import numpy as np
from omr.morphology import (
    dilation,
    chain,
    closing,
)


def get_coarse_masks(src, thresh=105):

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

    # get max areas
    areas = stats[:, cv2.CC_STAT_AREA]
    threshold = np.mean(areas) + np.std(areas)
    max_area_idcs = np.array(np.where(areas > threshold))[0]
    rem_area_idcs = np.array(np.where(areas <= threshold))[0]

    # get outer contour for each max area and fill it as mask
    max_area_contours = []
    max_area_masks = []
    for idx in max_area_idcs:
        mask = (labels == idx).astype(np.uint8)
        mask = chain(
            closing(12),
            dilation((12, 24)),
        )(mask.astype(np.uint8))
        contours, hierarchy = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        max_area_contours.append(contours[0])
        max_area_masks.append(mask)
        cv2.fillPoly(mask, contours, 1)

    # assign remaining areas to the contours via distance
    for idx in rem_area_idcs:
        x = stats[idx, cv2.CC_STAT_LEFT]
        y = stats[idx, cv2.CC_STAT_TOP]
        w = stats[idx, cv2.CC_STAT_WIDTH]
        h = stats[idx, cv2.CC_STAT_HEIGHT]
        area = stats[idx, cv2.CC_STAT_AREA]
        cx = int(x) + int(w / 2)
        cy = int(y) + int(h / 2)
        for i, label in enumerate(max_area_idcs):
            dist = cv2.pointPolygonTest(max_area_contours[i], (cx, cy), True)
            if dist > -thresh:
                max_area_masks[i][labels == idx] = 1

    return max_area_masks, max_area_contours


def extract_scaled_systems(img, max_area_masks):

    crops = []
    mappings = []
    system_bounding_boxes = []
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    vis = np.ones_like(gray) * 255

    for staff_idx in range(len(max_area_masks)):
        # extract from input img with mask
        mask = max_area_masks[staff_idx] > 0
        vis[mask] = gray[mask]

        # crop bounding box
        y_idcs, x_idcs = np.indices(mask.shape)
        x0 = np.min(x_idcs[mask])
        x1 = np.max(x_idcs[mask])
        y0 = np.min(y_idcs[mask])
        y1 = np.max(y_idcs[mask])
        bbox = [x0, y0, x1, y1]
        crop = vis[y0:y1, x0:x1]

        # rescale to get a height of 256
        factor = 256 / (y1 - y0)
        target_width = int((x1 - x0) * factor)
        n_blocks = (target_width // 256) + 1
        resized_crop = cv2.resize(crop, (target_width, 256), cv2.INTER_NEAREST)

        # pad crop to make the length a multiple of 256
        pad = np.ones((256, n_blocks * 256 - target_width), np.uint8) * 255
        padded_crop = np.concatenate((resized_crop, pad), axis=1)
        crops.append(padded_crop)

        def mapping(coords, inv=False):
            coords = np.array(coords)
            if not inv:
                xs = (coords[::2] - x0) * factor
                ys = (coords[1::2] - y0) * factor
            else:
                xs = (coords[::2] / factor) + x0
                ys = (coords[1::2] / factor) + y0
            coords[::2] = xs.astype(coords.dtype)
            coords[1::2] = ys.astype(coords.dtype)
            return coords

        mappings.append(mapping)
        system_bounding_boxes.append(bbox)

    return crops, mappings, system_bounding_boxes
