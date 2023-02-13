import numpy as np
import cv2


def spaghetti(img):
    # get connected components
    connectivity = 4
    (
        num_labels,
        labels,
        stats,
        centroids,
    ) = cv2.connectedComponentsWithStatsWithAlgorithm(
        img, connectivity, cv2.CV_32S, ccltype=cv2.CCL_DEFAULT
    )

    # remove background
    num_labels = num_labels
    stats = stats[1:]
    centroids = centroids[1:]
    num_labels = num_labels - 1
    labels = labels - 1

    # obtain contours
    contours = []
    for i in range(num_labels):
        mask = (labels == i).astype(np.uint8)
        sub_contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
        )
        contours.append(sub_contours[0])

    # obtain areas
    areas = stats[:, cv2.CC_STAT_AREA]

    return labels, areas, contours


def border_following(img):
    # extract outer contours
    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # visualize contours and obtain infos
    labels = np.zeros((img.shape[0], img.shape[1]), np.int32)
    areas = np.zeros(len(contours), np.uint32)
    for i, contour in enumerate(contours):
        labels = cv2.fillPoly(labels, [contour], i)
        areas[i] = cv2.contourArea(contour)

    return labels, areas, contours


# ---- Utility functions


def contour_mask(output_shape, contour, color):
    res = np.zeros(output_shape, np.uint8)
    return cv2.fillPoly(res, [contour], color)


# taken from https://stackoverflow.com/questions/55641425/check-if-two-contours-intersect
def ccw(A, B, C):
    return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])


def contour_intersect(cnt_ref, cnt_query):

    ## Contour is a list of points
    ## Connect each point to the following point to get a line
    ## If any of the lines intersect, then break

    for ref_idx in range(len(cnt_ref) - 1):
        ## Create reference line_ref with point AB
        A = cnt_ref[ref_idx][0]
        B = cnt_ref[ref_idx + 1][0]

        for query_idx in range(len(cnt_query) - 1):
            ## Create query line_query with point CD
            C = cnt_query[query_idx][0]
            D = cnt_query[query_idx + 1][0]

            ## Check if line intersect
            if ccw(A, C, D) != ccw(B, C, D) and ccw(A, B, C) != ccw(A, B, D):
                ## If true, break loop earlier
                return True

    return False


# ---- Utilities for visualisation


class RandomColor:
    def __init__(self):
        self.idx = 1

    def next(self):
        frac = self.idx * 7
        self.idx += 1
        return ((frac * 23) % 256, (frac * 13) % 256, (frac * 7) % 256)

    def reset(self):
        self.idx = 1


def vis_mask(sparse_img, contours, fill=True, line=True, overlay=True):
    res = np.zeros((sparse_img.shape[0], sparse_img.shape[1], 3), np.uint8)
    color = RandomColor()
    for contour in contours:
        if fill:
            res = cv2.fillPoly(res, [contour], (255, 255, 255))
        if line:
            res = cv2.drawContours(
                res,
                [contour],
                -1,
                color.next(),
                3,
            )

    # add original image on top
    if overlay:
        res[sparse_img > 0] = (100, 100, 100)
    return res
