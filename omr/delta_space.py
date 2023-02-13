import numpy as np


def analyze(sheet, min_delta=3, max_delta=200):
    img_w = sheet.shape[1]

    # create parameter space (gap-space)
    delta_y_space = np.zeros((max_delta, img_w), np.int32)

    # temporary per-column info
    last_y = np.zeros(img_w, np.int32)
    cur_start_y = np.zeros(img_w, np.int32)
    last_gap = np.zeros(img_w, np.int32)
    cur_thickness_y = np.zeros(img_w, np.int32)

    # gap_dict:
    #  maps from any valid gap to all points which
    #  are adjacent to that gap
    gap_dict = {}
    point_line_dict = {}

    # iterate over all white points
    ys, xs = np.where(sheet >= 0.5)
    for y, x in zip(ys, xs):

        gap = y - last_y[x]
        last_y[x] = y

        # no gap, last line gets thicker
        if gap == 0:
            cur_thickness_y[x] += 1
            continue

        # gap too small or too big -> reset line
        if gap >= max_delta or gap <= min_delta:
            cur_thickness_y[x] = 1
            last_gap[x] = gap
            continue

        # contribute to delta-space
        delta_y_space[gap, x] += 1

        # start point is the white point (with thickness), right before
        # a valid gap (i.e. a gap within delta limits) begins.
        # -> store start_point: (start_y, x, thickness, last_gap)
        start_point = (cur_start_y[x], x, cur_thickness_y[x], last_gap[x])
        point_line_dict["{}_{}".format(cur_start_y[x], x)] = (last_gap[x], gap)

        if last_gap[x] not in gap_dict:
            gap_dict[last_gap[x]] = [start_point]
        else:
            gap_dict[last_gap[x]].append(start_point)
        if gap not in gap_dict:
            gap_dict[gap] = [start_point]
        else:
            gap_dict[gap].append(start_point)

        # update current line beginning
        last_gap[x] = gap
        cur_start_y[x] = y
        cur_thickness_y[x] = 1

    gap_votes = np.sum(delta_y_space, 1)
    staff_gap = np.argmax(gap_votes)
    staff_thickness = np.argmax(
        np.bincount(
            np.squeeze([gap_dict[k] for k in gap_dict if k in [staff_gap]])[:, 2]
        )
    )

    return delta_y_space, gap_dict, staff_gap, staff_thickness


def get_staff_line_points(delta_y_space, gap_dict, shape, k=2):
    canvas = np.zeros(np.asarray(shape)[:2])
    top_k = np.argpartition(np.sum(delta_y_space, 1), -k)[-k:]

    staff_gap = np.argmax(np.sum(delta_y_space, 1))
    thicknesses = np.squeeze([gap_dict[k] for k in gap_dict if k in [staff_gap]])[:, 2]
    staff_thickness = np.argmax(np.bincount(thicknesses))

    points = []
    for delta in top_k:
        for y, x, t, _ in gap_dict[delta]:
            if t > staff_thickness or y == 0:
                continue
            canvas[y : y + t, x] += 1
            points.append((y, x, t))

    return canvas, points


def get_curvature_from_points(points, staff_gap, max_bounds):
    y_max, x_max = max_bounds
    canvas = np.zeros((y_max, x_max))
    deltas = np.zeros(x_max, np.float64)
    for y, x, t in points:
        canvas[y, x] += 1
    canvas = canvas > 0
    for x in range(x_max - 1):
        half_gap = int(np.round(0.5 * staff_gap))
        max_delta = 0
        for i in range(half_gap):
            num_points = np.sum(canvas[i:, x])
            if num_points <= 0:
                num_points = 1
            pos_delta = np.sum(canvas[i:, x] * canvas[: y_max - i, x + 1]) / num_points
            neg_delta = np.sum(canvas[: y_max - i, x] * canvas[i:, x + 1]) / num_points
            cur_max = max(pos_delta, neg_delta)
            sign = 1
            if pos_delta <= neg_delta:
                sign = -1
            if pos_delta == neg_delta:
                cur_max = 0
            if abs(max_delta) <= cur_max:
                max_delta = sign * cur_max
        deltas[x] = max_delta
    curvature = np.add.accumulate(deltas)
    if np.max(curvature) - np.min(curvature) < 150:
        curvature *= 0
    return canvas, curvature


def undistort(img, distortion, min_to_zero=True):
    img_h, img_w = img.shape[0], img.shape[1]
    res = np.zeros_like(img)

    distortion = -np.round(distortion).astype(np.int32)
    min_distortion = np.min(np.unique(distortion)) if min_to_zero else 0
    distortion = distortion - min_distortion

    for x, y in enumerate(distortion):
        res[: img_h - y, x] = img[y:, x]

    return min_distortion, res
