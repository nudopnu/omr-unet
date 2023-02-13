import numpy as np
from PIL import ImageColor
import cv2
from .bbox_utils import area, from_mask, intersection_over_union

COLORINGS = {
    "#ffffff": [46, 2],  # staff and ledger line
    "#6B238F": [41, 42, 40],  # beams and full rests
    "#FF7F00": [8],  # quarter noteheads
    "#8F2323": [9],  # half noteheads
    "#0040FF": [10],  # full noteheads
    "#AA00FF": [29, 30],
    "#00EAFF": [28, 23],  # sharps
    "#556b2f": [26, 21],  # flats
    "#FF00FF": [27, 22],  # naturals
    "#4F8F23": [36],  # piano
    "#23628F": [4],  # treble clef
    "#dc143c": [5],  # bass clef
    "#FF0000": [13, 14, 15, 16, 17, 18, 19, 20],  # flag
    "#FF0000": [12],  # stem and bar lines
    "#00ff00": [31],  # quarter rests
    "#ee82ee": [32],  # 8th rests
    "#66cdaa": [33],  # 16th rests
    "#696969": [11],  # aug dot
    "#0000ff": [24],  # double sharp
    "#008080": [37],  # DynamicM
    "#d8bfd8": [6, 7],  # TimeSig4, TimeSigCommon
    "#191970": [17] # 8th-flag down
}
COLORINGS_INV = {label: color for color in COLORINGS for label in COLORINGS[color]}


def getprob(prediction, label, n_blocks):
    label_prob = prediction[..., label]
    res = np.zeros((256, n_blocks * 256), prediction.dtype)
    for i in range(n_blocks):
        res[:, i * 256 : (i + 1) * 256] = label_prob[i]
    return res


def getrangeprob(prediction, labels, n_blocks):
    res = np.zeros((256, n_blocks * 256), prediction.dtype)

    for label in labels:
        label_prob = prediction[..., label]
        for i in range(n_blocks):
            res[:, i * 256 : (i + 1) * 256] += label_prob[i]
    return res


def getmax(prediction, n_blocks, label=None):
    res = np.zeros((256, n_blocks * 256), bool)
    if label == None:
        res = np.zeros((256, n_blocks * 256), np.uint8)
    for i in range(n_blocks):
        tmp = np.argmax(prediction[i], axis=2)
        if label != None:
            tmp = tmp == label
        res[:, i * 256 : (i + 1) * 256] = tmp
    return res


def predict(model, img, n_blocks):
    input = np.zeros((n_blocks, 256, 256, 1))
    for i in range(n_blocks):
        input[i] = 255 - img[:, i * 256 : (i + 1) * 256, None]
    return model.predict(input, verbose=0)


def detect_symbols(
    prediction,
    src_img,
    n_blocks,
    staff_gap,
    post_process=lambda img: img,
    k=3,
    surpress=[],
    warn=True,
):

    # get horizontal lines
    staff = getmax(prediction, n_blocks, 46)
    ledger = getmax(prediction, n_blocks, 2)

    # get all vertical lines
    vlines = getrangeprob(prediction, [12, 47, 48], n_blocks)
    vline_elements = []
    elem_id = 1

    # threshold the vlines probability
    vlines_thresh = vlines * 2
    vlines_thresh = cv2.morphologyEx(
        vlines_thresh, cv2.MORPH_CLOSE, np.ones((staff_gap, 1), np.uint8)
    )
    vlines_thresh = vlines_thresh.astype(np.uint8)

    # filter all components that are not long enough
    (nLabels, labels, stats, centroids) = cv2.connectedComponentsWithStats(
        vlines_thresh, 2, cv2.CV_32S
    )
    for label in range(1, nLabels):
        h = stats[label, cv2.CC_STAT_HEIGHT]

        # if too short -> noise
        if h < 1.5 * staff_gap:
            labels[labels == label] = 0

        w = stats[label, cv2.CC_STAT_WIDTH]
        x = stats[label, cv2.CC_STAT_LEFT]
        y = stats[label, cv2.CC_STAT_TOP]
        bbox = [x, y, x + w, y + h]

        vline_elements.append(
            {
                "id": elem_id,
                "bbox": bbox,
                "max_label": label,
                "top_k": [],
                "mask": labels == label,
            }
        )
        elem_id += 1

    vlines = labels.copy()
    surpress = surpress + [12, 47, 48]

    # get vertical lines
    stems = getmax(prediction, n_blocks, 12)
    barlines = getmax(prediction, n_blocks, 47)
    barlines_thick = getmax(prediction, n_blocks, 48)
    erease = [staff, stems, ledger, stems, barlines, barlines_thick]
    mask = erease[0] == 0
    for e in erease:
        mask *= e == 0
    remainder = getmax(prediction, n_blocks) * mask
    remainder = post_process(remainder)

    n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(remainder, 2)

    # analyze connected components
    res = np.ones((256, n_blocks * 256, 3), np.uint8) * 255
    res[src_img <= 150] = (0, 0, 0)
    missing_labels = []
    elements = []
    for i in range(1, n_labels):
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        if w <= 3 or h <= 3:
            labels[labels == i] = 0
            continue
        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]

        bbox = [x, y, x + w, y + h]

        # get max index in area and obtain color
        masked_labels = (labels == i) * remainder
        masked_labels = masked_labels[masked_labels != 0].reshape(-1)
        label_counts = np.bincount(masked_labels)

        # surpression
        for idx in surpress:
            if idx in masked_labels:
                label_counts[idx] = 0

        label_percentages = label_counts / len(masked_labels)
        max_label = label_counts.argmax()

        # get top k
        top_k_labels = [max_label for i in range(3)]
        if len(label_counts) >= k:
            top_k_partitions = np.argpartition(label_counts, -k)
            top_k_labels = top_k_partitions[-k:]
        top_k = [{"label": l, "percentage": label_percentages[l]} for l in top_k_labels]

        if max_label not in COLORINGS_INV:
            missing_labels.append(max_label)
            continue

        elements.append(
            {
                "id": elem_id,
                "bbox": bbox,
                "max_label": max_label,
                "top_k": top_k,
                "mask": labels == i,
            }
        )
        elem_id += 1

    if warn:
        print("Missing labels:", missing_labels)
    return vlines, vline_elements, elements, remainder


def visualize(img, elements):
    res = img.copy()
    for element in elements:
        max_label = element["max_label"]
        x0, y0, x1, y1 = element["bbox"]
        label_color = ImageColor.getcolor(COLORINGS_INV[max_label], "RGB")
        res = cv2.rectangle(res, (x0, y0), (x1, y1), label_color, 3)
    return res


def non_maximum_surpression(elements, ignore=[], only_self=[], prevent=[], eps=(1, 2)):
    result_elements = [e for e in elements if e["max_label"] in ignore]
    elements = [e for e in elements if e["max_label"] not in ignore]
    area_sorted = sorted(elements, key=lambda x: area(x["bbox"]))
    processed_ids = set()
    for element in area_sorted[::-1]:

        for element_b in area_sorted:
            already_removed = element_b["id"] in processed_ids
            same_element = element["id"] == element_b["id"]
            a_label = element["max_label"]
            b_label = element_b["max_label"]
            not_reflexive = a_label in only_self and a_label != b_label
            to_be_avoided = (a_label, b_label) in prevent
            if already_removed or same_element or not_reflexive or to_be_avoided:
                continue

            iou = intersection_over_union(element["bbox"], element_b["bbox"], eps=eps)
            if iou > 0:
                element = dict(element)
                element["mask"] = element["mask"] | element_b["mask"]
                element["bbox"] = from_mask(element["mask"])
                processed_ids.add(element_b["id"])
                processed_ids.add(element["id"])
                result_elements.append(element)

        if element["id"] not in processed_ids:
            processed_ids.add(element["id"])
            result_elements.append(element)

    return result_elements


def assign_double_dots(elements, staff_gap, staff_thickness, eps=3):

    # assign double dots
    resulting_elements = [e for e in elements if e["max_label"] not in [5, 3, 11]]
    dots = [e for e in elements if e["max_label"] in [5, 3, 11]]

    double_dots = []
    eps = 3
    already_processed = set()
    for a in dots:
        if a["id"] in already_processed:
            continue
        for b in dots:
            if a["id"] == b["id"] or b["id"] in already_processed:
                continue
            ax0, ay0, ax1, ay1 = a["bbox"]
            bx0, by0, bx1, by1 = b["bbox"]
            if abs(ax0 - bx0) <= eps and abs(ax1 - bx1) <= eps:
                if abs(ay0 - by0) < staff_gap + staff_thickness + eps:
                    double_dots.append([a, b])
                    already_processed.add(a["id"])
                    already_processed.add(b["id"])

    # ignore single dots
    for dot in dots:
        if dot["id"] not in already_processed:
            resulting_elements.append(dot)

    for (a, b) in double_dots:
        left = min(a["bbox"][0], b["bbox"][0])
        candidates = [
            e for e in elements if e["bbox"][2] < left and left - e["bbox"][2] < eps
        ]
        processed = False
        for candidate in candidates:
            if 5 in [k["label"] for k in candidate["top_k"]]:
                candidate["mask"] |= a["mask"]
                candidate["mask"] |= b["mask"]
                candidate["max_label"] = 5
                candidate["bbox"] = from_mask(candidate["mask"])
                processed = True
                break
        if not processed:
            resulting_elements.append(a)
            resulting_elements.append(b)

    return resulting_elements


def extract_stafflines(prediction, staff_gap, n_blocks):
    # get horizontal lines
    staff = getprob(prediction, 46, n_blocks) > 0.9
    staff = getmax(prediction, n_blocks, 46)

    # threshold horizontal projections of staff lines
    horizontal_projection = np.sum(staff, axis=1)
    hp_thresh = horizontal_projection[horizontal_projection > 0]
    thresh = np.mean(hp_thresh)

    # obtain stafflines by merging
    y_positions = np.argwhere(horizontal_projection > thresh)
    staff_lines = []
    last_y = -2
    for y in y_positions:

        if y - last_y == 1:
            staff_lines[-1][-1] += 1
            last_y = y
            continue

        # create staffline
        y0 = y[0]
        idcs = np.indices((staff.shape[1],))[staff[y] > 0]
        x0 = np.min(idcs)
        x1 = np.max(idcs)
        y1 = y0 + 1
        staff_line = [x0, y0, x1, y1]
        staff_lines.append(staff_line)

        last_y = y

    # group stafflines to staffs
    staffline_groups = []
    for line in staff_lines:
        if len(staffline_groups) == 0:
            staffline_groups.append([line])
            continue
        last_staffline_group = staffline_groups[-1]
        last_staffline = last_staffline_group[-1]
        if line[1] - (last_staffline[-1]) < 2 * staff_gap:
            last_staffline_group.append(line)
        else:
            staffline_groups.append([line])

    # filter non 5-liners
    staffline_groups = [g for g in staffline_groups if len(g) == 5]

    # extract bounding boxes
    staffline_bboxes = []
    for group in staffline_groups:
        group = np.asarray(group)
        x0 = np.min(group[..., 0])
        x1 = np.max(group[..., 2])
        y0 = np.min(group[..., 1])
        y1 = np.max(group[..., 3])
        staffline_bbox = [x0, y0, x1, y1]
        staffline_bboxes.append(staffline_bbox)

    return staffline_groups, staffline_bboxes


def extract_stems_and_barlines(staffline_bboxes, vline_elements):
    barlines = []
    stem_groups = [[] for _ in staffline_bboxes]
    for vle in vline_elements:
        bbox = vle["bbox"]
        intersections = []

        for idx, staff_bbox in enumerate(staffline_bboxes):
            iou = intersection_over_union(bbox, staff_bbox)
            if iou > 0:
                intersections.append(idx)

        if len(intersections) > 1:
            barlines.append(vle)
            continue

        if len(intersections) == 0:
            continue

        intersection = intersections[0]
        stem_groups[intersection].append(vle)

    return barlines, stem_groups
