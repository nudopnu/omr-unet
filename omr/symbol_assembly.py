import numpy as np
import cv2
from PIL import ImageColor
from skimage.feature import peak_local_max

from .bbox_utils import intersection_over_union, within, intersection_over_union


def extract_noteglyphs(
    elements, stem_groups, staffline_groups, staff_gap, staff_thickness, n_blocks
):

    # visualize staffs
    notehead_to_stem = {}
    stem_to_beam = {}
    stem_to_staff = {}
    noteheads_black_elements = [e for e in elements if e["max_label"] == 8]
    noteheads_half_elements = [e for e in elements if e["max_label"] == 9]
    noteheads_elements = noteheads_half_elements + noteheads_black_elements
    beam_elements = [e for e in elements if e["max_label"] == 41]

    # prepare noteheads map
    noteheads_black = np.zeros((256, 256 * n_blocks), np.int32)
    noteheads_half = np.zeros((256, 256 * n_blocks), np.int32)

    for idx, stem_group in enumerate(stem_groups):

        for stem in stem_group:
            bbox_stem = stem["bbox"]

            # match heads with stems
            for head in noteheads_elements:
                bbox_head = head["bbox"]
                iou = intersection_over_union(bbox_stem, bbox_head)
                if iou > 0:
                    x0, y0, x1, y1 = bbox_head

                    if head["max_label"] == 8:
                        noteheads_black[head["mask"]] = stem["id"]
                    else:
                        noteheads_half[head["mask"]] = stem["id"]
                    notehead_to_stem[head["id"]] = stem["id"]

            # match beams with stems
            for beam in beam_elements:
                bbox_beam = beam["bbox"]
                iou = intersection_over_union(bbox_stem, bbox_beam)
                if iou > 0:
                    stem_to_beam[stem["id"]] = beam["id"]

            stem_to_staff[stem["id"]] = idx

    # extract heads from noteheads_map
    noteheads = (noteheads_half).astype(np.uint8)
    note_glyphs = {}

    # fill holes of half noteheads
    kernel_size = staff_gap
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    noteheads = cv2.morphologyEx(noteheads, cv2.MORPH_CLOSE, kernel)
    noteheads_half = noteheads.copy()

    # add black noteheads
    noteheads[noteheads_black > 0] = 1

    dist_transform = cv2.distanceTransform(noteheads, cv2.DIST_L2, 5)
    dist_transform = cv2.erode(
        dist_transform, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    )

    maxima = peak_local_max(
        (dist_transform > 2) * dist_transform * noteheads,
        min_distance=staff_gap - staff_thickness,
        threshold_abs=0.9,
    )
    for y, x in maxima:
        stem_label_black = noteheads_black[y, x]
        stem_label_half = noteheads_half[y, x]  # + np.max(stem_label_black)

        # add to noteglyphs dict
        if stem_label_black > 0:
            glyph_type = "black"
            if stem_label_black in note_glyphs:
                note_glyphs[stem_label_black]["heads"].append([x, y])
            else:
                note_glyphs[stem_label_black] = {
                    "type": glyph_type,
                    "heads": [[x, y]],
                }
        elif stem_label_half > 0:
            glyph_type = "half"
            if stem_label_half in note_glyphs:
                note_glyphs[stem_label_half]["heads"].append([x, y])
            else:
                note_glyphs[stem_label_half] = {
                    "type": glyph_type,
                    "heads": [[x, y]],
                }

    # infer pitches
    eps = 0.5 * staff_gap + staff_thickness

    for id in note_glyphs:
        glyph = note_glyphs[id]

        # init pitches found
        glyph["pitches"] = []

        # obtain corresponding stafflines
        if id not in stem_to_staff:
            print(f"ERROR: {id} not found!")
            continue
        staff_idx = stem_to_staff[id]
        staffline_ys = np.asarray(staffline_groups[staff_idx])[:, 1]
        min_y = np.min(staffline_ys)
        max_y = np.max(staffline_ys)

        for head in glyph["heads"]:
            x, y = head
            if y < min_y - (eps - 5):
                pitch = 9 + round((min_y - y) / eps)
            elif y > max_y + eps:
                pitch = 1 - round((y - max_y) / eps)
            else:
                diffs = np.abs(staffline_ys - y)
                diffs = diffs - np.min(diffs)
                if sum(diffs < eps) > 1:
                    pitch = 9 - np.sum(np.argwhere(diffs < eps))
                else:
                    pitch = 9 - np.argwhere(diffs < eps)[0, 0] * 2

            glyph["pitches"].append(pitch)

    return note_glyphs, stem_to_staff, stem_to_beam


def obtain_glyphs(elements, staffline_bboxes, note_glyphs, stem_to_staff, stem_to_beam):
    def noteglyp_toglyph(n):
        noteglyph = note_glyphs[n]
        res = {
            "type": noteglyph["type"],
            "staff": stem_to_staff[n],
            "x": np.min(np.asarray(noteglyph["heads"])[:, 0]),
            "pitches": noteglyph["pitches"],
            "beamgroup": -1,
        }
        if n in stem_to_beam:
            res["beamgroup"] = stem_to_beam[n]
        return res

    # glyphs = [{"type": note_glyphs[n]["type"], "staff": stem_to_staff[n], "x": np.min(np.asarray(note_glyphs[n]["heads"])[:, 0]), "pitches": note_glyphs[n]["pitches"]} for n in note_glyphs]#
    glyphs = [noteglyp_toglyph(n) for n in note_glyphs]  #

    # add additional glyphs

    monolithic_glyphs = {
        29: "RestWhole",
        30: "RestHalf",
        31: "RestQuarter",
        32: "Rest8th",
        4: "ClefG",
        5: "ClefF",
    }
    ids = [id for id in monolithic_glyphs]
    potential_glyphs = [e for e in elements if e["max_label"] in ids]

    already_processed = set()
    for element in potential_glyphs:
        bbox = element["bbox"]

        if element["id"] in already_processed:
            continue

        for idx, staff_bbox in enumerate(staffline_bboxes):
            if (
                within(staff_bbox, bbox)
                or intersection_over_union(staff_bbox, bbox, eps=(0, 0)) > 0
            ):
                type = monolithic_glyphs[element["max_label"]]
                x = bbox[0]
                glyphs.append({"type": type, "staff": idx, "x": x})
                already_processed.add(element["id"])

    return glyphs
