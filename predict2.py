import keras
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
from omr.system_recognition import get_coarse_masks, extract_scaled_systems
from omr.symbol_detection import (
    detect_symbols,
    non_maximum_surpression,
    visualize,
    assign_double_dots,
    getrangeprob,
    predict,
    extract_stafflines,
    extract_stems_and_barlines,
)
from omr.staffline_detection import detect_staffs
from omr.morphology import (
    sharpen,
    dense,
    sparse,
    chain,
    single_channel,
    dilation,
    closing,
    opening,
    threshold,
    apply,
)
from omr.symbol_assembly import extract_noteglyphs, obtain_glyphs
from omr.commands import command_loop, imgDecoding

glob = {}


def loadModel(model_file):
    model = keras.models.load_model(os.path.join("..", "..", "models", model_file))
    glob["model"] = model
    return f"Loaded model '{model_file}'"


def predict64(img64):

    # Preprocessing
    img = apply(img64)(
        imgDecoding(),
        dense(),
    )

    src = apply(img)(
        sharpen(),
        single_channel(),
        sparse(),
        dilation((13, 1)),
    )

    # Extract Systems
    max_area_masks, max_area_contours = get_coarse_masks(src, thresh=20)
    systems, mappings, system_bounding_boxes = extract_scaled_systems(
        img, max_area_masks
    )

    # Detect Staffs
    (
        staff_areas,
        staff_bounding_boxes,
        staff_gap,
        staff_thickness,
        curvature,
    ) = detect_staffs(img)

    prevent = [
        (9, 5),
        (9, 3),
        (9, 11),
        (8, 5),
        (8, 3),
        (8, 11),
    ]

    # do omr
    res = []

    for idx, system in enumerate(systems):

        res_system = []
        n_blocks = system.shape[1] // 256

        pp = lambda img: img
        system = sharpen()(system)

        prediction = predict(glob["model"], system, n_blocks)
        vlines, vline_elements, elements, remainder = detect_symbols(
            prediction,
            system,
            n_blocks,
            staff_gap,
            post_process=pp,
            surpress=[10, 40, 42],
            warn=False,
        )
        elements = non_maximum_surpression(
            elements,
            ignore=[8, 11, 21, 22, 26, 41],
            only_self=[23, 24, 25, 27],
            prevent=prevent,
            eps=(1, 2),
        )
        elements = assign_double_dots(elements, staff_gap, staff_thickness, eps=4)

        staffline_groups, staffline_bboxes = extract_stafflines(
            prediction, staff_gap, n_blocks
        )
        barlines, stem_groups = extract_stems_and_barlines(
            staffline_bboxes, vline_elements
        )

        note_glyphs, stem_to_staff, stem_to_beam = extract_noteglyphs(
            elements,
            stem_groups,
            staffline_groups,
            staff_gap,
            staff_thickness,
            n_blocks,
        )
        glyphs = obtain_glyphs(
            elements, staffline_bboxes, note_glyphs, stem_to_staff, stem_to_beam
        )

        # prepare output
        for staff_idx in range(len(staffline_bboxes)):
            staff_glyphs = sorted(
                [g for g in glyphs if g["staff"] == staff_idx], key=lambda x: x["x"]
            )
            res_system.append(staff_glyphs)
        res.append(res_system)

    return res


commands = {
    "loadModel": loadModel,
    "predict": predict64,
    "ping": lambda x: x,
}

command_loop(commands)
# test:
# cat b64.txt | py scripts/predict.py simple_unet_256x256_03.h5 > tmp.txt
