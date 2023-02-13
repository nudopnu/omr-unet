import numpy as np
import cv2
from pythonRLSA import rlsa
from pythonRLSA.rlsa_fast import rlsa_fast


def __parse_size_param(size):
    if type(size) is tuple or type(size) == list:
        if len(size) == 1:
            size = size[0]
            return (size, size)
        if len(size) == 2:
            return size
    return (size, size)


def negation():
    def process(img):
        if img.dtype == np.uint8:
            return 255 - img
        return 1 - img

    return process


def sparse():
    def process(img):
        if np.sum(img > 0) > np.sum(img == 0):
            return negation()(img)
        return img

    return process


def dense():
    def process(img):
        if np.sum(img > 0) < np.sum(img == 0):
            return negation()(img)
        return img

    return process


def single_channel():
    def process(img):
        if len(img.shape) <= 2:
            return img
        else:
            return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    return process


def three_channels():
    def process(img):
        if len(img.shape) <= 2:
            return np.repeat(img[:, :, None], 3, 2)
        if img.shape[2] == 1:
            return np.repeat(img, 3, 2)
        return img

    return process


def threshold(t=lambda img: 0.5, asUint=True):
    def process(img):
        th = t(img)
        res = img > th
        if asUint:
            res = res.astype(np.uint8) * 255
        return res

    return process


def normalize(asUint=True):
    def process(img):
        res = img / np.max(img)
        if asUint:
            res = res.astype(np.uint8) * 255
        return res

    return process


def binary(threshold=0):
    def process(img):
        return img > threshold

    return process


def sharpen():
    def process(img):
        kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        return cv2.filter2D(img, -1, kernel)

    return process


def dilation(size=3, kernel="None"):
    sx, sy = __parse_size_param(size)

    def process(img):
        k = np.ones((sy, sx), img.dtype) if kernel == "None" else kernel
        return cv2.dilate(img, k)

    return process


def edge_detection(edge="top"):
    supported = ["top", "bottom", "left", "right"]
    if edge not in supported:
        raise Exception(
            '[edge-detection] Edge type "{}" not supported. Use one of {}.'.format(
                edge, supported
            )
        )

    def process(img):
        k = np.array(
            [
                [
                    -1,
                    -1,
                    -1,
                ],
                [
                    0,
                    0,
                    0,
                ],
                [
                    1,
                    1,
                    1,
                ],
            ]
        )
        if edge in supported[2:]:
            k = k.T
        if edge in ["bottom", "right"]:
            k = k * -1
        return cv2.filter2D(img, -1, k)

    return process


def erosion(size=3, kernel="None"):
    sx, sy = __parse_size_param(size)

    def process(img):
        k = np.ones((sy, sx), img.dtype) if kernel == "None" else kernel
        return cv2.erode(img, k)

    return process


def gaussian(sigma=3):
    sx, sy = __parse_size_param(sigma)

    def process(img):
        return cv2.GaussianBlur(img, (sx, sy), sx, sy)

    return process


def horizontal_dilation(size=(3, 5)):
    sx, sy = __parse_size_param(size)

    def process(img):
        kernel = np.ones((sy, sx), img.dtype)
        dil_a = cv2.dilate(img, kernel, anchor=(sx - 1, 0))
        dil_b = cv2.dilate(img, kernel, anchor=(0, 0))
        dil_res = (dil_a > 0) * (dil_b > 0)
        return dil_res.astype(img.dtype)

    return process


def closing(size=(3, 3)):
    sx, sy = __parse_size_param(size)

    def process(img):
        kernel = np.ones((sy, sx), img.dtype)
        return cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)

    return process


def opening(size=(3, 3)):
    sx, sy = __parse_size_param(size)

    def process(img):
        kernel = np.ones((sy, sx), img.dtype)
        return cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)

    return process


def run_length_smoothing(size, horizontal=True, vertical=True, fast=True):
    func = rlsa_fast if fast else rlsa

    def process(img):
        img = img.copy()
        return func(img, horizontal, vertical, size)

    return process


def chain(*morphologies):
    def process(img):
        # img = img.copy()
        for morphology in morphologies:
            img = morphology(img)
        return img

    return process


# basically reverse approach to chain
def apply(input):
    def process(*morphologies):
        return chain(*morphologies)(input)

    return process


def split_merge(merge="or", *morphologies):
    supported_operations = {
        "and": cv2.bitwise_and,
        "or": cv2.bitwise_or,
        "xor": cv2.bitwise_xor,
    }
    if merge not in supported_operations:
        raise Exception(
            '[split-merge] Edge type "{}" not supported. Use one of {}.'.format(
                merge, [k for k in supported_operations]
            )
        )
    merge_operation = supported_operations[merge]

    def process(img):
        results = [morphology(img) for morphology in morphologies]
        final_result = results[0]
        if len(results) == 1:
            return final_result
        for result in results[1:]:
            final_result = merge_operation(final_result, result)
        return final_result

    return process


# ----- Util functions


def getAreaAttributes(stats, centroids, index):
    x = stats[index, cv2.CC_STAT_LEFT]
    y = stats[index, cv2.CC_STAT_TOP]
    w = stats[index, cv2.CC_STAT_WIDTH]
    h = stats[index, cv2.CC_STAT_HEIGHT]
    area = stats[index, cv2.CC_STAT_AREA]
    (cx, cy) = centroids[index]
    return x, y, w, h, area, cx, cy


def getLocalExtrema(signal):
    minima = []
    maxima = []
    for i in range(len(signal) - 2):
        cur = signal[i + 1]
        left = signal[i]
        right = signal[i + 2]
        if cur > left and cur >= right:
            maxima.append(i + 1)
        if cur < left and cur <= right:
            minima.append(i + 1)
    return np.array(minima), np.array(maxima)


def tile_merge(size, operation=lambda tiles: tiles):
    def process(img):
        tiles = to_chunks(size)(img)
        intermediate_shape = tiles.shape
        tiles = operation(np.array([tile for col in tiles for tile in col]))
        tiles = tiles.reshape(intermediate_shape)
        res = stitch()(tiles)
        return res

    return process


def tile_merge_flat(size, operation=lambda tile: tile):
    def process(img):
        tiles = to_chunks(size)(img)
        intermediate_shape = tiles.shape
        tiles = np.array([operation(tile) for col in tiles for tile in col])
        tiles = tiles.reshape(intermediate_shape)
        return stitch()(tiles)

    return process