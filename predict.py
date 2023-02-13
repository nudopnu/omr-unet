
import os
import numpy as np
import cv2
import tensorflow as tf
from omr.morphology import apply, sparse, tile_merge_flat
from omr.commands import command_loop, imgDecoding, imgEncoding

glob = {}

def loadModel(model_file):
    model = tf.keras.models.load_model(os.path.join("..", "..", "models", model_file))
    glob["model"] = model
    return f"Loaded model '{model_file}'"

def predict(img64):
    return apply(img64)(
        imgDecoding(),
        lambda img: cv2.cvtColor(img, cv2.COLOR_RGBA2GRAY),
        lambda img: img[:, :, None],
        sparse(),
        tile_merge_flat((256, 256), _predict_tile),
        imgEncoding(),
    )

def _predict_tile(tile):
    res = glob["model"].predict(tile[None, :], verbose = 0)
    res = np.argmax(res[0], 2)
    return res

commands = {
    "loadModel": loadModel,
    "predict": predict,
    "ping": lambda x: x,
}

command_loop(commands)
# test: 
# cat b64.txt | py scripts/predict.py simple_unet_256x256_03.h5 > tmp.txt