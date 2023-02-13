import base64
import json

import cv2
import numpy as np

def receive():
    msg = input()
    msg = json.loads(msg)
    type = msg["type"]
    payload = msg["payload"]
    return type, payload

def reply(type, payload, to_str=True):
    if to_str:
        payload = str(payload)
    print(json.dumps({"type": type, "payload": payload}))
    
def command_loop(commands, condition=lambda: True):
    ERROR = "error"
    MESSAGE = "message"

    while condition():
        type, payload = receive()
        if type not in commands:
            reply(ERROR, "command not found")
        else:
            result = commands[type](payload)
            if result is not None:
                reply(MESSAGE, result)

def imgEncoding():
    return _encodeImg

def imgDecoding():
    return _decodeImg

def _decodeImg(img64):
    # Convert from base64 to img
    imgBytes = base64.b64decode(img64, altchars=None, validate=False)
    img = np.frombuffer(imgBytes, dtype=np.uint8)
    return cv2.imdecode(img, cv2.IMREAD_ANYCOLOR)

def _encodeImg(img):
    # Convert result to base64
    bytes = cv2.imencode(".png", img)[1]
    return base64.b64encode(bytes)
