import flask
from flask import Flask, request, Response
import jsonpickle
import numpy as np
import cv2

import numpy as np
import tensorflow as tf

import os

from tensorflow.keras.models import load_model
import cv2

from . import utils
from . import model_utils

from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array

import matplotlib.pyplot as plt
import json

app = flask.Flask(__name__)
app.config["DEBUG"] = True


@app.route('/api/image', methods=['POST'])
def test():
    r = request
    # convert string of image data to uint8
    nparr = np.fromstring(r.data, np.uint8)
    # decode image
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # do some fancy processing here....
    # test_image = r'dataset/test_detection/4.jpg'
    # image = cv2.imread(test_image)

    # build a response dict to send back to client
    det_class_result = model_utils.get_classified_face_masks_rest(
        image, r'models/face-detector', r'models/facemask-classifier')
    # encode response using jsonpickle
    response_json = json.dumps(det_class_result, indent=4)

    return Response(response=response_json, status=200, mimetype="application/json")


@app.route('/', methods=['GET'])
def home():
    return "<h1>Distant Reading Archive</h1><p>This site is a prototype API for distant reading of science fiction novels.</p>"


app.run()
