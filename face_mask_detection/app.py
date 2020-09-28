import json
import matplotlib.pyplot as plt
from . import model_utils
from . import utils
import flask
from flask import Flask, request, Response
import jsonpickle
import numpy as np
import cv2

import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model

import numpy as np
from .model_utils import get_classified_face_masks, load_caffe_model
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


facemask_classifier_path = r'models/facemask-classifier'
face_detector_path = r'models/face-detector'


def load_models():
    facemask_classifier = load_model(facemask_classifier_path)
    face_detector = load_caffe_model(face_detector_path)
    return facemask_classifier, face_detector


facemask_classifier, face_detector = load_models()

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
    print("Before function")
    det_class_result = get_classified_face_masks(
        image, face_detector, facemask_classifier)
    print("After function")
    # encode response using jsonpickle
    response_json = json.dumps(det_class_result, indent=4)

    return Response(response=response_json, status=200, mimetype="application/json")


@app.route('/', methods=['GET'])
def home():
    return "<h1>Distant Reading Archive</h1><p>This site is a prototype API for distant reading of science fiction novels.</p>"


if __name__ == '__main__':

    app.run()
