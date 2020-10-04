import json
import os

import cv2
import flask
import jsonpickle
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from face_mask_detection import model_utils, utils
from face_mask_detection.model_utils import (get_classified_face_masks,
                                             load_caffe_model)
from flask import Flask, Response, request
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

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
