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


facemask_classifier_path = r'models/face-mask-classifier'
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
    nparr = np.fromstring(r.data, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    det_class_result = get_classified_face_masks(
        image, face_detector, facemask_classifier)
    response_json = json.dumps(det_class_result, indent=4)

    return Response(response=response_json, status=200, mimetype="application/json")


@app.route('/', methods=['GET'])
def home():
    return "<h1>Facemask detection</h1><p>Detect faces, facemasks using API.</p>"


if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser("Flask app that server face mask detection from the image")
    parser.add_argument('--classifier_path', default=r'models/face-mask-classifier', type=str,
                        help='classification model path')
    parser.add_argument('--detector_path', default=r'models/face-detector', type=str,
                        help='detection model path')
    args = parser.parse_args()

    app.run()
