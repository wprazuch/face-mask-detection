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
                                             load_caffe_model, predict_facemask, detect_faces)
from flask import Flask, Response, request
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array


from flask_restful import Api, Resource

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


def load_models(facemask_classifier_path, face_detector_path):
    facemask_classifier = load_model(facemask_classifier_path)
    face_detector = load_caffe_model(face_detector_path)
    return facemask_classifier, face_detector


app = flask.Flask(__name__)
app.config["DEBUG"] = True


class Home(Resource):
    def get(self):
        return {"info": "Welcome to Facemask Detector app!"}


class FaceDetect(Resource):
    def __init__(self, **kwargs):
        self.cache = kwargs

    def get(self):
        return {"info": "FaceDetect works!"}

    def post(self):
        r = request
        nparr = np.fromstring(r.data, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        detections = detect_faces(image, self.cache['face_detector'])
        detections = detections[0, 0, ...].tolist()

        return {'result': detections}


class FacemaskClassify(Resource):

    def __init__(self, **kwargs):
        self.cache = kwargs

    def get(self):
        return "FacemaskClassify works!"

    def post(self):
        r = request
        nparr = np.fromstring(r.data, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        image = np.expand_dims(image, axis=0)

        classification_result, _, _ = predict_facemask(image, self.cache['facemask_classifier'])

        return {
            'result': classification_result
        }


class FacemaskDetect(Resource):

    def __init__(self, **kwargs):
        self.cache = kwargs

    def get(self):
        return "FacemaskDetect works!"

    def post(self):
        r = request
        nparr = np.fromstring(r.data, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        det_class_result = get_classified_face_masks(
            image, self.cache['face_detector'], self.cache['facemask_classifier'])

        return {
            'result': det_class_result
        }


def start_server():

    app = Flask(__name__)
    api = Api(app)

    # Load models
    facemask_classifier_path = r'models/face-mask-classifier'
    face_detector_path = r'models/face-detector'

    facemask_classifier, face_detector = load_models(facemask_classifier_path, face_detector_path)

    api.add_resource(Home, '/')
    api.add_resource(FaceDetect, '/facedetect', resource_class_kwargs={
                     'face_detector': face_detector})
    api.add_resource(FacemaskClassify, '/facemaskclassify', resource_class_kwargs={
        'facemask_classifier': facemask_classifier})
    api.add_resource(FacemaskDetect, '/facemaskdetect', resource_class_kwargs={
        'face_detector': face_detector,
        'facemask_classifier': facemask_classifier})

    app.run(debug=True)

    return app


if __name__ == '__main__':

    #app.run(debug=True, host='0.0.0.0')

    start_server()
