
import numpy as np
import tensorflow as tf

import os

from tensorflow.keras.models import load_model
import cv2

from face_mask_detection import utils

from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


def get_classified_face_masks_rest(image, face_detector_path, facemask_classifier_path):
    facemask_classifier = load_model(facemask_classifier_path)
    face_detector = load_caffe_model(face_detector_path)
    image_detection_classification_results = get_classified_face_masks(
        image, face_detector, facemask_classifier)
    return image_detection_classification_results


def detect_faces(image, face_detector):
    """Detects faces in the image

    Parameters
    ----------
    image : [type]
        [description]
    face_detector : [type]
        [description]

    Returns
    -------
    [type]
        [description]
    """
    blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300), (104.0, 177.0, 123.0))
    face_detector.setInput(blob)
    detections = face_detector.forward()
    return detections


def extract_face(image, detection):
    """Extracts face from the image

    Parameters
    ----------
    image : [type]
        [description]
    detection : [type]
        [description]

    Returns
    -------
    np.ndarray, dict
        Returns cropped image with the face, together with bounding box cords of it in the image
    """
    (h, w) = image.shape[:2]
    box = detection * np.array([w, h, w, h])
    (startX, startY, endX, endY) = box.astype("int")

    (start_x, start_y) = (max(0, startX), max(0, startY))
    (end_x, end_y) = (min(w - 1, endX), min(h - 1, endY))

    bounding_box_cords = {
        'start_x': int(start_x),
        'start_y': int(start_y),
        'end_x': int(end_x),
        'end_y': int(end_y)
    }

    face = image[startY:endY, startX:endX]
    face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
    face = cv2.resize(face, (224, 224))
    face = img_to_array(face)
    face = preprocess_input(face)
    face = np.expand_dims(face, axis=0)

    return face, bounding_box_cords


def get_classified_face_masks(image, face_detector, facemask_classifier):
    """Searches for faces in the image and returns class of the face,
    whether it does have a mask on it or not

    Parameters
    ----------
    image : [type]
        [description]
    face_detector : [type]
        [description]
    facemask_classifier : [type]
        [description]

    Returns
    -------
    list
        list of dicts with detection-classification metadata of every face in the image
    """

    detections = detect_faces(image, face_detector)

    image_detection_classification_results = []

    for i in range(0, detections.shape[2]):

        confidence = detections[0, 0, i, 2]

        if confidence > 0.5:
            detection = detections[0, 0, i, 3:7]
            face, detection_info = extract_face(image, detection)

            (mask, withoutMask) = facemask_classifier.predict(face)[0]

            label = "Mask" if mask > withoutMask else "No Mask"

            detection_info['class'] = label

            detection_info['probability'] = max(mask, withoutMask) * 100

            image_detection_classification_results.append(detection_info)

    return image_detection_classification_results


def load_caffe_model(caffe_model_path):
    """Loads model in caffe format from the path

    Parameters
    ----------
    caffe_model_path : [type]
        [description]

    Returns
    -------
    [type]
        [description]
    """
    prototxt_file = [file for file in os.listdir(caffe_model_path) if file.endswith('.prototxt')][0]
    caffe_model_file = [file for file in os.listdir(
        caffe_model_path) if file.endswith('.caffemodel')][0]
    model = cv2.dnn.readNet(os.path.join(caffe_model_path, prototxt_file),
                            os.path.join(caffe_model_path, caffe_model_file))
    return model
