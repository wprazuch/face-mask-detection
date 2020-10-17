from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.applications.mobilenet import preprocess_input
from tensorflow.image import resize
from sklearn.preprocessing import OneHotEncoder
import numpy as np
import os
import cv2


def prepare_dataset(dataset_path, input_shape, class_mapping):
    """Prepares dataset for training facemask classification network

    Parameters
    ----------
    dataset_path : [type]
        [description]
    input_shape : [type]
        [description]
    class_mapping : [type]
        [description]

    Returns
    -------
    [type]
        [description]
    """

    dir_classes = os.listdir(dataset_path)
    data = []
    labels = []
    exts = ('.png', '.jpg', '.jpeg')

    for class_directory in dir_classes:
        dir_path = os.path.join(dataset_path, class_directory)
        imgs_filenames = [filename for filename in os.listdir(dir_path) if filename.endswith(exts)]
        imgs_full_paths = [os.path.join(dir_path, img_filename) for img_filename in imgs_filenames]
        for img_path in imgs_full_paths:
            img = img_to_array(load_img(img_path))
            img = preprocess_image(img, input_shape)
            data.append(img)
            labels.append(class_mapping[class_directory])

    return data, labels


def preprocess_image(image, input_shape):
    """Preprocesses image for face-mask classification

    Parameters
    ----------
    image : [type]
        [description]
    input_shape : [type]
        [description]

    Returns
    -------
    [type]
        [description]
    """
    image = preprocess_input(image)
    image = resize(image, input_shape)
    return image


def to_one_hot(labels):
    """Converts numpy array with class labels to one-hot-encoding

    Parameters
    ----------
    labels : np.array (n_observations,)
        class labels

    Returns
    -------
    np.ndarray (n_observations, n_classes)
        [description]
    """
    onehot = OneHotEncoder()
    labels = np.array(labels).reshape(-1, 1)
    labels_one_hot = onehot.fit_transform(labels).toarray()

    return labels_one_hot


def visualize_detection_classification_results(image, detection_classification_results):
    """Returns image with bounding boxes around faces annotated with class labels

    Parameters
    ----------
    image : [type]
        [description]
    detection_classification_results : [type]
        [description]

    Returns
    -------
    [type]
        [description]
    """

    for result in detection_classification_results:

        label = result['class']

        color = (0, 255, 0) if label == "Mask" else (0, 0, 255)

        # include the probability in the label
        label = "{}: {:.2f}%".format(label, result['probability'])

        # display the label and bounding box rectangle on the output
        # frame
        cv2.putText(image, label, (result['start_x'], result['start_y'] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        cv2.rectangle(image, (result['start_x'], result['start_y']),
                      (result['end_x'], result['end_y']), color, 4)

        im_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    return im_rgb


def visualize_detection_classification_results(image, detection_classification_results):
    """Returns image with bounding boxes around faces annotated with class labels

    Parameters
    ----------
    image : [type]
        [description]
    detection_classification_results : [type]
        [description]

    Returns
    -------
    [type]
        [description]
    """

    for result in detection_classification_results:

        label = result['class']

        color = (0, 255, 0) if label == "Mask" else (0, 0, 255)

        # include the probability in the label
        label = "{}: {:.2f}%".format(label, result['probability'])

        # display the label and bounding box rectangle on the output
        # frame
        cv2.putText(image, label, (result['start_x'], result['start_y'] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        cv2.rectangle(image, (result['start_x'], result['start_y']),
                      (result['end_x'], result['end_y']), color, 4)

        im_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    return im_rgb
