from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.applications.mobilenet import preprocess_input
from tensorflow.image import resize
from sklearn.preprocessing import OneHotEncoder
import numpy as np
import os


def prepare_dataset(dataset_path, input_shape, class_mapping):

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
    image = preprocess_input(image)
    image = resize(image, input_shape)
    return image


def to_one_hot(labels):
    onehot = OneHotEncoder()
    labels = np.array(labels).reshape(-1, 1)
    labels_one_hot = onehot.fit_transform(labels).toarray()

    return labels_one_hot
