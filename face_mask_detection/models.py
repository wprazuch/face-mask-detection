from tensorflow.keras.preprocessing.image import img_to_array, load_img

from tensorflow.keras.applications.mobilenet import preprocess_input
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers
from tensorflow.keras import Model


def mobile_net2(input_shape):
    mobilenet = MobileNetV2(include_top=False, weights='imagenet', input_shape=input_shape)
    for layer in mobilenet.layers:
        layer.trainable = False
    mobilenet_out = mobilenet.output
    model = layers.GlobalAveragePooling2D()(mobilenet_out)
    model = layers.Dense(512, activation='relu')(model)
    model = layers.Dropout(0.5)(model)
    model = layers.Dense(2, activation='softmax')(model)

    network = Model(inputs=[mobilenet.input], outputs=[model])

    return network
