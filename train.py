from face_mask_detection import models, model_utils, utils

import argparse
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam


def parse_args():
    parser = argparse.ArgumentParser(
        description='Arguments for training a face mask classification model')
    parser.add_argument('--learning_rate', default=0.0001, type=float,
                        help='Learning rate for training a model')
    parser.add_argument('--epochs', default=8, type=int,
                        help='How many epochs to train')
    parser.add_argument('--batch_size', default=32, type=int,
                        help='Batch size during training')
    parser.add_argument('--save_path', type=str, default='models/face-mask-classifier',
                        help='Output path for Tensorflow model')
    parser.add_argument('--dataset_path', type=str, default='dataset/chandrikadeb')
    args = parser.parse_args()

    return args


def main():

    args = parse_args()

    model = models.mobile_net2(input_shape=(224, 224, 3))

    img_data_gen = ImageDataGenerator(
        rescale=1./255., samplewise_center=True,
        samplewise_std_normalization=True,
        rotation_range=30, zoom_range=0.10,
        width_shift_range=0.2, height_shift_range=0.2,
        shear_range=0.15, horizontal_flip=True,
        fill_mode='nearest',
        validation_split=0.2
    )

    train_generator = img_data_gen.flow_from_directory(
        args.dataset_path,
        subset='training',
        batch_size=args.batch_size
    )

    validation_generator = img_data_gen.flow_from_directory(
        args.dataset_path,
        subset='validation',
        batch_size=args.batch_size
    )

    model.compile(loss='binary_crossentropy', optimizer=Adam(
        learning_rate=args.learning_rate), metrics=['accuracy'])

    history = model.fit(train_generator,
                        steps_per_epoch=len(train_generator), validation_data=validation_generator,
                        validation_steps=len(validation_generator), epochs=args.epochs)

    model.save(args.save_path)


if __name__ == '__main__':
    main()
