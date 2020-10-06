import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras import layers


def get_model(im_h, im_w, n_cl):
    return Sequential([
        data_augmentation_layer(im_h, im_w),
        layers.experimental.preprocessing.Rescaling(1. / 255, input_shape=(im_h, im_w, 3)),
        layers.Conv2D(16, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(32, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(128, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(256, 3, padding='same', activation='relu'),
        layers.GlobalAveragePooling2D(),
        layers.Dense(n_cl)
    ])


def data_augmentation_layer(im_h, im_w):
    """
    Data augmentation layer to fight with overfitting.
    Different types of flips, small zoom and rotation data augmentation
    :param im_h: image height
    :param im_w: image width
    :return: sequential model for data augmentation
    """
    return Sequential(
        [
            layers.experimental.preprocessing.RandomFlip(input_shape=(im_h, im_w, 3)),
            layers.experimental.preprocessing.RandomRotation(0.1),
            layers.experimental.preprocessing.RandomZoom(0.1),
        ]
    )