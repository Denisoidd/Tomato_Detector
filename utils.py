import yaml
import cv2
import numpy as np

import tensorflow.keras.backend as K


def load_config(config_path):
    with open(config_path) as file:
        config = yaml.safe_load(file)
        return config


def load_image(path, resize=False):
    if resize:
        return np.expand_dims(cv2.resize(cv2.imread(path), (256, 256)), axis=0)
    return np.expand_dims(cv2.imread(path), axis=0)
