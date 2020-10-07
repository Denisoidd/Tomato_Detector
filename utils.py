import yaml
import cv2
import numpy as np

import tensorflow.keras.backend as K


def load_config(config_path):
    with open(config_path) as file:
        config = yaml.safe_load(file)
        return config


def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall


def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


def load_image(path, resize=False):
    if resize:
        return np.expand_dims(cv2.resize(cv2.imread(path), (256, 256)), axis=0)
    return np.expand_dims(cv2.imread(path), axis=0)
