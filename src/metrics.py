import numpy as np 
import tensorflow as tf
from tensorflow.keras.metrics import MeanIoU, sparse_categorical_accuracy


class VoidPixelMeanIoU(MeanIoU):
    def __init__(self, *args, void_pixel, **kwargs):
        super(VoidPixelMeanIoU, self).__init__(*args, **kwargs)
        self.void_pixel = void_pixel
    
    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred = tf.math.argmax(y_pred, axis=-1)
        if self.void_pixel is not None:
            y_true = tf.reshape(y_true, [-1])
            y_pred = tf.reshape(y_pred, [-1])
            mask = tf.math.not_equal(y_true, self.void_pixel)
            y_true = tf.boolean_mask(y_true, mask)
            y_pred = tf.boolean_mask(y_pred, mask)
        super(VoidPixelMeanIoU, self).update_state(y_true, y_pred, sample_weight)


class ArgmaxMeanIoU(MeanIoU):
    def __init__(self, *args, **kwargs):
        super(ArgmaxMeanIoU, self).__init__(*args, **kwargs)

    def update_state(self, y_true, y_pred, sample_weight=None):
        super(ArgmaxMeanIoU, self).update_state(y_true, tf.math.argmax(y_pred, axis=-1), sample_weight)

def sparse_categorical_accuracy_void_pixel(y_true, y_pred):


class SparseCategoricalAccuracyVoidPixel:
    def __init__(self, void_pixel):
        self.void_pixel = void_pixel
    
    def __call__(self, y_true, y_pred):
        y_true = tf.reshape(y_true, [-1])
        y_pred = tf.reshape(y_pred, [-1])
        mask = tf.math.not_equal(y_true, self.void_pixel)
        y_true = tf.boolean_mask(y_true, mask)
        y_pred = tf.boolean_mask(y_pred, mask)
        return sparse_categorical_accuracy(y_true, y_pred)
