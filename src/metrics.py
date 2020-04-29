import numpy as np 
import tensorflow as tf
from tensorflow.keras.metrics import MeanIoU

class VoidPixelMeanIoU(MeanIou):
    def __init__(self, *args, void_pixel=None, **kwargs):
        super(ArgmaxMeanIoU, self).__init__(*args, **kwargs)
        self.void_pixel = void_pixel
    
    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred = tf.math.argmax(y_pred, axis=-1)
        if self.void_pixel is not None:
            mask = tf.math.not_equal(y_true, self.void_pixel)
            y_true = tf.boolean_mask(y_true, mask)
            y_pred = tf.boolean_mask(y_pred, mask)
        super(ArgmaxMeanIoU, self).update_state(y_true, y_pred, sample_weight)

class ArgmaxMeanIoU(MeanIoU):
    def __init__(self, *args, **kwargs):
        super(ArgmaxMeanIoU, self).__init__(*args, **kwargs)

    def update_state(self, y_true, y_pred, sample_weight=None):
        super(ArgmaxMeanIoU, self).update_state(y_true, tf.math.argmax(y_pred, axis=-1), sample_weight)

class SparseCategoricalAccuracyVoidPixel:
    def __init__(self, void_pixel):
        self.void_pixel = void_pixel
    
    def __call__(self, y_true, y_pred):
        mask = tf.math.not_equal(y_true, self.void_pixel)
        y_true = tf.boolean_mask(y_true, mask)
        y_pred = tf.boolean_mask(y_pred, mask)
        return sparse_categorical_accuracy_void_pixel(y_true, y_pred)

