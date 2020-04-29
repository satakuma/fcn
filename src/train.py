import numpy as np
import matplotlib.pyplot as plt
import sys

import tensorflow as tf
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.optimizers import Adadelta, RMSprop, SGD
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint, LearningRateScheduler

from models import *
from metrics import SparseCategoricalAccuracyVoidPixel, VoidPixelMeanIoU
from dalib import ImageAugmentator

TRAIN_LEN = 8498
VAL_LEN = 736
nb_classes = 21
void_pixel = 255
target_size = (480, 480)
dataset_dir = sys.argv[1]
log_dir = sys.argv[2]

strategy = tf.distribute.MirroredStrategy()
with strategy.scope():
    model = build_fcn16s(nb_classes)
    optimizer = Adadelta(lr=8e-3)
    loss = "sparse_categorical_crossentropy"
    metrics = [
        SparseCategoricalAccuracyVoidPixel(void_pixel=void_pixel),
        VoidPixelMeanIoU(void_pixel=void_pixel),
    ]

    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    model.summary()


tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)
checkpoint_callback = ModelCheckpoint(filepath=f"{log_dir}/segmentation-fcn16s-voc-checkpoint-weights.h5", save_weights_only=True)
lr_schedule_callback = LearningRateScheduler(schedule=lambda ep, lr: lr * 0.95 if ep > 10 else lr, verbose=1)
callbacks = [checkpoint_callback, tensorboard_callback, lr_schedule_callback]

batch_size = 6

train_generator = ImageAugmentator(
    f"{dataset_dir}/train",
    batch_size=batch_size,
    target_size=target_size,
    num_classes=nb_classes,
    sampling="class",
    flip=True,
    scales=(480, 600),
    crop="class",
    preprocessing_function=preprocess_input,
    void_pixel=void_pixel,
    pad_to_multiply_of=32,
)

val_generator = ImageAugmentator(
    f"{dataset_dir}/val",
    batch_size=1,
    num_classes=nb_classes,
    sampling="random",
    flip=False,
    scales=None,
    crop=None,
    preprocessing_function=preprocess_input,
    void_pixel=void_pixel,
    pad_to_multiply_of=32,
)

model.fit(
    train_generator,
    steps_per_epoch=TRAIN_LEN // batch_size,
    epochs=50,
    validation_data=val_generator,
    validation_steps=VAL_LEN,
    callbacks=callbacks
)

model.save("segmentation-voc-fcn16s.h5")
