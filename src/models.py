import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, Dropout, Conv2DTranspose, Add
from tensorflow.keras.initializers import Zeros


def build_fcn32s(nb_classes, target_size=(None, None)):
    inputs = Input(shape=(*target_size, 3))
    vgg = VGG16(weights='imagenet', include_top=False, input_tensor=inputs, input_shape=(*target_size, 3))
    x = Conv2D(4096, (7, 7), activation='relu', padding='same')(vgg.output)
    x = Dropout(0.5)(x)
    x = Conv2D(4096, (1, 1), activation='relu', padding='same')(x)
    x = Dropout(0.5)(x)
    x = Conv2D(nb_classes, (1, 1), padding='same', kernel_initializer='he_normal')(x)
    x = Conv2DTranspose(nb_classes, (64, 64), strides=(32, 32), use_bias=False, padding='same', activation='softmax', name='fcn32s-transpose')(x)

    model = Model(inputs=inputs, outputs=x)
    return model


def build_fcn16s(nb_classes, target_size=(None, None)):
    inputs = Input(shape=(*target_size, 3))
    vgg = VGG16(weights='imagenet', include_top=False, input_tensor=inputs, input_shape=(*target_size, 3))
    x = Conv2D(4096, (7, 7), activation='relu', padding='same')(vgg.output)
    x = Dropout(0.5)(x)
    x = Conv2D(4096, (1, 1), activation='relu', padding='same')(x)
    x = Dropout(0.5)(x)
    x = Conv2D(nb_classes, (1, 1), padding='same', kernel_initializer='he_normal')(x)

    x = Conv2DTranspose(nb_classes, (4, 4), strides=(2, 2), use_bias=False, padding='same', activation='relu', name='fcn16s-transpose-first')(x)

    skip_con = Conv2D(nb_classes, (1, 1), strides=(1, 1), padding='same', bias_initializer=Zeros(), kernel_initializer=Zeros(), name='fcn16s-skip-con')(vgg.get_layer(name="block4_pool").output)
    x = Add()([x, skip_con])
    x = Conv2DTranspose(nb_classes, (32, 32), strides=(16, 16), use_bias=False, padding='same', activation='softmax', name='fcn16s-transpose-second')(x)

    model = Model(inputs=inputs, outputs=x)
    return model

