import os
import skimage.transform as trans
import numpy as np
from keras.models import *
from keras.layers import *
from tensorflow.keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as keras
from tensorflow.python import tf2


def conv_block(inputs, filters=64, kernel_size=3, activation='relu', dropout_rate=0.1):
    conv = Conv2D(filters, kernel_size, activation=activation, padding='same', kernel_initializer='he_normal')(inputs)
    conv_out = Conv2D(filters, kernel_size, activation=activation, padding='same', kernel_initializer='he_normal')(conv)
    drop_out = Dropout(dropout_rate)(conv_out)
    return conv_out, drop_out


def encode_conv_block(inputs, filters=64, kernel_size=3, activation='relu', pool_size=(2, 2), dropout_rate=0.1):
    conv_out, drop_out = conv_block(inputs, filters, kernel_size, activation, dropout_rate)
    pool_out = AveragePooling2D(pool_size=pool_size)(drop_out)
    return conv_out, drop_out, pool_out


def decode_conv_block(inputs, drop_out, up_sample_size=(2, 2), filters=64, kernel_size=3, activation='relu'):
    up = UpSampling2D(size=up_sample_size)(inputs)
    up = Conv2D(filters, kernel_size, activation=activation, padding='same', kernel_initializer='he_normal')(up)
    merged = concatenate([drop_out, up], axis=3)
    conv_out, drop_out = conv_block(merged, filters, kernel_size, activation)
    return conv_out, drop_out


def get_model(input_size=(256, 256, 1)):
    inputs = Input(input_size)

    conv_out1, drop_out1, pool_out1 = encode_conv_block(inputs, 32, 3, 'relu', (2, 2), 0.1)
    conv_out2, drop_out2, pool_out2 = encode_conv_block(pool_out1, 128, 3, 'relu', (2, 2), 0.1)

    conv_out5, drop_out5 = conv_block(pool_out2, 512, 3, 'relu', 0.01)

    decode_conv_out8, _ = decode_conv_block(drop_out5, drop_out2, (2, 2), 128, 3, 'relu')
    decode_conv_out9, _ = decode_conv_block(decode_conv_out8, drop_out1, (2, 2), 32, 3, 'relu')

    conv9 = Conv2D(16, 3, activation='relu', padding='same', kernel_initializer='he_normal')(decode_conv_out9)
    conv10 = Conv2D(14, 1, activation='sigmoid')(conv9)

    model = Model(inputs=inputs, outputs=conv10)
    print(model.summary())

    return model


# unet_model = get_model()
