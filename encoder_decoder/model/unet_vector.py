from keras.models import *
from keras.layers import *


def conv_block(inputs, filters=64, kernel_size=3, activation='relu', dropout_rate=0.1):
    conv = Conv2D(filters, kernel_size, activation=activation, padding='same', kernel_initializer='he_normal')(inputs)
    conv_out = Conv2D(filters, kernel_size, activation=activation, padding='same', kernel_initializer='he_normal')(conv)
    drop_out = Dropout(dropout_rate)(conv_out)
    return conv_out, drop_out


def encode_conv_block(inputs, filters=64, kernel_size=3, activation='relu', pool_size=(2, 2), dropout_rate=0.2):
    conv_out, drop_out = conv_block(inputs, filters, kernel_size, activation, dropout_rate)
    pool_out = MaxPooling2D(pool_size=pool_size)(drop_out)
    return conv_out, drop_out, pool_out


def decode_conv_block(inputs, drop_out, up_sample_size=(2, 2), filters=64, kernel_size=3, activation='relu'):
    up = UpSampling2D(size=up_sample_size)(inputs)
    up = Conv2D(filters, kernel_size, activation=activation, padding='same', kernel_initializer='he_normal')(up)
    merged = concatenate([drop_out, up], axis=3)
    conv_out, drop_out = conv_block(merged, filters, kernel_size, activation)
    return conv_out, drop_out


def get_model(input_size=(512, 512, 3)):
    inputs = Input(input_size)

    conv_out1, drop_out1, pool_out1 = encode_conv_block(inputs, 4, 3, 'relu', (2, 2), 0.1)
    conv_out2, drop_out2, pool_out2 = encode_conv_block(pool_out1, 8, 3, 'relu', (2, 2), 0.1)
    conv_out3, drop_out3, pool_out3 = encode_conv_block(pool_out2, 16, 3, 'relu', (2, 2), 0.1)
    conv_out4, drop_out4, pool_out4 = encode_conv_block(pool_out3, 32, 3, 'relu', (2, 2), 0.1)
    conv_out5, drop_out5, pool_out5 = encode_conv_block(pool_out4, 64, 3, 'relu', (2, 2), 0.1)
    conv_out6, drop_out6, pool_out6 = encode_conv_block(pool_out5, 128, 3, 'relu', (2, 2), 0.1)
    conv_out7, drop_out7, pool_out7 = encode_conv_block(pool_out6, 256, 3, 'relu', (2, 2), 0.1)
    # conv_out8, drop_out8, pool_out8 = encode_conv_block(pool_out7, 512, 3, 'relu', (2, 2), 0.1)

    conv_out9, drop_out9 = conv_block(pool_out7, 512, 3, 'relu', 0.1)

    # decode_conv_out8, _ = decode_conv_block(conv_out9, drop_out8, (2, 2), 512, 3, 'relu')
    decode_conv_out7, _ = decode_conv_block(conv_out9, drop_out7, (2, 2), 256, 3, 'relu')
    decode_conv_out6, _ = decode_conv_block(decode_conv_out7, drop_out6, (2, 2), 128, 3, 'relu')
    decode_conv_out5, _ = decode_conv_block(decode_conv_out6, drop_out5, (2, 2), 64, 3, 'relu')
    decode_conv_out4, _ = decode_conv_block(decode_conv_out5, drop_out4, (2, 2), 32, 3, 'relu')
    decode_conv_out3, _ = decode_conv_block(decode_conv_out4, drop_out3, (2, 2), 16, 3, 'relu')
    decode_conv_out2, _ = decode_conv_block(decode_conv_out3, drop_out2, (2, 2), 8, 3, 'relu')
    decode_conv_out1, _ = decode_conv_block(decode_conv_out2, drop_out1, (2, 2), 3, 3, 'relu')

    conv_0 = Conv2D(3, 3, activation='relu', padding='same', kernel_initializer='he_normal')(decode_conv_out1)
    conv_1 = Conv2D(3, 1, activation='sigmoid')(conv_0)

    model = Model(inputs=inputs, outputs=conv_1)

    return model

