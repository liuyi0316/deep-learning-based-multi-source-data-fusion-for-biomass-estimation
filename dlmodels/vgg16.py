import os

import tensorflow as tf

from tensorflow import keras

#import keras,os
#from keras.preprocessing.image import ImageDataGenerator
import numpy as np
from tensorflow.keras import layers, models
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D
from tensorflow.keras.models import Model
from .vgg16_s20 import vgg16_20

from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, MaxPooling2D

def vgg16(input_shape_10=(224, 224, 4), input_shape_20m=(112, 112, 6)):
    input_20, output_20 = vgg16_20(input_shape_20m)  # 假设这个函数返回的是处理 input_shape_20m 的特征图
    inputs = layers.Input(input_shape_10)

    # 第一个卷积块
    x = Conv2D(filters=64, kernel_size=(3, 3), padding='same', kernel_initializer='he_normal')(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(filters=64, kernel_size=(3, 3), padding='same', kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)
    
    # Concatenate
    merge = layers.concatenate([x, output_20])  # 确保这两个张量形状可以融合
    
    # 第二个卷积块
    x = Conv2D(filters=128, kernel_size=(3, 3), padding='same', kernel_initializer='he_normal')(merge)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(filters=128, kernel_size=(3, 3), padding='same', kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)

    # 第三个卷积块
    x = Conv2D(filters=256, kernel_size=(3, 3), padding='same', kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(filters=256, kernel_size=(3, 3), padding='same', kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(filters=256, kernel_size=(3, 3), padding='same', kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)

    # 第四个卷积块
    x = Conv2D(filters=512, kernel_size=(3, 3), padding='same', kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(filters=512, kernel_size=(3, 3), padding='same', kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(filters=512, kernel_size=(3, 3), padding='same', kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)

    # 第五个卷积块
    x = Conv2D(filters=512, kernel_size=(3, 3), padding='same', kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(filters=512, kernel_size=(3, 3), padding='same', kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(filters=512, kernel_size=(3, 3), padding='same', kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    # 注意，最后一个卷积块后未添加MaxPooling层，按照你的代码保留

    return inputs, input_20, x
