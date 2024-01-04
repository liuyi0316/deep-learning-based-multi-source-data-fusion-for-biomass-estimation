import os

import tensorflow as tf

from tensorflow import keras

#import keras,os
#from keras.preprocessing.image import ImageDataGenerator
import numpy as np
from tensorflow.keras import layers, models
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D
from tensorflow.keras.models import Model


def vgg16_20(input_shape=(112,112,6)): #sentinel-2 20m
    # 定义输入
    inputs = Input(shape=input_shape)

    # 第一个卷积块
    x = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(inputs)
    x = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(x)
    
    return inputs,x


def vgg16(input_shape=(224, 224, 4)):
    # 定义输入
    inputs = layers.Input(shape=input_shape)

    # 第一个卷积块
    x = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(inputs)
    x = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)
    
    
    #concatenate
    inputs_20, feature_20 = vgg16_20(input_shape=(112,112,6))
    merge = layers.concatenate([x, feature_20])  # 确保这两个张量形状可以融合
    
    # 第二个卷积块
    x = Conv2D(filters=128, kernel_size=(3, 3), padding='same', activation='relu')(merge)
    x = Conv2D(filters=128, kernel_size=(3, 3), padding='same', activation='relu')(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)

    # 第三个卷积块
    x = Conv2D(filters=256, kernel_size=(3, 3), padding='same', activation='relu')(x)
    x = Conv2D(filters=256, kernel_size=(3, 3), padding='same', activation='relu')(x)
    x = Conv2D(filters=256, kernel_size=(3, 3), padding='same', activation='relu')(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)

    # 第四个卷积块
    x = Conv2D(filters=512, kernel_size=(3, 3), padding='same', activation='relu')(x)
    x = Conv2D(filters=512, kernel_size=(3, 3), padding='same', activation='relu')(x)
    x = Conv2D(filters=512, kernel_size=(3, 3), padding='same', activation='relu')(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)

    # 第五个卷积块
    x = Conv2D(filters=512, kernel_size=(3, 3), padding='same', activation='relu')(x)
    x = Conv2D(filters=512, kernel_size=(3, 3), padding='same', activation='relu')(x)
    x = Conv2D(filters=512, kernel_size=(3, 3), padding='same', activation='relu')(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)

    

    return x 
