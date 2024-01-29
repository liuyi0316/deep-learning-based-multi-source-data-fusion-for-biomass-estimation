import os

import tensorflow as tf

from tensorflow import keras

#import keras,os
#from keras.preprocessing.image import ImageDataGenerator
import numpy as np
from tensorflow.keras import layers, models
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D
from tensorflow.keras.models import Model
from vgg16_s20 import vgg16_20



def vgg16(input_shape_10=(224, 224, 4),input_shape_20m):
    input_20, output_20 = vgg16_20(input_shape_20m)
    # 定义输入
    inputs = layers.Input(input_shape_10)

    # 第一个卷积块
    x = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(inputs)
    x = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)
    
    
    #concatenate
    inputs_20, feature_20 = vgg16_20(input_shape=(112,112,6))
    merge = layers.concatenate([x, output_20])  # 确保这两个张量形状可以融合
    
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
    #x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)

    

    return inputs,input_20,x 
