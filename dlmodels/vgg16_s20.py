#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os

import tensorflow as tf

from tensorflow import keras

#import keras,os
#from keras.preprocessing.image import ImageDataGenerator
import numpy as np
from tensorflow.keras import layers, models
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation

# In[ ]:


def vgg16_20(input_shape=(112,112,6)): #sentinel-2 20m
    # 定义输入
    inputs = Input(shape=input_shape)

    x = Conv2D(filters=64, kernel_size=(3, 3), padding='same', kernel_initializer='he_normal')(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(filters=64, kernel_size=(3, 3), padding='same', kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    return inputs,x


