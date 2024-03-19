#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch
import torch.nn as nn  
#from complex_neural_net import *
from .complex_layer import *
from tensorflow.keras.activations import relu
#import .activations import *
from tensorflow import keras

# In[ ]:
import numpy as np
import rasterio




def cvgg16(input_shape):
    ###input
    inputs = layers.Input(shape=input_shape)

    # 第一个卷积块
    x = CConv2D(filters=64, kernel_size=(3, 3), strides =(1,1), padding='same')(inputs)
    x = CBatchNorm(64)(x)
    x = CRelu()(x)
    x = CConv2D(filters=64, kernel_size=(3, 3), strides =(1,1), padding='same')(x)
    x = CBatchNorm(64)(x)
    x = CRelu()(x)
    x = CMaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(x)
    
    
    # 第二个卷积块
    x = CConv2D(filters=128, kernel_size=(3, 3), strides =(1,1), padding='same')(x)
    x = CBatchNorm(128)(x)
    x = CRelu()(x)
    x = CConv2D(filters=128, kernel_size=(3, 3), strides =(1,1), padding='same')(x)
    x = CBatchNorm(128)(x)
    x = CRelu()(x)
    x = CMaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(x)

    # 第三个卷积块
    x = CConv2D(filters=256, kernel_size=(3, 3), strides =(1,1), padding='same')(x)
    x = CBatchNorm(256)(x)
    x = CRelu()(x)
    x = CConv2D(filters=256, kernel_size=(3, 3), strides =(1,1), padding='same')(x)
    x = CBatchNorm(256)(x)
    x = CRelu()(x)
    x = CConv2D(filters=256, kernel_size=(3, 3), strides =(1,1), padding='same')(x)
    x = CBatchNorm(256)(x)
    x = CRelu()(x)
    x = CMaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(x)

    # 第四个卷积块
    x = CConv2D(filters=512, kernel_size=(3, 3), strides =(1,1), padding='same')(x)
    x = CBatchNorm(512)(x)
    x = CRelu()(x)
    x = CConv2D(filters=512, kernel_size=(3, 3), strides =(1,1), padding='same')(x)
    x = CBatchNorm(512)(x)
    x = CRelu()(x)
    x = CConv2D(filters=512, kernel_size=(3, 3), strides =(1,1), padding='same')(x)
    x = CBatchNorm(512)(x)
    x = CRelu()(x)
    x = CMaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(x)

    # 第五个卷积块
    x = CConv2D(filters=512, kernel_size=(3, 3), strides =(1,1), padding='same')(x)
    x = CBatchNorm(512)(x)
    x = CRelu()(x)
    x = CConv2D(filters=512, kernel_size=(3, 3), strides =(1,1), padding='same')(x)
    x = CBatchNorm(512)(x)
    x = CRelu()(x)
    x = CConv2D(filters=512, kernel_size=(3, 3), strides =(1,1), padding='same')(x)
    x = CBatchNorm(512)(x)
    x = CRelu()(x)#######COMPLEX relu
    #x = CMaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(x)########output shape!!!
    
    x = MagnitudeOperation()(x)

    # 创建模型
    #model = vggmodel.Model(inputs=inputs, outputs=x)#########返回layer不是model!!!!!

    return inputs,x

