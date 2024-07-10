from tensorflow.keras import layers, models
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


def conv_batchnorm_relu(x, filters, kernel_size, strides=(1,1)):
   # """卷积 + 批标准化 + ReLU激活函数"""
    x = CConv2D(filters, kernel_size, strides=strides, padding="same")(x)
    x = CBatchNorm()(x)
    x = CRelu()(x)
    return x

def identity_block(input_tensor, filters):
 #   """残差块，输入和输出尺寸相同"""
    x = conv_batchnorm_relu(input_tensor,filters, kernel_size=3)
    x =  CConv2D(filters, (3,3),(1,1), padding="same")(x)
    x = CBatchNorm()(x)

    x = layers.add([x, input_tensor])
    x = CRelu()(x)
    return x

def conv_block(input_tensor, filters, strides=(2,2)):
  #  """残差块，包括一个步长不为1的卷积层以改变维度"""
    x = conv_batchnorm_relu(input_tensor, filters, kernel_size=3, strides=strides)
    x = CConv2D(filters,(3,3),(1,1), padding="same")(x)
    x = CBatchNorm()(x)

    shortcut = CConv2D(filters, (1,1), strides=strides, padding="same")(input_tensor)
    shortcut = CBatchNorm()(shortcut)

    x = layers.add([x, shortcut])
    x = CRelu()(x)
    return x

def sresnet18(input_shape_s1):
    inputs = layers.Input(input_shape_s1,name='s1')


    # 初始卷积层
    x = conv_batchnorm_relu(inputs, 64, kernel_size=(7,7), strides=(2,2))
    x = CMaxPool2D(pool_size=(3, 3), strides=(2, 2), padding='same')(x)
    # 第一个卷积块
    x = conv_block(x, 64, strides=(1,1))
    x = identity_block(x, 64)

    # 第二个卷积块
    x = conv_block(x, 128)
    x = identity_block(x, 128)

    # 第三个卷积块
    x = conv_block(x, 256)
    x = identity_block(x, 256)
    
    x = MagnitudeOperation()(x)


    return inputs, x



