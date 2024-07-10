

import keras,os
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
from tensorflow.keras import layers, models
from tensorflow.keras.layers import Input, Conv2D, Conv2DTranspose
from tensorflow.keras.models import Model
from tensorflow.keras.losses import MeanSquaredError
from keras.layers import Dense, Dropout
#from dlmodels import*
import dlmodels
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, Conv2DTranspose

def fusion(sentinel_1, sentinel_2, GEDI):
    merge = layers.concatenate([sentinel_1, sentinel_2, GEDI])
    
    # Fusion
    x = Conv2D(filters=512, kernel_size=(3, 3), padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(1e-5))(merge)##512
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(filters=256, kernel_size=(3, 3), padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(5e-5))(x)##256
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    

    
    x = Conv2D(filters=128, kernel_size=(3, 3), padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(1e-5))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(filters=64, kernel_size=(3, 3), padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(5e-5))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(filters=32, kernel_size=(3, 3), padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(1e-5))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    return x




def task(fusion_output):
    # 卷积后尝试使用Batch Normalization
    x = Conv2D(filters=32, kernel_size=(6, 6), padding='valid', kernel_initializer='he_normal', kernel_regularizer=l2(1e-5))(fusion_output)
    x = BatchNormalization()(x)  # 添加Batch Normalization
    x = Activation('relu')(x)  # 激活函数改为单独一层
    
    output_250 = Conv2D(filters=1, kernel_size=(3, 3), padding='same', activation='linear', name='output_250')(x)

    # #上采样部分也加上Batch Normalization
    # y = Conv2DTranspose(filters=32, kernel_size=(3, 3), strides=(2, 2), padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(1e-5))(fusion_output)
    # y = BatchNormalization()(y)  # 添加Batch Normalization
    # y = Activation('relu')(y)  # 激活函数改为单独一层
    # y = Conv2D(filters=32, kernel_size=(6, 6), padding='valid', kernel_initializer='he_normal', kernel_regularizer=l2(1e-5))(y)
    # y = BatchNormalization()(y)  # 再次添加Batch Normalization
    # y = Activation('relu')(y)  # 激活函数改为单独一层
    # output_100 = Conv2D(filters=1, kernel_size=(3, 3), padding='same', activation='linear', name='output_100')(y)
    
    return output_250#, output_100



def create_model(input_shape_s1, input_shape_s10,input_shape_s20, input_shape_gedi):
    # 创建编码器
    input_s1, output_s1 = dlmodels.sresnet18(input_shape_s1)
    input_s10,input_s20,output_s2 = dlmodels.resnet18(input_shape_s10, input_shape_s20)
    input_gedi, output_gedi = dlmodels.vgg16_gedi(input_shape=input_shape_gedi)
    


    # # 获取编码器的输出
    # input_s1, output_s1 = sentinel_1_model.output
    # input_s10,input_s20,output_s2 = sentinel_2_model.output
    # input_gedi, output_gedi = gedi_model.output

    # 融合编码器输出
    fusion_output = fusion(output_s1, output_s2, output_gedi)

    # 为任务创建输出
    #output_250, output_100 = task(fusion_output)
    output_250 = task(fusion_output)

    # 定义完整模型
    model = models.Model(inputs=[input_s1, input_s10, input_s20, input_gedi], 
                         outputs=[output_250])

    return model


# %%




