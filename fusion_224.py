

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

    x = Conv2D(filters=32, kernel_size=(6, 6), padding='valid', kernel_initializer='he_normal', kernel_regularizer=l2(1e-5))(fusion_output)
    x = BatchNormalization()(x)  
    x = Activation('relu')(x)  
    
    output_250 = Conv2D(filters=1, kernel_size=(3, 3), padding='same', activation='linear', name='output_250')(x)

  
 
    
    return output_250#, output_100



def create_model(input_shape_s1, input_shape_s10,input_shape_s20, input_shape_gedi):
   
    input_s1, output_s1 = dlmodels.sresnet18(input_shape_s1)
    input_s10,input_s20,output_s2 = dlmodels.resnet18(input_shape_s10, input_shape_s20)
    input_gedi, output_gedi = dlmodels.vgg16_gedi(input_shape=input_shape_gedi)
    


  
    # input_s1, output_s1 = sentinel_1_model.output
    # input_s10,input_s20,output_s2 = sentinel_2_model.output
    # input_gedi, output_gedi = gedi_model.output

   
    fusion_output = fusion(output_s1, output_s2, output_gedi)

 
    #output_250, output_100 = task(fusion_output)
    output_250 = task(fusion_output)

   
    model = models.Model(inputs=[input_s1, input_s10, input_s20, input_gedi], 
                         outputs=[output_250])

    return model


# %%




