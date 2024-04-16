from tensorflow.keras import layers, models
from .vgg16_s20 import vgg16_20

def conv_batchnorm_relu(x, filters, kernel_size, strides=1):
   # """卷积 + 批标准化 + ReLU激活函数"""
    x = layers.Conv2D(filters, kernel_size, strides=strides, padding="same", kernel_initializer="he_normal")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    return x

def identity_block(input_tensor, filters):
 #   """残差块，输入和输出尺寸相同"""
    x = conv_batchnorm_relu(input_tensor, filters, kernel_size=3)
    x = layers.Conv2D(filters, 3, padding="same", kernel_initializer="he_normal")(x)
    x = layers.BatchNormalization()(x)

    x = layers.add([x, input_tensor])
    x = layers.Activation("relu")(x)
    return x

def conv_block(input_tensor, filters, strides=2):
  #  """残差块，包括一个步长不为1的卷积层以改变维度"""
    x = conv_batchnorm_relu(input_tensor, filters, kernel_size=3, strides=strides)
    x = layers.Conv2D(filters, 3, padding="same", kernel_initializer="he_normal")(x)
    x = layers.BatchNormalization()(x)

    shortcut = layers.Conv2D(filters, 1, strides=strides, padding="same", kernel_initializer="he_normal")(input_tensor)
    shortcut = layers.BatchNormalization()(shortcut)

    x = layers.add([x, shortcut])
    x = layers.Activation("relu")(x)
    return x

def resnet18(input_shape_10=(224, 224, 4), input_shape_20m=(112, 112, 6)):
    input_20, output_20 = vgg16_20(input_shape_20m)  # 假设这个函数返回的是处理 input_shape_20m 的特征图
    inputs = layers.Input(input_shape_10)

    # 初始卷积层
    x = conv_batchnorm_relu(inputs, 64, kernel_size=7, strides=2)
    merge = layers.concatenate([x, output_20])  # 确保这两个张量形状可以融合
    x = layers.MaxPooling2D(pool_size=(3, 3), strides=2, padding="same")(merge)
    
    # 第一个卷积块
    x = conv_block(x, 64, strides=1)
    x = identity_block(x, 64)

    # 第二个卷积块
    x = conv_block(x, 128)
    x = identity_block(x, 128)

    # 第三个卷积块
    x = conv_block(x, 256)
    x = identity_block(x, 256)


    return inputs, input_20, x



