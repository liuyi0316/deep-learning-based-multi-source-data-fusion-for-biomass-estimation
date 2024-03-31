#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import tensorflow as tf
from tensorflow.keras import layers
from tensorflow import keras
from tensorflow.keras.layers import Layer, MaxPooling2D

# In[ ]:


# class CConv2D(layers.Layer):
#     def __init__(self, filters, kernel_size, strides, padding, **kwargs):
#         super(CConv2D, self).__init__(**kwargs)
#         self.filters = filters
#         self.kernel_size = kernel_size
#         self.strides = strides
#         self.padding = padding

#         self.re_conv = layers.Conv2D(filters=filters, kernel_size=kernel_size, 
#                                      strides=strides, padding=padding, use_bias=False)
#         self.im_conv = layers.Conv2D(filters=filters, kernel_size=kernel_size, 
#                                      strides=strides, padding=padding, use_bias=False)

#     def build(self, input_shape):
#         # Initialize weights
#         self.re_conv.build(input_shape)
#         self.im_conv.build(input_shape)
#         self.re_conv.kernel.assign(tf.keras.initializers.GlorotUniform()(self.re_conv.kernel.shape))
#         self.im_conv.kernel.assign(tf.keras.initializers.GlorotUniform()(self.im_conv.kernel.shape))
        
#     def call(self, inputs):
#         # Split the channels into real and imaginary parts if needed
#         # Here, it's assumed that the real and imaginary parts are interleaved in the channels dimension
#         x_re = inputs[..., ::2]  # Real parts: Take every other channel starting from 0
#         x_im = inputs[..., 1::2]  # Imaginary parts: Take every other channel starting from 1

#         # Perform the convolution separately on the real and imaginary parts
#         out_re = self.re_conv(x_re) - self.im_conv(x_im)
#         out_im = self.re_conv(x_im) + self.im_conv(x_re)

#         # Stack the real and imaginary parts back together
#         out = tf.concat([out_re, out_im], axis=-1)  # Concatenate along the channels dimension

# #     def call(self, inputs):
# #         # Assuming the input is in the format [batch, height, width, channels, 2]
# #         x_re = inputs[..., 0]
# #         x_im = inputs[..., 1]

# #         out_re = self.re_conv(x_re) - self.im_conv(x_im)
# #         out_im = self.re_conv(x_im) + self.im_conv(x_re)

# #        out = tf.stack([out_re, out_im], axis=-1)

#         return out


# In[ ]:
# class CConv2D(layers.Layer):
#     def __init__(self, filters, kernel_size, strides=(1, 1), padding='same', **kwargs):
#         super(CConv2D, self).__init__(**kwargs)
#         self.filters = filters
#         self.kernel_size = kernel_size
#         self.strides = strides
#         self.padding = padding
#         # 实部卷积层
#         self.re_conv = layers.Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding=padding)
#         # 虚部卷积层
#         self.im_conv = layers.Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding=padding)

#     def call(self, inputs):
#         # 假设 inputs 的最后一个维度已经是复数通道，其中实部和虚部交替出现
#         # 对实部和虚部分别进行卷积处理
#         out_re = self.re_conv(inputs[..., ::2]) - self.im_conv(inputs[..., 1::2])
#         out_im = self.re_conv(inputs[..., 1::2]) + self.im_conv(inputs[..., ::2])
#         # 将实部和虚部的结果沿通道维度拼接
#         out = tf.concat([out_re, out_im], axis=-1)
#         return out
    
# from tensorflow.keras import layers, initializers
# import tensorflow as tf
from tensorflow.keras import initializers

class CConv2D(layers.Layer):
    def __init__(self, filters, kernel_size, strides=(1, 1), padding='same', **kwargs):
        super(CConv2D, self).__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        # 实部卷积层，使用 He 初始化
        self.re_conv = layers.Conv2D(
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            kernel_initializer=initializers.he_normal()  # 指定 He 初始化
        )
        # 虚部卷积层，同样使用 He 初始化
        self.im_conv = layers.Conv2D(
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            kernel_initializer=initializers.he_normal()  # 指定 He 初始化
        )

    def call(self, inputs):
        # 假设 inputs 的最后一个维度已经是复数通道，其中实部和虚部交替出现
        # 对实部和虚部分别进行卷积处理
        out_re = self.re_conv(inputs[..., ::2]) - self.im_conv(inputs[..., 1::2])
        out_im = self.re_conv(inputs[..., 1::2]) + self.im_conv(inputs[..., ::2])
        # 将实部和虚部的结果沿通道维度拼接
        out = tf.concat([out_re, out_im], axis=-1)
        return out



# class CBatchNorm(layers.Layer):
#     def __init__(self, in_channels, **kwargs):
#         super(CBatchNorm, self).__init__(**kwargs)
#         self.in_channels = in_channels

#         # 创建实部和虚部的批量标准化层
#         self.re_batch = layers.BatchNormalization()
#         self.im_batch = layers.BatchNormalization()

#     def call(self, inputs):
#         # 假设输入是 [batch, height, width, channels, 2] 格式，最后一个维度是复数的实部和虚部
#         x_re = inputs[..., 0]
#         x_im = inputs[..., 1]

#         # 分别对实部和虚部进行批量标准化
#         out_re = self.re_batch(x_re)
#         out_im = self.im_batch(x_im)

#         # 将处理过的实部和虚部重新堆叠
#         out = tf.stack([out_re, out_im], axis=-1)

#         return out

class CBatchNorm(layers.Layer):
    def __init__(self, in_channels, **kwargs):
        super(CBatchNorm, self).__init__(**kwargs)
        # 这里不需要指定通道数，BatchNormalization在build时会自动适配
        self.re_batch = layers.BatchNormalization()
        self.im_batch = layers.BatchNormalization()
        self.in_channels = in_channels

    def call(self, inputs):
        # 分割实部和虚部
        real_part = inputs[..., ::2]  # 选择偶数索引通道，即实部
        imag_part = inputs[..., 1::2]  # 选择奇数索引通道，即虚部

        # 分别对实部和虚部进行批量标准化
        out_re = self.re_batch(real_part)
        out_im = self.im_batch(imag_part)

        # 初始化一个空列表来存储交替重组后的结果
        re_im_combined = []
        for re, im in zip(tf.unstack(out_re, axis=-1), tf.unstack(out_im, axis=-1)):
            re_im_combined.append(re)
            re_im_combined.append(im)

        # 沿最后一个维度将实部和虚部交替重组
        out = tf.stack(re_im_combined, axis=-1)

        return out

# In[ ]:


# class CMaxPool2D(layers.Layer):
#     def __init__(self, pool_size=(2, 2), strides=(2, 2), padding='same', **kwargs):
#         super(CMaxPool2D, self).__init__(**kwargs)
#         # 保存池化层的参数
#         self.pool_size = pool_size
#         self.strides = strides
#         self.padding = padding

#     def call(self, inputs):
#         # 分离复数输入的实部和虚部
#         real_part = inputs[..., 0]
#         imag_part = inputs[..., 1]

#         # 计算幅度
#         magnitude = tf.sqrt(tf.square(real_part) + tf.square(imag_part))

#         # 对幅度进行标准的最大池化
#         pooled_magnitude = tf.nn.max_pool(
#             magnitude, 
#             ksize=[1, *self.pool_size, 1], 
#             strides=[1, *self.strides, 1], 
#             padding=self.padding.upper()
#         )

#         # 获取最大池化的索引
#         max_indices = tf.argmax(magnitude, axis=-1)

#         # 使用索引从实部和虚部中获取最大值
#         # 注意: 这里的gather和stack需要经过适当的转换，下面的代码可能需要修改
#         out_re = tf.gather_nd(real_part, max_indices, batch_dims=1)
#         out_im = tf.gather_nd(imag_part, max_indices, batch_dims=1)

#         # 将处理过的实部和虚部重新堆叠
#         out = tf.stack([out_re, out_im], axis=-1)

#         return out


# In[ ]:

# class CMaxPool2D(layers.Layer):
#     def __init__(self, pool_size=(2, 2), strides=(2, 2), padding='same', **kwargs):
#         super(CMaxPool2D, self).__init__(**kwargs)
#         self.pool_size = pool_size
#         self.strides = strides
#         self.padding = padding

#     def call(self, inputs):
#         # 提取实数部分和虚数部分
#         real_part = inputs[..., 0]
#         imag_part = inputs[..., 1]

#         # 计算复数的模长
#         magnitude = tf.sqrt(tf.square(real_part) + tf.square(imag_part))

#         # 使用 TensorFlow 内置的池化函数对模长进行池化
#         pooled_magnitude = tf.nn.max_pool(magnitude,
#                                            ksize=[1, *self.pool_size, 1],  # 设置 ksize 参数的长度为 4
#                                            strides=[1, *self.strides, 1],
#                                            padding=self.padding.upper())

#         return out



# class CMaxPool2D(tf.keras.layers.Layer):
#     def __init__(self, pool_size=(2, 2), strides=(2, 2), padding='VALID', **kwargs):
#         super(CMaxPool2D, self).__init__(**kwargs)
#         self.pool_size = pool_size
#         self.strides = strides
#         self.padding = padding

#     def call(self, inputs):
#         real_part = inputs[..., 0]
#         imag_part = inputs[..., 1]
#         magnitude = tf.sqrt(tf.square(real_part) + tf.square(imag_part))
#         pooled_magnitude = tf.nn.max_pool(magnitude, ksize=56, 
#                                           strides=2, padding='VALID')
#         return pooled_magnitude


class CMaxPool2D(Layer):
    def __init__(self, pool_size=(2, 2), strides=None, padding='valid', **kwargs):
        super(CMaxPool2D, self).__init__(**kwargs)
        self.pool_size = pool_size
        self.strides = strides
        self.padding = padding
        # 创建两个MaxPooling2D层实例，一个用于实部，一个用于虚部
        self.max_pool = MaxPooling2D(pool_size=self.pool_size, strides=self.strides, padding=self.padding)

    def call(self, inputs):
        # 初始化一个空列表来存储池化后的结果
        pooled_outputs = []
        # 对每个复数通道的实部和虚部分别进行池化
        for i in range(0, inputs.shape[-1], 2):  # 步长为2，分别处理每个复数的实部和虚部
            x_re = inputs[..., i]  # 实部
            x_im = inputs[..., i+1]  # 虚部
            # 对实部和虚部进行池化
            out_re = self.max_pool(tf.expand_dims(x_re, axis=-1))
            out_im = self.max_pool(tf.expand_dims(x_im, axis=-1))
            # 将池化后的实部和虚部重新组合
            pooled_outputs.append(out_re)
            pooled_outputs.append(out_im)
        # 将所有池化后的输出沿最后一个维度合并
        out = tf.concat(pooled_outputs, axis=-1)
        return out

    def compute_output_shape(self, input_shape):
        # 计算输出的形状
        space = input_shape[1:-1]
        new_space = []
        for dim in space:
            if dim is None:
                new_dim = None
            else:
                new_dim = (dim + self.pool_size[0] - 1) // self.pool_size[0]  # 假设pool_size在两个维度上是相同的
            new_space.append(new_dim)
        return (input_shape[0],) + tuple(new_space) + (input_shape[-1],)



# class CMaxPool2D(layers.Layer):
#     def __init__(self, kernel_size, **kwargs):
#         super(CMaxPool2D, self).__init__(**kwargs)
#         self.kernel_size = kernel_size

#         # 创建实部和虚部的最大池化层
#         self.CMax_re = layers.MaxPooling2D(pool_size=kernel_size, **kwargs)
#         self.CMax_im = layers.MaxPooling2D(pool_size=kernel_size, **kwargs)

#     def call(self, inputs):
#         # 假设输入是 [batch, height, width, channels, 2] 格式，最后一个维度是复数的实部和虚部
#         x_re = inputs[..., 0]
#         x_im = inputs[..., 1]

#         # 分别对实部和虚部进行最大池化
#         out_re = self.CMax_re(x_re)
#         out_im = self.CMax_im(x_im)

#         # 将处理过的实部和虚部重新堆叠
#         out = tf.stack([out_re, out_im], axis=-1)

#         return out


# In[ ]:


class MagnitudeOperation(layers.Layer):
    def __init__(self, **kwargs):
        super(MagnitudeOperation, self).__init__(**kwargs)

    def call(self, inputs):
        magnitudes = []  # 存储每个复数通道的幅度
        for i in range(0, inputs.shape[-1], 2):
            x_re = inputs[..., i]  # 实部
            x_im = inputs[..., i+1]  # 虚部
           
            magnitude = tf.sqrt(tf.square(x_re) + tf.square(x_im) + 1e-10)
            magnitudes.append(magnitude)

        # 沿最后一个维度堆叠所有的幅度
        out = tf.stack(magnitudes, axis=-1)
        return out

        

class CRelu(layers.Layer):
    def __init__(self, **kwargs):
        super(CRelu, self).__init__(**kwargs)
    
    def call(self, inputs):
        # 初始化一个空列表来存储激活后的结果
        activated_outputs = []
        # 循环处理每个复数的实部和虚部
        for i in range(0, inputs.shape[-1], 2):
            x_re = inputs[..., i]  # 实部
            x_im = inputs[..., i+1]  # 虚部
            
            # 对实部和虚部应用ReLU激活函数
            activated_re = tf.nn.relu(x_re)
            activated_im = tf.nn.relu(x_im)
            
            # 将激活后的实部和虚部添加到列表中
            activated_outputs.append(activated_re)
            activated_outputs.append(activated_im)
        
        # 沿最后一个维度将激活后的实部和虚部交替重组
        out = tf.stack(activated_outputs, axis=-1)
        #out = tf.reshape(out, shape=inputs.shape)
        
        return out
