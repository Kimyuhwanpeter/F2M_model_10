# -*- coding:utf-8 -*-
import tensorflow as tf

l1_l2 = tf.keras.regularizers.L1L2(0.00001, 0.000001)
l1 = tf.keras.regularizers.l1(0.00001)

class InstanceNormalization(tf.keras.layers.Layer):
  #"""Instance Normalization Layer (https://arxiv.org/abs/1607.08022)."""

    def __init__(self, epsilon=1e-5):
        super(InstanceNormalization, self).__init__()
        self.epsilon = epsilon
    
    def build(self, input_shape):
        self.scale = self.add_weight(
            name='scale',
            shape=input_shape[-1:],
            initializer=tf.random_normal_initializer(0., 0.02),
            trainable=True)
        self.offset = self.add_weight(
            name='offset',
            shape=input_shape[-1:],
            initializer='zeros',
            trainable=True)
    
    def call(self, x):
        mean, variance = tf.nn.moments(x, axes=[1, 2], keepdims=True)
        inv = tf.math.rsqrt(variance + self.epsilon)
        normalized = (x - mean) * inv
        return self.scale * normalized + self.offset

def residual_block(input, dilation=1):

    h = input

    h = tf.keras.layers.ReLU()(h)
    h = tf.pad(h, [[0,0],[dilation,dilation],[dilation,dilation],[0,0]], mode='REFLECT', constant_values=0)
    h = tf.keras.layers.Conv2D(filters=256, kernel_size=3, padding="valid", use_bias=False,
                                kernel_regularizer=l1_l2, activity_regularizer=l1, dilation_rate=dilation)(h)
    h = InstanceNormalization()(h)
    h = tf.keras.layers.ReLU()(h)

    h = tf.pad(h, [[0,0],[dilation,dilation],[dilation,dilation],[0,0]], mode='REFLECT', constant_values=0)
    h = tf.keras.layers.DepthwiseConv2D(kernel_size=3, padding="valid", use_bias=False,
                                        depthwise_regularizer=l1_l2, activity_regularizer=l1, dilation_rate=dilation)(h)
    h = InstanceNormalization()(h)
    h = tf.keras.layers.ReLU()(h)

    h = tf.keras.layers.Conv2D(filters=256, kernel_size=1, strides=1, padding="valid", use_bias=False,
                                kernel_regularizer=l1_l2, activity_regularizer=l1)(h)
    h = InstanceNormalization()(h)
    
    return h + input

def F2M_generator(input_shape=(256, 256, 3)):
    # 입력을 256으로 하되, 출력은 1024로 리사이즈하여 사용하기!
    h = inputs = tf.keras.Input(input_shape)
    t = targets = tf.keras.Input(input_shape)

    h = tf.keras.layers.ZeroPadding2D((3,3))(h)
    h = tf.keras.layers.Conv2D(filters=32, kernel_size=7, strides=1, padding="valid", use_bias=False,
                               kernel_regularizer=l1_l2, activity_regularizer=l1)(h)
    h = InstanceNormalization()(h)

    t = tf.keras.layers.ZeroPadding2D((3,3))(t)
    t = tf.keras.layers.Conv2D(filters=32, kernel_size=7, strides=1, padding="valid", use_bias=False,
                               kernel_regularizer=l1_l2, activity_regularizer=l1)(t)
    t = InstanceNormalization()(t)

    h_t = tf.concat([h, t], -1)
    h_t = tf.keras.layers.ReLU()(h_t)  # [256, 256, 64] # 이 부분에 relu 를해서 업데이트 에러가 나는것같음
    h_t_x, h_t_y = tf.image.image_gradients(h_t)
    h_t_grad = tf.math.add(tf.math.abs(h_t_x), tf.math.abs(h_t_y))

    h = tf.keras.layers.Conv2D(filters=64, kernel_size=3, strides=2, padding="same", use_bias=False,
                               kernel_regularizer=l1_l2, activity_regularizer=l1)(h)
    h = InstanceNormalization()(h)

    t = tf.keras.layers.Conv2D(filters=64, kernel_size=3, strides=2, padding="same", use_bias=False,
                               kernel_regularizer=l1_l2, activity_regularizer=l1)(t)
    t = InstanceNormalization()(t)

    h_t2 = tf.concat([h,t], -1)
    h_t2 = tf.keras.layers.ReLU()(h_t2) # [128, 128, 128]
    h_t2_x, h_t2_y = tf.image.image_gradients(h_t2)
    h_t2_grad = tf.math.add(tf.math.abs(h_t2_x), tf.math.abs(h_t2_y))

    h = tf.keras.layers.Conv2D(filters=128, kernel_size=3, strides=2, padding="same", use_bias=False,
                               kernel_regularizer=l1_l2, activity_regularizer=l1)(h)
    h = InstanceNormalization()(h)

    t = tf.keras.layers.Conv2D(filters=128, kernel_size=3, strides=2, padding="same", use_bias=False,
                               kernel_regularizer=l1_l2, activity_regularizer=l1)(t)
    t = InstanceNormalization()(t)

    h_t3 = tf.concat([h,t], -1)
    h_t3 = tf.keras.layers.ReLU()(h_t3) # [64, 64, 256]
    
    ########################################################################################################
    h_t = tf.keras.layers.Conv2D(filters=128, kernel_size=3, strides=2, padding="same", use_bias=False,
                                 kernel_regularizer=l1_l2, activity_regularizer=l1)(h_t)
    h_t = InstanceNormalization()(h_t)  # [128, 128, 128]

    h_t2 = tf.keras.layers.Conv2D(filters=128, kernel_size=1, strides=1, padding="same", use_bias=False,
                                  kernel_regularizer=l1_l2, activity_regularizer=l1)(h_t2)
    h_t2 = InstanceNormalization()(h_t2)    # [128, 128, 128]

    h_t2 = h_t + h_t2
    h_t2 = tf.keras.layers.ReLU()(h_t2) # [128, 128, 128]

    h_t2 = tf.keras.layers.Conv2D(filters=256, kernel_size=3, strides=2, padding="same", use_bias=False,
                                  kernel_regularizer=l1_l2, activity_regularizer=l1)(h_t2)
    h_t2 = InstanceNormalization()(h_t2)    # [64, 64, 256]

    h_t3 = tf.keras.layers.Conv2D(filters=256, kernel_size=1, strides=1, padding="same", use_bias=False,
                                  kernel_regularizer=l1_l2, activity_regularizer=l1)(h_t3)
    h_t3 = InstanceNormalization()(h_t3)    # [64, 64, 256]

    h_t3 = h_t2 + h_t3  # [64, 64, 256]
    h = h_t3
    ########################################################################################################

    for i in range(6):
        h = residual_block(h, dilation=i+1)

    h_mid = tf.keras.layers.GlobalAveragePooling2D()(h)

    h = tf.keras.layers.Conv2DTranspose(filters=128, kernel_size=3, strides=2, padding="same", use_bias=False,
                                        kernel_regularizer=l1_l2, activity_regularizer=l1)(h)
    h = InstanceNormalization()(h)
    h = tf.keras.layers.ReLU()(h)   # [128, 128, 128]
    #h = h_t2_grad + h

    h = tf.keras.layers.Conv2DTranspose(filters=64, kernel_size=3, strides=2, padding="same", use_bias=False,
                                        kernel_regularizer=l1_l2, activity_regularizer=l1)(h)
    h = InstanceNormalization()(h)
    h = tf.keras.layers.ReLU()(h)   # [256, 256, 64]
    #h = h + h_t_grad

    h = tf.keras.layers.Conv2DTranspose(filters=32, kernel_size=3, strides=2, padding="same", use_bias=False,
                                        kernel_regularizer=l1_l2, activity_regularizer=l1)(h)
    h = InstanceNormalization()(h)
    h = tf.keras.layers.ReLU()(h)   # [512, 512, 32]

    h = tf.keras.layers.ZeroPadding2D((3,3))(h)
    h = tf.keras.layers.Conv2D(filters=3, kernel_size=7, strides=1, padding="valid")(h)
    h = tf.image.resize(h, [256, 256])
    h = tf.nn.tanh(h)

    return tf.keras.Model(inputs=[inputs, targets], outputs=[h, h_mid])

def F2M_discriminator(input_shape=(256, 256, 3)):

    h = inputs = tf.keras.Input(input_shape)

    h = tf.keras.layers.Conv2D(filters=64, kernel_size=4, strides=2, padding="same", use_bias=False,
                               kernel_regularizer=l1_l2, activity_regularizer=l1)(h)
    h = InstanceNormalization()(h)
    h = tf.keras.layers.LeakyReLU(alpha=0.2)(h)

    h = tf.keras.layers.Conv2D(filters=128, kernel_size=4, strides=2, padding="same", use_bias=False,
                               kernel_regularizer=l1_l2, activity_regularizer=l1)(h)
    h = InstanceNormalization()(h)
    h = tf.keras.layers.LeakyReLU(alpha=0.2)(h)

    h = tf.keras.layers.Conv2D(filters=256, kernel_size=4, strides=2, padding="same", use_bias=False,
                               kernel_regularizer=l1_l2, activity_regularizer=l1)(h)
    h = InstanceNormalization()(h)
    h = tf.keras.layers.LeakyReLU(alpha=0.2)(h)

    h = tf.keras.layers.Conv2D(filters=512, kernel_size=4, strides=1, padding="same", use_bias=False,
                               kernel_regularizer=l1_l2, activity_regularizer=l1)(h)
    h = InstanceNormalization()(h)
    h = tf.keras.layers.LeakyReLU(alpha=0.2)(h)

    h = tf.keras.layers.Conv2D(filters=1, kernel_size=4, strides=1, padding="same")(h)

    return tf.keras.Model(inputs=inputs, outputs=h)