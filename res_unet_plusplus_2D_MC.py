# -*- coding: utf-8 -*-
"""
Created on Wed Dec  7 10:25:06 2022

@author: Mateo
"""

#%% Imports
from tensorflow import keras
import tensorflow.keras.backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Conv2DTranspose, concatenate
from tensorflow.keras.layers import GlobalAveragePooling2D, Reshape, Dense, Multiply, multiply
from tensorflow.keras.layers import BatchNormalization, Activation, Add, add
from tensorflow.keras.layers import Lambda, UpSampling2D

#%% Blocks
# Inspired from https://github.com/DebeshJha/ResUNetPlusPlus/blob/0d64ce906acb2876c45c2ed7097dbe0a8aadae07/m_resunet.py#L1

def SqueezeExciteBlock(x, ratio=8):
    channel_axis = -1
    filters = x.shape[channel_axis]
    se_shape = (1, 1, filters)
    
    se = GlobalAveragePooling2D()(x)
    se = Reshape(se_shape)(se)
    se = Dense(filters // ratio, activation='relu', kernel_initializer='he_normal', use_bias=False)(se)
    se = Dense(filters, activation='sigmoid', kernel_initializer='he_normal', use_bias=False)(se)
    
    return Multiply()([x, se])

def StemBlock(x, filters, kernel_size=(3, 3), strides=1):
    # Conv 1
    conv = Conv2D(filters, kernel_size, padding='same', strides=strides)(x)
    conv = BatchNormalization()(conv)
    conv = Activation('relu')(conv)
    conv = Conv2D(filters, kernel_size, padding='same')(conv)

    # Shortcut
    s  = Conv2D(filters, (1, 1), padding='same', strides=strides)(x)
    s = BatchNormalization()(s)

    # Add
    conv = Add()([conv, s])
    return SqueezeExciteBlock(conv)

def ConvolutionBlock(x, filters, kernel_size=(3, 3), padding='same', strides=1):
    conv = BatchNormalization()(x)
    conv = Activation('relu')(conv)
    conv = Conv2D(filters, kernel_size, padding=padding, strides=strides)(conv)
    return conv

def ResidualBlock(x, filters, kernel_size=(3, 3), padding='same', strides=1):
    res = x
    res = ConvolutionBlock(res, filters, kernel_size=kernel_size, padding=padding, strides=strides)
    res = ConvolutionBlock(res, filters, kernel_size=kernel_size, padding=padding, strides=1)
    
    shortcut = Conv2D(filters, kernel_size=(1, 1), padding=padding, strides=strides)(x)
    shortcut = BatchNormalization()(shortcut)
    
    out = Add()([shortcut, res])
    return SqueezeExciteBlock(out)

def ASPPBlock(x, filters, kernel_size=(3, 3), padding='same', rate_scale=1):
    x1 = Conv2D(filters, kernel_size, dilation_rate=(6 * rate_scale, 6 * rate_scale), padding=padding)(x)
    x1 = BatchNormalization()(x1)

    x2 = Conv2D(filters, kernel_size, dilation_rate=(12 * rate_scale, 12 * rate_scale), padding=padding)(x)
    x2 = BatchNormalization()(x2)

    x3 = Conv2D(filters, kernel_size, dilation_rate=(18 * rate_scale, 18 * rate_scale), padding=padding)(x)
    x3 = BatchNormalization()(x3)

    x4 = Conv2D(filters, kernel_size, padding="same")(x)
    x4 = BatchNormalization()(x4)

    y = Add()([x1, x2, x3, x4])
    return Conv2D(filters, (1, 1), padding="same")(y)

def ExpandAs(tensor, rep):
    # Anonymous lambda function to expand the specified axis by a factor of argument, rep.
    # If tensor has shape (512,512,N), lambda will return a tensor of shape (512,512,N*rep), if specified axis=2
    my_repeat = Lambda(lambda x, repnum: K.repeat_elements(x, repnum, axis=3),
                       arguments={'repnum': rep})(tensor)
    return my_repeat

def AttentionGatingBlock(x, g, inter_shape):
    shape_x = K.int_shape(x)
    shape_g = K.int_shape(g)

    # Getting the gating signal to the same number of filters as the inter_shape
    phi_g = Conv2D(filters=inter_shape,
                   kernel_size=1,
                   strides=1,
                   padding='same')(g)

    # Getting the x signal to the same shape as the gating signal
    theta_x = Conv2D(filters=inter_shape,
                     kernel_size=3,
                     strides=(shape_x[1] // shape_g[1],
                              shape_x[2] // shape_g[2]),
                     padding='same')(x)

    # Element-wise addition of the gating and x signals
    add_xg = add([phi_g, theta_x])
    add_xg = Activation('relu')(add_xg)

    # 1x1x1 convolution
    psi = Conv2D(filters=1, kernel_size=1, padding='same')(add_xg)
    psi = Activation('sigmoid')(psi)
    shape_sigmoid = K.int_shape(psi)

    # Upsampling psi back to the original dimensions of x signal
    upsample_sigmoid_xg = UpSampling2D(size=(shape_x[1] // shape_sigmoid[1],
                                             shape_x[2] //
                                             shape_sigmoid[2]))(psi)

    # Expanding the filter axis to the number of filters in the original x signal
    upsample_sigmoid_xg = ExpandAs(upsample_sigmoid_xg, shape_x[3])

    # Element-wise multiplication of attention coefficients back onto original x signal
    attn_coefficients = multiply([upsample_sigmoid_xg, x])

    # Final 1x1x1 convolution to consolidate attention signal to original x dimensions
    output = Conv2D(filters=shape_x[3],
                    kernel_size=1,
                    strides=1,
                    padding='same')(attn_coefficients)
    output = BatchNormalization()(output)
    return output

#%% create_res_unet_plusplus_2D_MC
def create_res_unet_plusplus_2D_MC(input_shape, num_classes, levels=3, convs_per_level=1, start_features=16, model_name = 'ResUNetPlusPlus2D'):
    keras.backend.clear_session()
    
    convs = []
    
    # Input placeholder
    X_input = Input(input_shape)
    next_input = StemBlock(X_input, start_features)
    
    # Contracting Path
    for i in range(levels):
        conv = next_input
        for j in range(convs_per_level-1):
            conv = ResidualBlock(conv, start_features * (2**i), (3,3), padding='same')
        convs.append(ResidualBlock(conv, start_features * (2**i), (3,3), padding='same'))
        next_input = MaxPooling2D((2, 2))(convs[i])
    
    # Bridge
    next_input = ASPPBlock(next_input, start_features * (2**levels))
    
    # Expansive Path
    #for i in range(levels-1,-1,0):
    for i in range(levels-1):
        uconv = AttentionGatingBlock(convs[levels-i-1], next_input, start_features * (2**(levels-i)))
        deconv = Conv2DTranspose(start_features * (2**levels-i-1), (3,3), strides=(2,2), padding='same')(uconv)
        uconv = concatenate([ deconv, convs[levels-i-2] ])
        for j in range(convs_per_level):
            uconv = ResidualBlock(uconv, start_features * (2**levels-i-1), (3,3), padding='same')
        next_input = uconv
    
    # Output
    output = ASPPBlock(next_input, start_features)
    output = Conv2D(num_classes, (1, 1), padding = 'same', activation = 'softmax')(output)
    
    # Model
    return Model(inputs=X_input, outputs=output, name=model_name)
