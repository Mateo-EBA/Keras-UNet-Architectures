# -*- coding: utf-8 -*-
"""
Created on Tue Oct 11 12:20:52 2022

@author: Mateo
"""

#%% Imports
from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Conv2DTranspose, concatenate
from tensorflow.keras.layers import BatchNormalization, Activation, Add

#%% Blocks
# Inspired from https://github.com/nikhilroxtomar/Deep-Residual-Unet/blob/master/Deep%20Residual%20UNet.ipynb

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
    
    return Add()([shortcut, res])

#%% create_res_unet_2D
def create_res_unet_2D(input_shape, levels, convs_per_level, start_features, model_name = 'ResUNet2D'):
    keras.backend.clear_session()
    
    convs = []
    
    # Input placeholder
    X_input = Input(input_shape)
    next_input = X_input
    
    # Contracting Path
    for i in range(levels):
        conv = next_input
        for j in range(convs_per_level-1):
            conv = ResidualBlock(conv, start_features * (2**i), (3,3), padding='same')
        convs.append(ResidualBlock(conv, start_features * (2**i), (3,3), padding='same'))
        next_input = MaxPooling2D((2, 2))(convs[i])
    
    # Bottle Neck
    #next_input = Dropout(0.5)(next_input)
    next_input = ResidualBlock(next_input, start_features * (2**levels), (3,3), padding='same')
    next_input = ResidualBlock(next_input, start_features * (2**levels), (3,3), padding='same')
    #next_input = Dropout(0.5)(next_input)
    
    # Expansive Path
    for i in range(levels-1,-1,-1):
        deconv = Conv2DTranspose(start_features * (2**i), (3,3), strides=(2,2), padding='same')(next_input)
        uconv = concatenate([ deconv, convs[i] ])
        for j in range(convs_per_level):
            uconv = ResidualBlock(uconv, start_features * (2**i), (3,3), padding='same')
        next_input = uconv
    
    # Output
    output_layer = Conv2D(1, (1, 1), padding = 'same', activation = 'sigmoid')(next_input)
    
    # Model
    return Model(inputs = X_input, outputs = output_layer, name = model_name)




