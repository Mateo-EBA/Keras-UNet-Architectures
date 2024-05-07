# -*- coding: utf-8 -*-
"""
Created on Wed Sep 21 10:48:54 2022

@author: Mateo
"""
#%% Imports
from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Dropout, Conv2DTranspose, concatenate

#%% create_unet_2D
def create_unet_2D(input_shape, levels, convs_per_level, start_features, dropout=0.5, model_name = 'UNet2D'):
    keras.backend.clear_session()
    
    convs = []
    
    # Input placeholder
    X_input = Input(input_shape)
    next_input = X_input
    
    # Contracting Path
    for i in range(levels):
        conv = next_input
        for j in range(convs_per_level-1):
            conv = Conv2D(start_features * (2**i), (3,3), activation = "relu", padding = "same")(conv)
        convs.append(Conv2D(start_features * (2**i), (3,3), activation = "relu", padding = "same")(conv))
        next_input = MaxPooling2D((2, 2))(convs[i])
    drop = Dropout(dropout)(next_input)
    
    # Bottle Neck
    convm = Conv2D(start_features * (2**levels), (3,3), activation = "relu", padding = "same")(drop)
    next_input = Conv2D(start_features * (2**levels), (3,3), activation = "relu", padding = "same")(convm)
    
    # Expansive Path
    for i in range(levels-1,-1,-1):
        deconv = Conv2DTranspose(start_features * (2**i), (3,3), strides = (2,2), padding = "same")(next_input)
        uconv = concatenate([ deconv, convs[i] ])
        for j in range(convs_per_level):
            uconv = Conv2D(start_features * (2**i), (3,3), activation = "relu", padding = "same")(uconv)
        if (i == (levels-1)):
            next_input = Dropout(dropout)(uconv)
        else:
            next_input = uconv
    
    # Output
    output_layer = Conv2D(1, (1, 1), padding = "same", activation = "sigmoid")(next_input)
    
    # Model
    model = Model(inputs = X_input, outputs = output_layer, name = model_name)

    return model







