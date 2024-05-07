# -*- coding: utf-8 -*-
"""
Created on Wed Dec  7 10:25:06 2022

@author: Mateo
"""
#%% Imports
from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Dropout, Conv2DTranspose, concatenate

#%% create_unet_2D_MC
def create_unet_2D_MC(input_shape, num_classes, levels=3, convs_per_level=1, start_features=16, dropout=0.5, model_name = 'UNet2DMC'):
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
    output_layer = Conv2D(num_classes, (1, 1), padding = 'same', activation = 'softmax')(next_input)
    
    # Model
    model = Model(inputs = X_input, outputs = output_layer, name = model_name)

    return model


