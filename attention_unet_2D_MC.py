# -*- coding: utf-8 -*-
"""
Created on Wed Dec  7 10:25:06 2022

@author: Mateo
"""
#%% Imports
from tensorflow import keras
import tensorflow.keras.backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Conv2DTranspose
from tensorflow.keras.layers import UpSampling2D, BatchNormalization, add, multiply, Activation, Lambda

#%% AttnGatingBlock Stuff
# Inspired from https://towardsdatascience.com/a-detailed-explanation-of-the-attention-u-net-b371a5590831

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

#%% create_attention_unet_2D_MC
def create_attention_unet_2D_MC(input_shape, num_classes, levels=3, convs_per_level=1, start_features=16, model_name = 'AttentionUNet2DMC'):
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
    
    # Bottle Neck
    #next_input = Dropout(0.5)(next_input)
    next_input = Conv2D(start_features * (2**levels), (3,3), activation = "relu", padding = "same")(next_input)
    next_input = Conv2D(start_features * (2**levels), (3,3), activation = "relu", padding = "same")(next_input)
    #next_input = Dropout(0.5)(next_input)
    
    # Expansive Path
    for i in range(levels-1,-1,-1):
        deconv = Conv2DTranspose(start_features * (2**i), (3,3), strides = (2,2), padding = "same")(next_input)
        uconv = AttentionGatingBlock(convs[i], deconv, start_features * (2**(i+1)))
        for j in range(convs_per_level):
            uconv = Conv2D(start_features * (2**i), (3,3), activation = "relu", padding = "same")(uconv)
        next_input = uconv
    
    # Output
    output_layer = Conv2D(num_classes, (1, 1), padding = 'same', activation = 'softmax')(next_input)
    
    # Model
    model = Model(inputs = X_input, outputs = output_layer, name = model_name)

    return model


