# -*- coding: utf-8 -*-
"""
Created on Fri Sep  9 09:27:14 2022

@author: Mateo
"""

#%% Imports
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Concatenate, Conv3D, Conv3DTranspose, ReLU, Softmax, BatchNormalization

#%% Shorthands

def Conv_Batch_ReLU(inputData, out_channels, kernel_size, strides, padding):
    conv = Conv3D(filters=out_channels, kernel_size=kernel_size, strides=strides, padding=padding)(inputData)
    conv = BatchNormalization()(conv)
    conv = ReLU()(conv)
    return conv

def UpConv_Batch_ReLU(inputData, out_channels, kernel_size, strides, padding):
    conv = Conv3DTranspose(filters=out_channels, kernel_size=kernel_size, strides=strides, padding=padding)(inputData)
    conv = BatchNormalization()(conv)
    conv = ReLU()(conv)
    return conv

#%% Layer Blocks

def InitialConv(inputData, out_channels=16):
    conv = Conv3D(filters=out_channels, kernel_size=5, strides=1, padding='same')(inputData)
    conv = BatchNormalization()(conv)
    conv = ReLU()(conv)
    
    convDown = Conv_Batch_ReLU(conv, out_channels * 2, kernel_size=2, strides=2, padding='valid')
    
    return conv, convDown
    
def DownConvBlock2b(inputData, out_channels=32):
    conv = Conv_Batch_ReLU(inputData, out_channels, kernel_size=5, strides=1, padding='same')
    conv = Conv_Batch_ReLU(conv, out_channels, kernel_size=5, strides=1, padding='same')
    convDown = Conv_Batch_ReLU(conv, out_channels * 2, kernel_size=2, strides=2, padding='valid')
    return conv, convDown

def DownConvBlock3b(inputData, out_channels=64):
    conv = Conv_Batch_ReLU(inputData, out_channels, kernel_size=5, strides=1, padding='same')
    conv = Conv_Batch_ReLU(conv, out_channels, kernel_size=5, strides=1, padding='same')
    conv = Conv_Batch_ReLU(conv, out_channels, kernel_size=5, strides=1, padding='same')
    convDown = Conv_Batch_ReLU(conv, out_channels * 2, kernel_size=2, strides=2, padding='valid')
    return conv, convDown

def UpConvBlock3b(inputData, out_channels=256, undersampling_factor=2):
    conv = Conv_Batch_ReLU(inputData, out_channels, kernel_size=5, strides=1, padding='same')
    conv = Conv_Batch_ReLU(conv, out_channels, kernel_size=5, strides=1, padding='same')
    conv = Conv_Batch_ReLU(conv, out_channels, kernel_size=5, strides=1, padding='same')
    conv = UpConv_Batch_ReLU(conv, out_channels // undersampling_factor, kernel_size=2, strides=2, padding='valid')
    return conv

def UpConvBlock2b(inputData, out_channels=256, undersampling_factor=2):
    conv = Conv_Batch_ReLU(inputData, out_channels, kernel_size=5, strides=1, padding='same')
    conv = Conv_Batch_ReLU(conv, out_channels, kernel_size=5, strides=1, padding='same')
    conv = UpConv_Batch_ReLU(conv, out_channels // undersampling_factor, kernel_size=2, strides=2, padding='valid')
    return conv

def FinalConv(inputData, num_outs=2, out_channels=32):
    conv = Conv_Batch_ReLU(inputData, out_channels, kernel_size=5, strides=1, padding='same')
    
    conv = Conv3D(filters=num_outs, kernel_size=1, strides=1, padding='valid')(conv)
    conv = BatchNormalization()(conv)
    conv = Softmax()(conv)
    
    return conv

#%% create_vnet_3D
def create_vnet_3D(inputShape, numOuts=2, channels=16, name='VNet3D'):
    
    # Input placeholder
    inputData = Input(inputShape)
    
    # Initial Conv
    cConv1, cConv1Down = InitialConv(inputData, out_channels=channels)
    
    # Contracting Path
    cConv2, cConv2Down = DownConvBlock2b(cConv1Down, out_channels=channels * 2)
    cConv3, cConv3Down = DownConvBlock3b(cConv2Down, out_channels=channels * 4)
    cConv4, cConv4Down = DownConvBlock3b(cConv3Down, out_channels=channels * 8)
    
    # Expanding Path (3b, 3b, 3b, 2b)
    eConv4Up = UpConvBlock3b(cConv4Down, out_channels=channels * 16, undersampling_factor=2)
    cat4 = Concatenate()([ eConv4Up, cConv4 ])
    
    eConv3Up = UpConvBlock3b(cat4, out_channels=channels * 16, undersampling_factor=4)
    cat3 = Concatenate()([ eConv3Up, cConv3 ])
    
    eConv2Up = UpConvBlock3b(cat3, out_channels=channels * 8, undersampling_factor=4)
    cat2 = Concatenate()([ eConv2Up, cConv2 ])

    eConv1Up = UpConvBlock3b(cat2, out_channels=channels * 4, undersampling_factor=4)
    cat1 = Concatenate()([ eConv1Up, cConv1 ])
    
    # Output
    outputData = FinalConv(cat1, num_outs=numOuts, out_channels=channels * 2)
    
    # Model
    model = Model(inputs=inputData, outputs=outputData, name=name)
    return model

