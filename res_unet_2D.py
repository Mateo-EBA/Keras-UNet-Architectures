#%% --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
# Imports
from tensorflow import keras
from keras.models import Model
from keras.layers import (
    Input,
    Conv2D,
    MaxPooling2D,
    Dropout,
    Conv2DTranspose,
    concatenate,
    BatchNormalization,
    Activation,
    Add
)

#%% --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
# Blocks
# Inspired from https://github.com/nikhilroxtomar/Deep-Residual-Unet/blob/master/Deep%20Residual%20UNet.ipynb
def ConvolutionBlock(x, filters:int, kernel_size:tuple|list=(3, 3), padding:str='same'):
    conv = BatchNormalization()(x)
    conv = Activation('relu')(conv)
    conv = Conv2D(filters, kernel_size, padding=padding, strides=1)(conv)
    return conv

def ResidualBlock(x, filters:int, kernel_size:tuple|list=(3, 3), padding:str='same'):
    res = ConvolutionBlock(x,   filters, kernel_size=kernel_size, padding=padding)
    res = ConvolutionBlock(res, filters, kernel_size=kernel_size, padding=padding)
    shortcut = Conv2D(filters, kernel_size=(1, 1), padding=padding, strides=1)(x)
    shortcut = BatchNormalization()(shortcut)
    return Add()([shortcut, res])

#%% --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
# create_res_unet_2D
def create_res_unet_2D(input_shape:tuple|list,
                       levels:int=3,
                       convs_per_level:int=2,
                       start_features:int=32,
                       dropout:float=0.5,
                       output_activation:str='sigmoid',
                       model_name:str='ResUNet2D',
                       verbose:int=1) -> keras.Model:
    """
    Create a 2D Residual U-Net model.

    Args:
        input_shape (tuple | list): The shape of the input tensor.
        levels (int, optional): The number of levels in the U-Net architecture. Defaults to 3.
        convs_per_level (int, optional): The number of convolutional layers per level. Defaults to 2.
        start_features (int, optional): The number of features in the first convolutional layer. Defaults to 32.
        dropout (float, optional): The dropout rate. Defaults to 0.5.
        output_activation (str, optional): The activation function for the output layer.
            Defaults to 'sigmoid' ('softmax' for multiclass).
        model_name (str, optional): The name of the model. Defaults to 'ResUNet2D'.
        verbose (int, optional): Verbosity level. Defaults to 1.

    Returns:
        keras.Model: The 2D Residual U-Net model.
    """
    keras.backend.clear_session()
    convs = []
    if len(input_shape) == 2:
        input_shape = (input_shape[0], input_shape[1], 1)
    if verbose > 0:
        print(f"Input image's dimensions must be multiple of 2^levels ({2**levels})")
    
    # Input placeholder
    X_input = Input(input_shape)
    next_input = X_input
    
    # Encoder
    for i in range(levels):
        conv = next_input
        for _ in range(convs_per_level-1):
            conv = ResidualBlock(conv, start_features * (2**i), (3,3), padding='same')
        convs.append(ResidualBlock(conv, start_features * (2**i), (3,3), padding='same'))
        next_input = MaxPooling2D((2, 2))(convs[i])
    next_input = Dropout(dropout)(next_input)
    
    # Bottle Neck
    next_input = ResidualBlock(next_input, start_features * (2**levels), (3,3), padding='same')
    next_input = ResidualBlock(next_input, start_features * (2**levels), (3,3), padding='same')
    next_input = Dropout(dropout)(next_input)
    
    # Decoder
    for i in range(levels-1,-1,-1):
        deconv = Conv2DTranspose(start_features * (2**i), (3,3), strides=(2,2), padding='same')(next_input)
        uconv = concatenate([ deconv, convs[i] ])
        for _ in range(convs_per_level):
            uconv = ResidualBlock(uconv, start_features * (2**i), (3,3), padding='same')
        next_input = uconv
    next_input = Dropout(dropout)(next_input)
    
    # Output
    output_layer = Conv2D(1, (1, 1), padding='same', activation=output_activation)(next_input)
    
    # Model
    return keras.Model(inputs=X_input, outputs=output_layer, name=model_name)




