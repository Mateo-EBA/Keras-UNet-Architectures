#%% --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
# Imports
from tensorflow import keras
import keras.backend as K
from keras.layers import (
    Input,
    Conv2D,
    MaxPooling2D,
    Dropout,
    Conv2DTranspose,
    UpSampling2D,
    BatchNormalization,
    add,
    multiply,
    Activation,
    Lambda
)

#%% --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
# Shorthands
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

#%% --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
# create_attention_unet_2D
def create_attention_unet_2D(input_shape:tuple|list,
                             levels:int=3,
                             convs_per_level:int=2,
                             start_features:int=32,
                             dropout:float=0.5,
                             output_activation:str='sigmoid',
                             model_name:str='AttentionUNet2D',
                             verbose:int=1) -> keras.Model:
    """
    Create a 2D Attention U-Net model.

    Args:
        input_shape (tuple | list): The shape of the input tensor.
        levels (int, optional): The number of levels in the U-Net architecture. Defaults to 3.
        convs_per_level (int, optional): The number of convolutional layers per level. Defaults to 2.
        start_features (int, optional): The number of features in the first convolutional layer. Defaults to 32.
        dropout (float, optional): The dropout rate. Defaults to 0.5.
        output_activation (str, optional): The activation function for the output layer.
            Defaults to "sigmoid" ('softmax' for multiclass).
        model_name (str, optional): The name of the model. Defaults to 'AttentionUNet2D'.
        verbose (int, optional): Verbosity level. Defaults to 1.

    Returns:
        keras.Model: The 2D Attention U-Net model.
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
            conv = Conv2D(start_features * (2**i), (3,3), activation='relu', padding='same')(conv)
        convs.append(Conv2D(start_features * (2**i), (3,3), activation='relu', padding='same')(conv))
        next_input = MaxPooling2D((2, 2))(convs[i])
    next_input = Dropout(dropout)(next_input)
    
    # Bottle Neck
    next_input = Conv2D(start_features * (2**levels), (3,3), activation='relu', padding='same')(next_input)
    next_input = Conv2D(start_features * (2**levels), (3,3), activation='relu', padding='same')(next_input)
    next_input = Dropout(dropout)(next_input)
    
    # Decoder
    for i in range(levels-1,-1,-1):
        deconv = Conv2DTranspose(start_features * (2**i), (3,3), strides=(2,2), padding='same')(next_input)
        uconv = AttentionGatingBlock(convs[i], deconv, start_features * (2**(i+1)))
        for _ in range(convs_per_level):
            uconv = Conv2D(start_features * (2**i), (3,3), activation='relu', padding='same')(uconv)
        next_input = uconv
    next_input = Dropout(dropout)(next_input)
    
    # Output
    output_layer = Conv2D(1, (1, 1), padding='same', activation=output_activation)(next_input)
    
    # Model
    model = keras.Model(inputs=X_input, outputs=output_layer, name=model_name)

    return model



