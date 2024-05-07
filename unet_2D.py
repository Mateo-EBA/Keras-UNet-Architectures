#%% --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
# Imports
from tensorflow import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Dropout, Conv2DTranspose, concatenate

#%% --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
# create_unet_2D
def create_unet_2D(input_shape:tuple|list,
                   levels:int=3,
                   convs_per_level:int=2,
                   start_features:int=32,
                   dropout:float=0.5,
                   output_activation:str='sigmoid',
                   model_name:str='UNet2D',
                   verbose:int=1) -> keras.Model:
    """
    Create a 2D U-Net model.

    Args:
        input_shape (tuple | list): The shape of the input tensor.
        levels (int): The number of levels in the U-Net architecture.
        convs_per_level (int): The number of convolutional layers per level.
        start_features (int): The number of features in the first convolutional layer.
        dropout (float, optional): The dropout rate. Defaults to 0.5.
        output_activation (str, optional): The activation function for the output layer.
            Defaults to 'sigmoid'. Use 'softmax' for multiclass.
        model_name (str, optional): The name of the model. Defaults to 'UNet2D'.
        verbose (int, optional): Verbosity level. Defaults to 1.

    Returns:
        keras.Model: The 2D U-Net model.
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
    convm = Conv2D(start_features * (2**levels), (3,3), activation='relu', padding='same')(next_input)
    next_input = Conv2D(start_features * (2**levels), (3,3), activation='relu', padding='same')(convm)
    next_input = Dropout(dropout)(next_input)
    
    # Decoder
    for i in range(levels-1,-1,-1):
        deconv = Conv2DTranspose(start_features * (2**i), (3,3), strides=(2,2), padding='same')(next_input)
        uconv = concatenate([ deconv, convs[i] ])
        for _ in range(convs_per_level):
            uconv = Conv2D(start_features * (2**i), (3,3), activation='relu', padding='same')(uconv)
        next_input = uconv
    next_input = Dropout(dropout)(next_input)
    
    # Output
    output_layer = Conv2D(1, (1, 1), padding='same', activation=output_activation)(next_input)
    
    # Model
    model = keras.Model(inputs=X_input, outputs=output_layer, name=model_name)

    return model

