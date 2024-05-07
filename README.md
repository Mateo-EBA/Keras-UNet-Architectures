# Keras UNet Architectures

This repository contains various U-Net architecture variants implemented in Keras for image segmentation tasks. These models provide different features and capabilities for building U-Net models quickly and efficiently.

## Models Included
1. **UNet2D**: Basic 2D U-Net model.
2. **ResUNet2D**: 2D Residual U-Net model.
3. **ResUNetPlusPlus2D**: 2D Residual U-Net++ model.
4. **AttentionUNet2D**: 2D Attention U-Net model.

## Usage
Each model is implemented as a function that can be easily called to create the corresponding architecture. The models offer flexibility in terms of levels, convolutional layers per level, starting features, dropout rate, output and activation.

## How to Use
1. Clone the repository.
2. Import the desired U-Net architecture function into your project.
3. Call the function with the appropriate parameters to create the model.

## Example
```python
from unet_models import create_unet_2D

# Create a 2D U-Net model
model = create_unet_2D(
    input_shape=(256, 256, 3),
    levels=4,
    convs_per_level=2,
    start_features=64,
    dropout=0.5,
    output_activation='sigmoid'
)
```