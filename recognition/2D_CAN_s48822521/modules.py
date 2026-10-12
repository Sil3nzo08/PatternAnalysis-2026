"""
PURPOSE: Containing the source code of the components of your model. Each component must be implemented as a class or a function in PyTorch or TensorFlow/Keras. Your implementation of the model should not depend on NumPy (or other Python libraries not part of PyTorch / TensorFlow) in any way unless otherwise approved by the teaching staff.
"""

"""
U-Net Architecture:


The network architecture is illustrated in Figure 1. It consists of a contracting
path (left side) and an expansive path (right side). The contracting path follows
the typical architecture of a convolutional network. It consists of the repeated
application of two 3x3 convolutions (unpadded convolutions), each followed by
a rectified linear unit (ReLU) and a 2x2 max pooling operation with stride 2
for downsampling. At each downsampling step we double the number of feature
channels. Every step in the expansive path consists of an upsampling of the
feature map followed by a 2x2 convolution (“up-convolution”) that halves the
number of feature channels, a concatenation with the correspondingly cropped
feature map from the contracting path, and two 3x3 convolutions, each followed by a ReLU. The cropping is necessary due to the loss of border pixels
in every convolution. At the final layer a 1x1 convolution is used to map each
64-component feature vector to the desired number of classes. In total the network has 23 convolutional layers.
To allow a seamless tiling of the output segmentation map (see Figure 2), it is
important to select the input tile size such that all 2x2 max-pooling operations
are applied to a layer with an even x- and y-size.


"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    """
    One block: 2x (3x3 convolutions + RELU)
    Normalisation added to perhaps make training more stable
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1), # padding=1 to preserve spatial size
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)


class DownBlock(nn.Module):
    """
    Max pool to downsample, then DoubleConv    
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.max_pool = nn.MaxPool2d(2) # 2x2 max-pool
        self.conv_block = DoubleConv(in_channels, out_channels)

    def forward(self, x):
        return self.conv_block(self.max_pool(x))


class BasicUNet2D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        # Encoder stuff
        self.start = DoubleConv(1, 32) # No max-pool at start
        self.down_layer1 = DownBlock(32, 64) 
        self.down_layer2 = DownBlock(64, 128)
        self.down_layer3 = DownBlock(128, 256)
        self.down_layer4 = DownBlock(256, 512) # At bottle-neck now