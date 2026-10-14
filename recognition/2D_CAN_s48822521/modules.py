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

class UpBlock(nn.Module):
    """
    2x2 Up-convolution, then DoubleConv
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up_conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2) # Stride = 2 to double spatial size
        self.conv_block = DoubleConv(out_channels * 2, out_channels)

    def forward(self, x, skip_connection):
        x = self.up_conv(x)             # (B, out_channels, 2H, 2W)

        # From diagram, looks like the copied info is first, then we concatenate current info, x, onto its end. Same channels as x after up-convolution is applied, so we have double the out_channels.
        x = torch.cat([skip_connection, x], dim=1) # (B, out_channels*2, 2H, 2W)

        return self.conv_block(x)   # (B, out_channels, 2H, 2W)


class BasicUNet2D(nn.Module):
    def __init__(self, in_channels, base_channels, num_classes):
        super().__init__()

        # Encoder stuff
        self.down_layer1 = DoubleConv(in_channels, base_channels) # No max-pool at start
        self.down_layer2 = DownBlock(base_channels, base_channels*2) 
        self.down_layer3 = DownBlock(base_channels*2, base_channels*4)
        self.down_layer4 = DownBlock(base_channels*4, base_channels*8)

        self.down_layer_bottleneck = DownBlock(base_channels*8, base_channels*16) # At bottle-neck now

        # Decoder stuff
        self.up_layer4 = UpBlock(base_channels*16, base_channels*8)
        self.up_layer3 = UpBlock(base_channels*8, base_channels*4)
        self.up_layer2 = UpBlock(base_channels*4, base_channels*2)
        self.up_layer1 = UpBlock(base_channels*2, base_channels)

        # Final 1x1 convolution
        self.final_conv = nn.Conv2d(base_channels, num_classes, kernel_size=1)

    def forward(self, x):
        # Encoder
        skip_connection1 = self.down_layer1(x)
        skip_connection2 = self.down_layer2(skip_connection1)
        skip_connection3 = self.down_layer3(skip_connection2)
        skip_connection4 = self.down_layer4(skip_connection3)

        bottleneck = self.down_layer_bottleneck(skip_connection4)

        # Decoder
        dec4 = self.up_layer4(bottleneck, skip_connection4)
        dec3 = self.up_layer3(dec4, skip_connection3)
        dec2 = self.up_layer2(dec3, skip_connection2)
        dec1 = self.up_layer1(dec2, skip_connection1)

        return self.final_conv(dec1)


if __name__ == "__main__":
    model = BasicUNet2D(in_channels=1, base_channels=32, num_classes=6)
    x = torch.randn(2, 1, 256, 128)
    y = model(x)
    print(y.shape)                                     # torch.Size([2, 6, 256, 128])
    print(sum(p.numel() for p in model.parameters()))  # roughly 7.8M at base 32