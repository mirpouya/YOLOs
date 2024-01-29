import torch
import torch.nn as nn

# model architecture
"""
architecture config:
(kernel_size, filters, stride, padding)
'M' is maxpooling with stride 2*2 and kernel 2*2
"""

architecture_config = [
    (7, 64, 2, 3),
    "M",
    (3, 192, 1, 1),
    "M",
    (1, 128, 1, 0),
    (3, 256, 1, 1),
    (1, 256, 1, 0),
    (3, 512, 1, 1),
    "M",
    [(1, 256, 1, 0), (3, 512, 1, 1), 4],
    (1, 512, 1, 0),
    (3, 1024, 1, 1),
    "M",
    [(1, 512, 1, 0), (3, 1024, 1, 1), 2],
    (3, 1024, 1, 1),
    (3, 1024, 2, 1),
    (3, 1024, 1, 1),
    (3, 1024, 1, 1),
]

# building convolution neural network: Conv, BatchNorm, Relu
class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(CNNBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.batchnorm_2d = nn.BatchNorm2d(out_channels)
        self.leakyrelu = nn.LeakyReLU(0.1)

    def forward(self, x):
        return self.leakyrelu(self.batchnorm_2d(self.conv(x)))



class Yolov1(nn.Module):
    def __init__(self, in_channels = 3, **kwargs):
        super(Yolov1, self).__init__()
        self.architecture = architecture_config
        self.in_channels = in_channels
        self.darknet = self._create_conv_layers(self.architecture)
        self.fcl = self._create_fcl(**kwargs)

    def forward(self, x):
        x = self.darknet(x)
        return self.fcl(torch.flatten(x, start_dim=1))  # not to flatten examples
    

    def _create_conv_layers(self, architecture):
        layers = []
        in_channels = self.in_channels

        for x in architecture:
            # x -> (kernel_size, filters, stride, padding)
            if type(x) == tuple:
                layers += [
                    CNNBlock(
                        in_channels, x[1], kernel_size=x[0], stride=x[2], padding=x[3]
                    )
                ]
                in_channels = x[1]

            elif type(x) == str:
                layers += [
                    nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
                ]

            elif type(x) == list:
                conv1 = x[0]
                conv2 = x[1]
                repeat = x[2]

                for _ in range(repeat):
                    layers += [
                        CNNBlock(
                            in_channels, out_channels=conv1[1], kernel_size=conv1[0], stride=conv1[2], padding=conv1[3]
                        )
                    ]

                    layers += [
                        CNNBlock(
                            in_channels=conv1[1], out_channels=conv2[1], kernel_size=conv2[0], stride=conv2[2], padding=conv2[3]
                        )
                    ]




