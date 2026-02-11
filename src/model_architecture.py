import torch
from torch import nn

class SeperableConv2d(nn.Module):
    """Depthwise Separable Convolution."""
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=1, bias=True):
        super(SeperableConv2d, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size, stride=stride, padding=padding, groups=in_channels, bias=bias)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)

    def forward(self, x):
        return self.pointwise(self.depthwise(x))

class ConvBlock(nn.Module):
    """Convolutional block with separable conv, batch norm, and activation."""
    def __init__(self, in_channels, out_channels, use_act=True, use_bn=True, discriminator=False, **kwargs):
        super(ConvBlock, self).__init__()
        self.use_act = use_act
        self.cnn = SeperableConv2d(in_channels, out_channels, **kwargs, bias=not use_bn)
        self.bn = nn.BatchNorm2d(out_channels) if use_bn else nn.Identity()
        self.act = nn.LeakyReLU(0.2, inplace=True) if discriminator else nn.PReLU(num_parameters=out_channels)
    def forward(self, x):
        return self.act(self.bn(self.cnn(x))) if self.use_act else self.bn(self.cnn(x))

class UpsampleBlock(nn.Module):
    """Upsample block with PixelShuffle."""
    def __init__(self, in_channels, scale_factor):
        super(UpsampleBlock, self).__init__()
        self.conv = SeperableConv2d(in_channels, in_channels * scale_factor**2, kernel_size=3, stride=1, padding=1)
        self.ps = nn.PixelShuffle(scale_factor)
        self.act = nn.PReLU(num_parameters=in_channels)
    def forward(self, x):
        return self.act(self.ps(self.conv(x)))

class ResidualBlock(nn.Module):
    """Residual block with two ConvBlocks."""
    def __init__(self, in_channels):
        super(ResidualBlock, self).__init__()
        self.block1 = ConvBlock(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
        self.block2 = ConvBlock(in_channels, in_channels, kernel_size=3, stride=1, padding=1, use_act=False)
    def forward(self, x):
        return self.block2(self.block1(x)) + x

class Generator(nn.Module):
    """SRGAN Generator."""
    def __init__(self, in_channels=3, num_channels=64, num_blocks=16, upscale_factor=4):
        super(Generator, self).__init__()
        self.initial = ConvBlock(in_channels, num_channels, kernel_size=9, stride=1, padding=4, use_bn=False)
        self.residual = nn.Sequential(*[ResidualBlock(num_channels) for _ in range(num_blocks)])
        self.convblock = ConvBlock(num_channels, num_channels, kernel_size=3, stride=1, padding=1, use_act=False)
        self.upsampler = nn.Sequential(*[UpsampleBlock(num_channels, scale_factor=2) for _ in range(upscale_factor//2)])
        self.final_conv = SeperableConv2d(num_channels, in_channels, kernel_size=9, stride=1, padding=4)
    def forward(self, x):
        initial = self.initial(x)
        x = self.residual(initial)
        x = self.convblock(x) + initial
        x = self.upsampler(x)
        return (torch.tanh(self.final_conv(x)) + 1) / 2

class Discriminator(nn.Module):
    """SRGAN Discriminator."""
    def __init__(self, in_channels=3, features=(64,64,128,128,256,256,512,512)):
        super(Discriminator, self).__init__()
        blocks = []
        for idx, feature in enumerate(features):
            blocks.append(
                ConvBlock(in_channels, feature, kernel_size=3, stride=1 + idx % 2, padding=1,
                          discriminator=True, use_act=True, use_bn=False if idx==0 else True)
            )
            in_channels = feature
        self.blocks = nn.Sequential(*blocks)
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((6,6)),
            nn.Flatten(),
            nn.Linear(512*6*6, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, 1)
        )
    def forward(self, x):
        return torch.sigmoid(self.classifier(self.blocks(x)))
