import torch.nn as nn

class SegNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(SegNet, self).__init__()

        #encoder
        self.enc1 = self.conv_block(in_channels, 64)
        self.enc2 = self.conv_block(64, 128)
        self.enc3 = self.conv_block(128, 256)
        
        #decoder
        self.dec1 = self.conv_block(256, 128)
        self.dec2 = self.conv_block(128, 64)
        self.dec3 = self.conv_block(64, out_channels)
        
    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )

    def forward(self, x):
        #encoder path
        x1 = self.enc1(x)
        x2 = self.enc2(x1)
        x3 = self.enc3(x2)
        
        #decoder path
        x = self.dec1(x3)
        x = self.dec2(x)
        x = self.dec3(x)
        
        return x
