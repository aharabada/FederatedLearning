import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class PupilSegmentationUNet(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Encoder Path
        self.conv_down1 = DoubleConv(1, 64)
        self.conv_down2 = DoubleConv(64, 128)
        self.conv_down3 = DoubleConv(128, 256)
        self.conv_down4 = DoubleConv(256, 512)
        
        # Bottleneck
        self.bottleneck = DoubleConv(512, 1024)
        
        # Decoder Path
        self.up1 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.conv_up1 = DoubleConv(1024, 512)
        
        self.up2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.conv_up2 = DoubleConv(512, 256)
        
        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv_up3 = DoubleConv(256, 128)
        
        self.up4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv_up4 = DoubleConv(128, 64)
        
        # Final 1x1 Convolution für binäre Segmentierung
        self.final_conv = nn.Conv2d(64, 1, kernel_size=1)

    def forward(self, x):
        # Encoder Path mit Skip Connections
        conv1 = self.conv_down1(x)
        x = F.max_pool2d(conv1, 2)
        
        conv2 = self.conv_down2(x)
        x = F.max_pool2d(conv2, 2)
        
        conv3 = self.conv_down3(x)
        x = F.max_pool2d(conv3, 2)
        
        conv4 = self.conv_down4(x)
        x = F.max_pool2d(conv4, 2)
        
        # Bottleneck
        x = self.bottleneck(x)
        
        # Decoder Path mit Skip Connections
        x = self.up1(x)
        x = torch.cat([x, conv4], dim=1)
        x = self.conv_up1(x)
        
        x = self.up2(x)
        x = torch.cat([x, conv3], dim=1)
        x = self.conv_up2(x)
        
        x = self.up3(x)
        x = torch.cat([x, conv2], dim=1)
        x = self.conv_up3(x)
        
        x = self.up4(x)
        x = torch.cat([x, conv1], dim=1)
        x = self.conv_up4(x)
        
        # Finale 1x1 Convolution und Sigmoid für binäre Segmentierung
        return torch.sigmoid(self.final_conv(x))
    
    
    
def load_model(model_path: str = "models/host_model_unet.pth"):
    model = PupilSegmentationUNet()
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model
