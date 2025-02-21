import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, dropout_p=0.1):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=dropout_p),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=dropout_p)
        )
    
    def forward(self, x):
        return self.double_conv(x)

class PupilSegmentationUNet(nn.Module):
    def __init__(self, dropout_p=0.1):
        super().__init__()
        self.dropout_p = dropout_p
        
        # Encoder Path
        self.conv_down1 = DoubleConv(1, 64, dropout_p)
        self.conv_down2 = DoubleConv(64, 128, dropout_p)
        self.conv_down3 = DoubleConv(128, 256, dropout_p)
        self.conv_down4 = DoubleConv(256, 512, dropout_p)
        
        # Bottleneck
        self.bottleneck = DoubleConv(512, 1024, dropout_p)
        
        # Decoder Path
        self.up1 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.conv_up1 = DoubleConv(1024, 512, dropout_p)
        
        self.up2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.conv_up2 = DoubleConv(512, 256, dropout_p)
        
        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv_up3 = DoubleConv(256, 128, dropout_p)
        
        self.up4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv_up4 = DoubleConv(128, 64, dropout_p)
        
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

    def enable_dropout(self):
        """Aktiviert Dropout-Layer auch während der Inferenz"""
        for m in self.modules():
            if isinstance(m, nn.Dropout2d):
                m.train()

    def monte_carlo_inference(self, x: torch.Tensor, num_samples: int = 30) -> dict:
        """
        Führt Monte Carlo Dropout Inferenz durch.
        
        Args:
            x (torch.Tensor): Eingabebild
            num_samples (int): Anzahl der Monte Carlo Samples
            
        Returns:
            dict: Dictionary mit mean_prediction, std_prediction und entropy
        """
        self.eval()  # Modell in Evaluierungsmodus setzen
        self.enable_dropout()  # Dropout während Inferenz aktivieren
        
        predictions = []
        with torch.no_grad():
            for _ in range(num_samples):
                pred = self.forward(x)
                predictions.append(pred)
                
        # Stack predictions und berechne Statistiken
        predictions = torch.stack(predictions)
        mean_prediction = torch.mean(predictions, dim=0)
        std_prediction = torch.std(predictions, dim=0)
        
        # Berechne Entropy als Unsicherheitsmaß
        # Clip values to avoid log(0)
        epsilon = 1e-7
        mean_prediction_clipped = torch.clamp(mean_prediction, epsilon, 1-epsilon)
        entropy = -(mean_prediction_clipped * torch.log(mean_prediction_clipped) + 
                   (1 - mean_prediction_clipped) * torch.log(1 - mean_prediction_clipped))
        
        return {
            'mean_prediction': mean_prediction,
            'std_prediction': std_prediction,
            'entropy': entropy
        }

def load_model(model_path: str = "models/host_model_unet.pth", dropout_p: float = 0.1):
    model = PupilSegmentationUNet(dropout_p=dropout_p)
    model.load_state_dict(torch.load(model_path))
    return model