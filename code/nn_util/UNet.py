import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    """
    A module consisting of two convolutional layers each followed by batch normalization,
    ReLU activation, and dropout. This is used as a building block for the U-Net architecture.
    
    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        dropout_p (float): Dropout probability.
    """
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
    """
    A U-Net model for pupil segmentation in images. This model uses an encoder-decoder
    architecture with skip connections and dropout for regularization and Monte Carlo
    Dropout inference. The model is designed to perform binary segmentation of the pupil
    in eye images.
    
    Args:
        dropout_p (float): Dropout probability.
        
    Methods:
        forward(x): Performs a forward pass through the network.
        enable_dropout(): Activates Dropout Layers for Monte Carlo Inference.
        mc_consistency_loss(image, num_samples=10, percentage=0.1): Computes the
            consistency loss for Monte Carlo Dropout inference.
    """
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
        
        # Final 1x1 Convolution for binary segmentation
        return torch.sigmoid(self.final_conv(x))

    def enable_dropout(self) -> None:
        """Activates Dropout Layers for Monte Carlo Inference"""
        for m in self.modules():
            if isinstance(m, nn.Dropout2d):
                m.train()

        
    def mc_consistency_loss(self, image, num_samples=10, percentage=0.1) -> torch.Tensor:
        """
        Computes the consistency loss for Monte Carlo Dropout inference.
        
        Args:
            image (torch.Tensor): The input image tensor for which the consistency loss is computed.
            num_samples (int): The number of Monte Carlo samples to draw. Default is 10.
            percentage (float): The percentage of pixels with the highest variance to consider for the loss. Default is 0.1.
        
        Returns:
            torch.Tensor: The computed consistency loss.
        """
        self.train()
        self.enable_dropout()

        predictions = []
        with torch.enable_grad():
            for _ in range(num_samples):
                pred = self(image)
                predictions.append(pred)
        predictions = torch.stack(predictions)

        variance = torch.var(predictions, dim=0)
        
        top_variances = torch.topk(variance.flatten(), int((144*144) * percentage))
        loss = torch.mean(top_variances.values)
        
        return loss

def load_model(model_path, dropout_p: float = 0.1) -> PupilSegmentationUNet:
    """
    Loads a PupilSegmentationUNet model from a specified file path.
    
    Args:
        model_path (str): Path to the file containing the model state dictionary.
        dropout_p (float): Dropout probability to be used in the model. Default is 0.1.
        
    Returns:
        PupilSegmentationUNet: The loaded model with the specified dropout probability.
    """
    model = PupilSegmentationUNet(dropout_p=dropout_p)
    model.load_state_dict(torch.load(model_path))
    return model