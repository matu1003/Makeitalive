import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # Use bilinear interpolation for upsampling (simpler and avoids artifacts compared to ConvTranspose)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # Handle cases where input size is not perfectly a power of 2
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
                        
        # Concatenate skip connection
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class MotionFlowUNet(nn.Module):
    """
    Réseau U-Net simple pour l'estimation de Motion Flow.
    Prend en entrée UNE image (3 canaux RGB) et prédit un flow 2D (2 canaux X, Y)
    pour animer cette image statique en générant le mouvement futur "naturel".
    """
    def __init__(self, in_channels=3, out_channels=2):
        super(MotionFlowUNet, self).__init__()
        
        # Encodeur (Descente)
        self.inc = DoubleConv(in_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        # Bottleneck
        self.down4 = Down(512, 1024)
        
        # Décodeur (Montée avec Skip Connections)
        self.up1 = Up(1024 + 512, 512)
        self.up2 = Up(512 + 256, 256)
        self.up3 = Up(256 + 128, 128)
        self.up4 = Up(128 + 64, 64)
        
        # Tête de prédiction finale (Flow en 2 canaux: dx, dy)
        self.outc = nn.Conv2d(64, out_channels, kernel_size=1)
        
    def forward(self, x):
        # Encodeur
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        
        # Décodeur
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        
        # Le flow final : valeurs typiquement petites (on peut ajouter du Tanh si on borne le flow, 
        # mais on laisse linaire pour l'instant (le réseau apprend les déplacements en pixels absolus ou relatifs))
        flow = self.outc(x)
        
        return flow

# Test rapide des dimensions
if __name__ == "__main__":
    net = MotionFlowUNet()
    # Batch de 2 images RGB, 256x256
    dummy_input = torch.randn(2, 3, 256, 256)
    out_flow = net(dummy_input)
    print(f"Input shape: {dummy_input.shape}") 
    print(f"Output Flow shape: {out_flow.shape} (2 canaux = DX, DY)")
