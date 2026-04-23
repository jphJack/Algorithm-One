import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class LightweightBackbone(nn.Module):
    def __init__(self, in_channels=1, feature_dim=256):
        super(LightweightBackbone, self).__init__()
        
        self.stage1 = nn.Sequential(
            ConvBlock(in_channels, 32, kernel_size=3, stride=2, padding=1),
            ConvBlock(32, 32, kernel_size=3, stride=1, padding=1),
        )
        
        self.stage2 = nn.Sequential(
            ConvBlock(32, 64, kernel_size=3, stride=2, padding=1),
            ConvBlock(64, 64, kernel_size=3, stride=1, padding=1),
        )
        
        self.stage3 = nn.Sequential(
            ConvBlock(64, 128, kernel_size=3, stride=2, padding=1),
            ConvBlock(128, 128, kernel_size=3, stride=1, padding=1),
        )
        
        self.stage4 = nn.Sequential(
            ConvBlock(128, 256, kernel_size=3, stride=2, padding=1),
            ConvBlock(256, 256, kernel_size=3, stride=1, padding=1),
        )
        
        self.stage5 = nn.Sequential(
            ConvBlock(256, feature_dim, kernel_size=3, stride=2, padding=1),
            ConvBlock(feature_dim, feature_dim, kernel_size=3, stride=1, padding=1),
        )
        
        self.out_channels = feature_dim
    
    def forward(self, x):
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.stage5(x)
        return x


class DualStreamBackbone(nn.Module):
    def __init__(self, in_channels=1, feature_dim=256):
        super(DualStreamBackbone, self).__init__()
        
        self.print_backbone = LightweightBackbone(in_channels, feature_dim)
        self.vein_backbone = LightweightBackbone(in_channels, feature_dim)
    
    def forward(self, print_img, vein_img):
        print_feat = self.print_backbone(print_img)
        vein_feat = self.vein_backbone(vein_img)
        return print_feat, vein_feat


if __name__ == '__main__':
    model = DualStreamBackbone(in_channels=3, feature_dim=256)
    
    print_img = torch.randn(2, 3, 128, 128)
    vein_img = torch.randn(2, 3, 128, 128)
    
    print_feat, vein_feat = model(print_img, vein_img)
    print(f"掌纹特征形状: {print_feat.shape}")
    print(f"掌静脉特征形状: {vein_feat.shape}")
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"总参数量: {total_params:,}")
