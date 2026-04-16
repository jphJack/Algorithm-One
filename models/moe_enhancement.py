import torch
import torch.nn as nn
import torch.nn.functional as F


class DetailEnhanceBlock(nn.Module):
    def __init__(self, channels):
        super(DetailEnhanceBlock, self).__init__()
        
        self.depthwise = nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels, bias=False)
        self.pointwise = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        identity = x
        out = self.depthwise(x)
        out = self.pointwise(out)
        out = self.bn(out)
        out = self.relu(out)
        out = out + identity
        return out


class HighFreqExpert(nn.Module):
    def __init__(self, channels):
        super(HighFreqExpert, self).__init__()
        
        laplacian_kernel = torch.tensor([
            [0, 1, 0],
            [1, -4, 1],
            [0, 1, 0]
        ], dtype=torch.float32).view(1, 1, 3, 3)
        self.register_buffer('laplacian_kernel', laplacian_kernel)
        
        self.detail_enhance1 = DetailEnhanceBlock(channels)
        self.detail_enhance2 = DetailEnhanceBlock(channels)
        
        self.output_conv = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.output_bn = nn.BatchNorm2d(channels)
    
    def forward(self, x):
        B, C, H, W = x.shape
        
        laplacian_features = []
        for c in range(C):
            channel_feat = x[:, c:c+1, :, :]
            lap_feat = F.conv2d(channel_feat, self.laplacian_kernel, padding=1)
            laplacian_features.append(lap_feat)
        laplacian_feat = torch.cat(laplacian_features, dim=1)
        
        detail_feat = self.detail_enhance1(x)
        detail_feat = self.detail_enhance2(detail_feat)
        
        fused = laplacian_feat + detail_feat
        
        out = self.output_conv(fused)
        out = self.output_bn(out)
        
        return out


class MidFreqExpert(nn.Module):
    def __init__(self, channels):
        super(MidFreqExpert, self).__init__()
        
        self.branch1 = nn.Sequential(
            nn.Conv2d(channels, channels // 4, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels // 4),
            nn.ReLU(inplace=True)
        )
        
        self.branch2 = nn.Sequential(
            nn.Conv2d(channels, channels // 4, kernel_size=3, padding=2, dilation=2, bias=False),
            nn.BatchNorm2d(channels // 4),
            nn.ReLU(inplace=True)
        )
        
        self.branch3 = nn.Sequential(
            nn.Conv2d(channels, channels // 4, kernel_size=3, padding=4, dilation=4, bias=False),
            nn.BatchNorm2d(channels // 4),
            nn.ReLU(inplace=True)
        )
        
        self.aspp_fuse = nn.Sequential(
            nn.Conv2d(channels // 4 * 3, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )
        
        self.dir_conv1 = nn.Sequential(
            nn.Conv2d(channels, channels // 2, kernel_size=(1, 7), padding=(0, 3), bias=False),
            nn.BatchNorm2d(channels // 2),
            nn.ReLU(inplace=True)
        )
        
        self.dir_conv2 = nn.Sequential(
            nn.Conv2d(channels, channels // 2, kernel_size=(7, 1), padding=(3, 0), bias=False),
            nn.BatchNorm2d(channels // 2),
            nn.ReLU(inplace=True)
        )
        
        self.dir_fuse = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )
        
        self.output_conv = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.output_bn = nn.BatchNorm2d(channels)
    
    def forward(self, x):
        aspp_out1 = self.branch1(x)
        aspp_out2 = self.branch2(x)
        aspp_out3 = self.branch3(x)
        aspp_fused = torch.cat([aspp_out1, aspp_out2, aspp_out3], dim=1)
        aspp_out = self.aspp_fuse(aspp_fused)
        
        dir_out1 = self.dir_conv1(x)
        dir_out2 = self.dir_conv2(x)
        dir_fused = torch.cat([dir_out1, dir_out2], dim=1)
        dir_out = self.dir_fuse(dir_fused)
        
        fused = aspp_out + dir_out
        fused = fused + x
        
        out = self.output_conv(fused)
        out = self.output_bn(out)
        
        return out


class LowFreqExpert(nn.Module):
    def __init__(self, channels):
        super(LowFreqExpert, self).__init__()
        
        self.encoder_conv = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=7, padding=3, groups=channels, bias=False),
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.bottleneck = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )
        
        self.decoder_conv = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )
        
        self.lowpass = nn.AvgPool2d(kernel_size=5, stride=1, padding=2)
        
        self.alpha = nn.Parameter(torch.tensor(0.5))
        
        self.output_conv = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.output_bn = nn.BatchNorm2d(channels)
    
    def forward(self, x):
        B, C, H, W = x.shape
        
        enc_feat = self.encoder_conv(x)
        skip_feat = enc_feat.clone()
        enc_feat = self.pool(enc_feat)
        
        bottleneck_feat = self.bottleneck(enc_feat)
        
        dec_feat = F.interpolate(bottleneck_feat, size=(H, W), mode='bilinear', align_corners=True)
        dec_feat = torch.cat([dec_feat, skip_feat], dim=1)
        dec_feat = self.decoder_conv(dec_feat)
        
        lp_feat = self.lowpass(x)
        
        fused = self.alpha * dec_feat + (1 - self.alpha) * lp_feat
        
        out = self.output_conv(fused)
        out = self.output_bn(out)
        
        return out


class GateNetwork(nn.Module):
    def __init__(self, channels, num_experts=3):
        super(GateNetwork, self).__init__()
        
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // 4),
            nn.ReLU(inplace=True),
            nn.Linear(channels // 4, num_experts)
        )
    
    def forward(self, x):
        B, C, H, W = x.shape
        pooled = self.global_pool(x).view(B, C)
        weights = self.fc(pooled)
        weights = F.softmax(weights, dim=-1)
        return weights


class MoEEnhancement(nn.Module):
    def __init__(self, channels, num_experts=3):
        super(MoEEnhancement, self).__init__()
        
        self.experts = nn.ModuleList([
            HighFreqExpert(channels),
            MidFreqExpert(channels),
            LowFreqExpert(channels)
        ])
        
        self.gate = GateNetwork(channels, num_experts)
    
    def forward(self, x, return_gate_weights=False):
        weights = self.gate(x)
        
        expert_outputs = [expert(x) for expert in self.experts]
        
        B, C, H, W = x.shape
        out = torch.zeros(B, C, H, W, device=x.device)
        for i, expert_out in enumerate(expert_outputs):
            out = out + weights[:, i].view(B, 1, 1, 1) * expert_out
        
        if return_gate_weights:
            return out, weights
        return out


if __name__ == '__main__':
    model = MoEEnhancement(channels=256)
    
    x = torch.randn(2, 256, 7, 6)
    out = model(x)
    print(f"输入形状: {x.shape}")
    print(f"输出形状: {out.shape}")
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"总参数量: {total_params:,}")
