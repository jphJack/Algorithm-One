import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossAttentionExpert(nn.Module):
    def __init__(self, channels):
        super(CrossAttentionExpert, self).__init__()
        
        self.query_conv_p = nn.Conv2d(channels, channels // 8, kernel_size=1)
        self.key_conv_v = nn.Conv2d(channels, channels // 8, kernel_size=1)
        self.value_conv_v = nn.Conv2d(channels, channels, kernel_size=1)
        
        self.query_conv_v = nn.Conv2d(channels, channels // 8, kernel_size=1)
        self.key_conv_p = nn.Conv2d(channels, channels // 8, kernel_size=1)
        self.value_conv_p = nn.Conv2d(channels, channels, kernel_size=1)
        
        self.output_conv = nn.Conv2d(channels * 4, channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(channels)
    
    def forward(self, f_p, f_v):
        B, C, H, W = f_p.shape
        
        q_p = self.query_conv_p(f_p).view(B, -1, H * W).permute(0, 2, 1)
        k_v = self.key_conv_v(f_v).view(B, -1, H * W)
        v_v = self.value_conv_v(f_v).view(B, -1, H * W)
        
        attn_p2v = torch.bmm(q_p, k_v)
        attn_p2v = F.softmax(attn_p2v, dim=-1)
        cross_p2v = torch.bmm(v_v, attn_p2v.permute(0, 2, 1))
        cross_p2v = cross_p2v.view(B, C, H, W)
        
        q_v = self.query_conv_v(f_v).view(B, -1, H * W).permute(0, 2, 1)
        k_p = self.key_conv_p(f_p).view(B, -1, H * W)
        v_p = self.value_conv_p(f_p).view(B, -1, H * W)
        
        attn_v2p = torch.bmm(q_v, k_p)
        attn_v2p = F.softmax(attn_v2p, dim=-1)
        cross_v2p = torch.bmm(v_p, attn_v2p.permute(0, 2, 1))
        cross_v2p = cross_v2p.view(B, C, H, W)
        
        fused = torch.cat([f_p, f_v, cross_p2v, cross_v2p], dim=1)
        out = self.output_conv(fused)
        out = self.bn(out)
        
        return out


class MultiScaleConvExpert(nn.Module):
    def __init__(self, channels):
        super(MultiScaleConvExpert, self).__init__()
        
        self.concat_conv = nn.Conv2d(channels * 2, channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        
        self.branch1 = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.branch3 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.branch5 = nn.Conv2d(channels, channels, kernel_size=5, padding=2, bias=False)
        
        self.output_conv = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
    
    def forward(self, f_p, f_v):
        concat = torch.cat([f_p, f_v], dim=1)
        concat = self.concat_conv(concat)
        concat = F.relu(self.bn1(concat))
        
        out1 = self.branch1(concat)
        out3 = self.branch3(concat)
        out5 = self.branch5(concat)
        
        fused = out1 + out3 + out5
        
        out = self.output_conv(fused)
        out = self.bn2(out)
        
        return out


class ChannelInteractionExpert(nn.Module):
    def __init__(self, channels):
        super(ChannelInteractionExpert, self).__init__()
        
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        self.mlp_v = nn.Sequential(
            nn.Linear(channels, channels // 4),
            nn.ReLU(inplace=True),
            nn.Linear(channels // 4, channels),
            nn.Sigmoid()
        )
        
        self.mlp_p = nn.Sequential(
            nn.Linear(channels, channels // 4),
            nn.ReLU(inplace=True),
            nn.Linear(channels // 4, channels),
            nn.Sigmoid()
        )
        
        self.output_conv = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(channels)
    
    def forward(self, f_p, f_v):
        B, C, H, W = f_p.shape
        
        global_p = self.global_pool(f_p).view(B, C)
        global_v = self.global_pool(f_v).view(B, C)
        
        weights_p = self.mlp_v(global_v).view(B, C, 1, 1)
        weights_v = self.mlp_p(global_p).view(B, C, 1, 1)
        
        f_p_calibrated = f_p * weights_p
        f_v_calibrated = f_v * weights_v
        
        fused = f_p_calibrated + f_v_calibrated
        
        out = self.output_conv(fused)
        out = self.bn(out)
        
        return out


class FusionGateNetwork(nn.Module):
    def __init__(self, channels, num_experts=3):
        super(FusionGateNetwork, self).__init__()
        
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels * 2, channels // 2),
            nn.ReLU(inplace=True),
            nn.Linear(channels // 2, num_experts)
        )
    
    def forward(self, f_p, f_v):
        B, C, H, W = f_p.shape
        pooled_p = self.global_pool(f_p).view(B, C)
        pooled_v = self.global_pool(f_v).view(B, C)
        pooled = torch.cat([pooled_p, pooled_v], dim=-1)
        weights = self.fc(pooled)
        weights = F.softmax(weights, dim=-1)
        return weights


class MoEFusion(nn.Module):
    def __init__(self, channels, num_experts=3):
        super(MoEFusion, self).__init__()
        
        self.num_experts = num_experts
        
        self.experts = nn.ModuleList([
            CrossAttentionExpert(channels),
            MultiScaleConvExpert(channels),
            ChannelInteractionExpert(channels)
        ])
        
        self.gate = FusionGateNetwork(channels, num_experts)
        self._gate_weights = None
    
    def load_balancing_loss(self):
        if self._gate_weights is None:
            return torch.tensor(0.0, device=next(self.parameters()).device)
        f = self._gate_weights.mean(dim=0)
        return self.num_experts * (f * f).sum() - 1.0
    
    def forward(self, f_p, f_v, return_gate_weights=False):
        weights = self.gate(f_p, f_v)
        self._gate_weights = weights.detach()
        
        expert_outputs = [expert(f_p, f_v) for expert in self.experts]
        
        B, C, H, W = f_p.shape
        out = torch.zeros(B, C, H, W, device=f_p.device)
        for i, expert_out in enumerate(expert_outputs):
            out = out + weights[:, i].view(B, 1, 1, 1) * expert_out
        
        if return_gate_weights:
            return out, weights
        return out


if __name__ == '__main__':
    model = MoEFusion(channels=256)
    
    f_p = torch.randn(2, 256, 7, 6)
    f_v = torch.randn(2, 256, 7, 6)
    
    out = model(f_p, f_v)
    print(f"掌纹特征形状: {f_p.shape}")
    print(f"掌静脉特征形状: {f_v.shape}")
    print(f"融合输出形状: {out.shape}")
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"总参数量: {total_params:,}")
