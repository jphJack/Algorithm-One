import torch
import torch.nn as nn
from .backbone import DualStreamBackbone
from .moe_enhancement import MoEEnhancement
from .moe_fusion import MoEFusion
from .classifier import Classifier


class VIBENet(nn.Module):
    def __init__(self, num_classes=290, feature_dim=256):
        super(VIBENet, self).__init__()
        
        self.backbone = DualStreamBackbone(in_channels=3, feature_dim=feature_dim)
        
        self.print_enhancement = MoEEnhancement(feature_dim, num_experts=3)
        self.vein_enhancement = MoEEnhancement(feature_dim, num_experts=3)
        
        self.fusion = MoEFusion(feature_dim, num_experts=3)
        
        self.classifier = Classifier(feature_dim, num_classes)
    
    def forward(self, print_img, vein_img, return_gate_weights=False):
        print_feat, vein_feat = self.backbone(print_img, vein_img)
        
        if return_gate_weights:
            print_enhanced, print_gate_weights = self.print_enhancement(print_feat, return_gate_weights=True)
            vein_enhanced, vein_gate_weights = self.vein_enhancement(vein_feat, return_gate_weights=True)
        else:
            print_enhanced = self.print_enhancement(print_feat)
            vein_enhanced = self.vein_enhancement(vein_feat)
        
        if print_enhanced.shape[2:] != vein_enhanced.shape[2:]:
            target_h = min(print_enhanced.shape[2], vein_enhanced.shape[2])
            target_w = min(print_enhanced.shape[3], vein_enhanced.shape[3])
            print_enhanced = nn.functional.interpolate(
                print_enhanced, size=(target_h, target_w), mode='bilinear', align_corners=True
            )
            vein_enhanced = nn.functional.interpolate(
                vein_enhanced, size=(target_h, target_w), mode='bilinear', align_corners=True
            )
        
        if return_gate_weights:
            fused_feat, fusion_gate_weights = self.fusion(print_enhanced, vein_enhanced, return_gate_weights=True)
        else:
            fused_feat = self.fusion(print_enhanced, vein_enhanced)
        
        output = self.classifier(fused_feat)
        
        if return_gate_weights:
            gate_weights = {
                'print_enhancement': print_gate_weights,
                'vein_enhancement': vein_gate_weights,
                'fusion': fusion_gate_weights
            }
            return output, gate_weights
        
        return output


if __name__ == '__main__':
    model = VIBENet(num_classes=290, feature_dim=256)
    
    print_img = torch.randn(2, 3, 128, 128)
    vein_img = torch.randn(2, 3, 128, 128)
    
    output = model(print_img, vein_img)
    print(f"掌纹图像形状: {print_img.shape}")
    print(f"掌静脉图像形状: {vein_img.shape}")
    print(f"输出形状: {output.shape}")
    
    output, gate_weights = model(print_img, vein_img, return_gate_weights=True)
    print(f"\n门控权重:")
    for name, weights in gate_weights.items():
        print(f"  {name}: {weights.shape} -> {weights}")
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n总参数量: {total_params:,}")
