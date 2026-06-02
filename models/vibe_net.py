import torch
import torch.nn as nn
from .backbone import DualStreamBackbone
from .moe_fusion import MoEFusion
from .classifier import Classifier


class VIBENet(nn.Module):
    def __init__(
        self,
        num_classes=290,
        feature_dim=256,
        out_stages=None,
        reducer_channels=64,
        moe_num_experts=3,
        use_multiscale_extractor=False,
        classifier_embed_dim=256,
        classifier_margin=0.5,
        classifier_scale=30.0,
        classifier_dropout=0.5,
    ):
        super(VIBENet, self).__init__()

        self.backbone = DualStreamBackbone(
            in_channels=3, feature_dim=feature_dim,
            out_stages=out_stages, reducer_channels=reducer_channels,
            moe_num_experts=moe_num_experts,
            enable_stage_enhancement=True,
            use_multiscale_extractor=use_multiscale_extractor,
        )

        self.fusion = MoEFusion(feature_dim, num_experts=moe_num_experts)

        self.classifier = Classifier(
            feature_dim,
            num_classes,
            embed_dim=classifier_embed_dim,
            margin=classifier_margin,
            scale=classifier_scale,
            dropout=classifier_dropout,
        )

    def compute_load_balancing_loss(self):
        return self.backbone.load_balancing_loss() + self.fusion.load_balancing_loss()

    def forward(self, print_img, vein_img, labels=None, return_gate_weights=False, return_embedding=False):
        if return_gate_weights:
            print_feat, vein_feat, stage_gate_weights = self.backbone(
                print_img, vein_img, return_gate_weights=True
            )
        else:
            print_feat, vein_feat = self.backbone(print_img, vein_img)

        if print_feat.shape[2:] != vein_feat.shape[2:]:
            target_h = min(print_feat.shape[2], vein_feat.shape[2])
            target_w = min(print_feat.shape[3], vein_feat.shape[3])
            print_feat = nn.functional.interpolate(
                print_feat, size=(target_h, target_w), mode='bilinear', align_corners=True
            )
            vein_feat = nn.functional.interpolate(
                vein_feat, size=(target_h, target_w), mode='bilinear', align_corners=True
            )

        if return_gate_weights:
            fused_feat, fusion_gate_weights = self.fusion(print_feat, vein_feat, return_gate_weights=True)
        else:
            fused_feat = self.fusion(print_feat, vein_feat)

        if return_embedding:
            output, embedding = self.classifier(fused_feat, labels=labels, return_embedding=True)
        else:
            output = self.classifier(fused_feat, labels=labels)

        if return_gate_weights:
            gate_weights = {
                'print_stage_enhancement': stage_gate_weights.get('print', {}),
                'vein_stage_enhancement': stage_gate_weights.get('vein', {}),
                'fusion': fusion_gate_weights,
            }
            if return_embedding:
                return output, gate_weights, embedding
            return output, gate_weights

        if return_embedding:
            return output, embedding
        return output


if __name__ == '__main__':
    model = VIBENet(num_classes=290, feature_dim=256, out_stages=[3, 4, 5], reducer_channels=64)

    print_img = torch.randn(2, 3, 128, 128)
    vein_img = torch.randn(2, 3, 128, 128)

    output = model(print_img, vein_img)
    print(f"掌纹图像形状: {print_img.shape}")
    print(f"掌静脉图像形状: {vein_img.shape}")
    print(f"输出形状: {output.shape}")

    output, gate_weights = model(print_img, vein_img, return_gate_weights=True)
    print(f"\n门控权重:")
    for name, weights in gate_weights.items():
        if isinstance(weights, dict):
            for stage, stage_weights in weights.items():
                print(f"  {name} stage {stage}: {stage_weights.shape} -> {stage_weights}")
        else:
            print(f"  {name}: {weights.shape} -> {weights}")

    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n总参数量: {total_params:,}")
