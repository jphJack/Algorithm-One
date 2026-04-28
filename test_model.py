import torch
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.vibe_net import VIBENet
from dataset import get_dataloader
import config


def test_model_forward():
    print("=" * 50)
    print("测试模型前向传播")
    print("=" * 50)

    dataset_cfg = config.get_dataset_config(config.DEFAULT_DATASET)
    num_classes = dataset_cfg['num_classes']
    in_channels = dataset_cfg['in_channels']
    img_size = dataset_cfg['img_size']

    model = VIBENet(
        num_classes=num_classes, feature_dim=config.FEATURE_DIM,
        out_stages=config.OUT_STAGES, reducer_channels=config.REDUCER_CHANNELS
    )
    model.eval()

    print_img = torch.randn(2, in_channels, img_size[0], img_size[1])
    vein_img = torch.randn(2, in_channels, img_size[0], img_size[1])

    with torch.no_grad():
        output = model(print_img, vein_img)

    print(f"掌纹图像输入形状: {print_img.shape}")
    print(f"掌静脉图像输入形状: {vein_img.shape}")
    print(f"模型输出形状: {output.shape}")
    print(f"预期输出形状: [2, {num_classes}]")

    assert output.shape == (2, num_classes), f"输出形状不正确: {output.shape}"
    print("✓ 模型前向传播测试通过!")

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n模型参数统计:")
    print(f"  总参数量: {total_params:,}")
    print(f"  可训练参数量: {trainable_params:,}")

    return True


def test_load_balancing_loss():
    print("\n" + "=" * 50)
    print("测试负载均衡损失")
    print("=" * 50)

    dataset_cfg = config.get_dataset_config(config.DEFAULT_DATASET)
    num_classes = dataset_cfg['num_classes']
    in_channels = dataset_cfg['in_channels']
    img_size = dataset_cfg['img_size']

    model = VIBENet(
        num_classes=num_classes, feature_dim=config.FEATURE_DIM,
        out_stages=config.OUT_STAGES, reducer_channels=config.REDUCER_CHANNELS
    )
    model.eval()

    print_img = torch.randn(4, in_channels, img_size[0], img_size[1])
    vein_img = torch.randn(4, in_channels, img_size[0], img_size[1])

    with torch.no_grad():
        output = model(print_img, vein_img)
        lb_loss = model.compute_load_balancing_loss()

    print(f"模型输出形状: {output.shape}")
    print(f"负载均衡损失: {lb_loss.item():.6f}")
    print(f"负载均衡损失权重: {config.LOAD_BALANCE_WEIGHT}")

    assert lb_loss.item() >= -1e-6, f"负载均衡损失不应为负数: {lb_loss.item()}"

    print(f"\n各MoE模块门控权重分布:")
    print(f"  掌纹增强: {model.print_enhancement._gate_weights.mean(dim=0)}")
    print(f"  掌静脉增强: {model.vein_enhancement._gate_weights.mean(dim=0)}")
    print(f"  融合模块: {model.fusion._gate_weights.mean(dim=0)}")

    print("\n✓ 负载均衡损失测试通过!")
    return True


def test_dataloader():
    print("\n" + "=" * 50)
    print("测试数据加载器")
    print("=" * 50)

    dataset_cfg = config.get_dataset_config(config.DEFAULT_DATASET)
    img_size = dataset_cfg['img_size']

    try:
        train_loader = get_dataloader(
            config.DEFAULT_DATASET,
            mode='train',
            batch_size=4,
            num_workers=0,
            shuffle=True
        )

        print(f"训练集样本数: {len(train_loader.dataset)}")
        print(f"批次数量: {len(train_loader)}")

        for print_img, vein_img, label in train_loader:
            print(f"\n批次数据形状:")
            print(f"  掌纹图像: {print_img.shape}")
            print(f"  掌静脉图像: {vein_img.shape}")
            print(f"  标签: {label}")
            break

        assert print_img.shape[2:] == img_size, f"掌纹图像尺寸不正确: {print_img.shape}"
        assert vein_img.shape[2:] == img_size, f"掌静脉图像尺寸不正确: {vein_img.shape}"

        print("\n✓ 数据加载器测试通过!")
        return True

    except Exception as e:
        print(f"✗ 数据加载器测试失败: {e}")
        return False


def test_full_pipeline():
    print("\n" + "=" * 50)
    print("测试完整流程")
    print("=" * 50)

    dataset_cfg = config.get_dataset_config(config.DEFAULT_DATASET)
    num_classes = dataset_cfg['num_classes']

    model = VIBENet(
        num_classes=num_classes, feature_dim=config.FEATURE_DIM,
        out_stages=config.OUT_STAGES, reducer_channels=config.REDUCER_CHANNELS
    )
    model.eval()

    try:
        train_loader = get_dataloader(
            config.DEFAULT_DATASET,
            mode='train',
            batch_size=4,
            num_workers=0,
            shuffle=False
        )

        for print_img, vein_img, label in train_loader:
            with torch.no_grad():
                output = model(print_img, vein_img)
                lb_loss = model.compute_load_balancing_loss()

            print(f"输入批次形状:")
            print(f"  掌纹: {print_img.shape}")
            print(f"  掌静脉: {vein_img.shape}")
            print(f"输出形状: {output.shape}")
            print(f"负载均衡损失: {lb_loss.item():.6f}")
            print(f"标签: {label}")

            pred = torch.argmax(output, dim=1)
            print(f"预测类别: {pred}")
            break

        print("\n✓ 完整流程测试通过!")
        return True

    except Exception as e:
        print(f"✗ 完整流程测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_module_shapes():
    print("\n" + "=" * 50)
    print("测试各模块输出形状")
    print("=" * 50)

    from models.backbone import DualStreamBackbone, LightweightBackbone, Reducer, MultiScaleFeatureExtractor
    from models.moe_enhancement import MoEEnhancement, HighFreqExpert, MidFreqExpert, LowFreqExpert
    from models.moe_fusion import MoEFusion, CrossAttentionExpert, MultiScaleConvExpert, ChannelInteractionExpert
    from models.classifier import Classifier

    B, C, H, W = 2, 256, 16, 16

    print("\n1. 骨干网络测试:")
    backbone = LightweightBackbone(in_channels=3, feature_dim=256, out_stages=[3, 4, 5])
    x = torch.randn(B, 3, 128, 128)
    features = backbone(x)
    for stage, feat in features.items():
        print(f"   Stage {stage}: {feat.shape}")

    print("\n2. Reducer测试:")
    reducer = Reducer(128, 64)
    x = torch.randn(B, 128, 16, 16)
    print(f"   输入: {x.shape} -> 输出: {reducer(x).shape}")

    print("\n3. 多尺度特征提取器测试:")
    stage_channels = {3: 128, 4: 256, 5: 256}
    extractor = MultiScaleFeatureExtractor(stage_channels, reducer_channels=64, feature_dim=256)
    features = {
        3: torch.randn(B, 128, 16, 16),
        4: torch.randn(B, 256, 8, 8),
        5: torch.randn(B, 256, 4, 4),
    }
    out = extractor(features)
    print(f"   输入: 3个尺度特征 -> 输出: {out.shape}")

    print("\n4. 双支流骨干网络测试:")
    dual_backbone = DualStreamBackbone(in_channels=3, feature_dim=256)
    print_img = torch.randn(B, 3, 128, 128)
    vein_img = torch.randn(B, 3, 128, 128)
    print_feat, vein_feat = dual_backbone(print_img, vein_img)
    print(f"   掌纹特征: {print_feat.shape}")
    print(f"   掌静脉特征: {vein_feat.shape}")

    print("\n5. MoE特征增强模块测试:")
    moe_enhance = MoEEnhancement(C)
    x = torch.randn(B, C, H, W)
    out = moe_enhance(x)
    lb_loss = moe_enhance.load_balancing_loss()
    print(f"   MoE增强: {x.shape} -> {out.shape}")
    print(f"   负载均衡损失: {lb_loss.item():.6f}")
    print(f"   门控权重均值: {moe_enhance._gate_weights.mean(dim=0)}")

    print("\n6. MoE融合模块测试:")
    moe_fusion = MoEFusion(C)
    f_p = torch.randn(B, C, H, W)
    f_v = torch.randn(B, C, H, W)
    out = moe_fusion(f_p, f_v)
    lb_loss = moe_fusion.load_balancing_loss()
    print(f"   MoE融合: ({f_p.shape}, {f_v.shape}) -> {out.shape}")
    print(f"   负载均衡损失: {lb_loss.item():.6f}")
    print(f"   门控权重均值: {moe_fusion._gate_weights.mean(dim=0)}")

    print("\n7. 分类器测试:")
    dataset_cfg = config.get_dataset_config(config.DEFAULT_DATASET)
    num_classes = dataset_cfg['num_classes']
    classifier = Classifier(C, num_classes)
    x = torch.randn(B, C, H, W)
    out = classifier(x)
    print(f"   输入: {x.shape} -> 输出: {out.shape}")

    print("\n✓ 各模块形状测试通过!")
    return True


if __name__ == '__main__':
    print("\n" + "=" * 60)
    print("VIBE网络测试套件")
    print("=" * 60)

    results = []

    results.append(("模型前向传播", test_model_forward()))
    results.append(("负载均衡损失", test_load_balancing_loss()))
    results.append(("各模块形状", test_module_shapes()))
    results.append(("数据加载器", test_dataloader()))
    results.append(("完整流程", test_full_pipeline()))

    print("\n" + "=" * 60)
    print("测试结果汇总")
    print("=" * 60)

    for name, passed in results:
        status = "✓ 通过" if passed else "✗ 失败"
        print(f"  {name}: {status}")

    all_passed = all(r[1] for r in results)
    print("\n" + ("=" * 60))
    if all_passed:
        print("所有测试通过! 模型可以正常运行。")
    else:
        print("部分测试失败，请检查错误信息。")
    print("=" * 60)
