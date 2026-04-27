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


class Reducer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Reducer, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class LightweightBackbone(nn.Module):
    def __init__(self, in_channels=1, feature_dim=256, out_stages=None):
        super(LightweightBackbone, self).__init__()
        if out_stages is None:
            out_stages = [3, 4, 5]
        self.out_stages = out_stages

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
        self.stage_channels = {2: 64, 3: 128, 4: 256, 5: feature_dim}

    def forward(self, x):
        features = {}

        x = self.stage1(x)
        x = self.stage2(x)
        if 2 in self.out_stages:
            features[2] = x
        x = self.stage3(x)
        if 3 in self.out_stages:
            features[3] = x
        x = self.stage4(x)
        if 4 in self.out_stages:
            features[4] = x
        x = self.stage5(x)
        if 5 in self.out_stages:
            features[5] = x

        return features


class MultiScaleFeatureExtractor(nn.Module):
    def __init__(self, stage_channels, reducer_channels=64, feature_dim=256):
        super(MultiScaleFeatureExtractor, self).__init__()
        self.stages = sorted(stage_channels.keys())

        self.reducers = nn.ModuleDict({
            str(k): Reducer(stage_channels[k], reducer_channels)
            for k in self.stages
        })

        concat_channels = reducer_channels * len(self.stages)
        self.projection = nn.Sequential(
            nn.Conv2d(concat_channels, feature_dim, 1, bias=False),
            nn.BatchNorm2d(feature_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(feature_dim, feature_dim, 3, padding=1, bias=False),
            nn.BatchNorm2d(feature_dim),
            nn.ReLU(inplace=True)
        )

    def forward(self, features):
        target_size = features[self.stages[0]].shape[2:]

        reduced = []
        for k in self.stages:
            feat = features[k]
            reduced_feat = self.reducers[str(k)](feat)
            if reduced_feat.shape[2:] != target_size:
                reduced_feat = F.interpolate(
                    reduced_feat, size=target_size, mode='bilinear', align_corners=True
                )
            reduced.append(reduced_feat)

        concat = torch.cat(reduced, dim=1)
        return self.projection(concat)


class DualStreamBackbone(nn.Module):
    def __init__(self, in_channels=1, feature_dim=256, out_stages=None, reducer_channels=64):
        super(DualStreamBackbone, self).__init__()
        if out_stages is None:
            out_stages = [3, 4, 5]

        self.print_backbone = LightweightBackbone(in_channels, feature_dim, out_stages)
        self.vein_backbone = LightweightBackbone(in_channels, feature_dim, out_stages)

        stage_channels = {k: self.print_backbone.stage_channels[k] for k in out_stages}

        self.print_extractor = MultiScaleFeatureExtractor(stage_channels, reducer_channels, feature_dim)
        self.vein_extractor = MultiScaleFeatureExtractor(stage_channels, reducer_channels, feature_dim)

        self.out_channels = feature_dim

    def forward(self, print_img, vein_img):
        print_features = self.print_backbone(print_img)
        vein_features = self.vein_backbone(vein_img)

        print_feat = self.print_extractor(print_features)
        vein_feat = self.vein_extractor(vein_features)

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
