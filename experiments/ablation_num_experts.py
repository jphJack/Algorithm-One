import os
import sys
import argparse
import time
import math
import random
import csv
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from models.moe_enhancement import HighFreqExpert, MidFreqExpert, LowFreqExpert, GateNetwork
from models.moe_fusion import CrossAttentionExpert, MultiScaleConvExpert, ChannelInteractionExpert, FusionGateNetwork
from models.backbone import DualStreamBackbone
from models.classifier import Classifier
from dataset import get_dataloader
import config

from utils import setup_matplotlib, get_result_dir, DATASETS, EXPERT_COLORS


ENHANCEMENT_EXPERT_CLASSES = [HighFreqExpert, MidFreqExpert, LowFreqExpert]
FUSION_EXPERT_CLASSES = [CrossAttentionExpert, MultiScaleConvExpert, ChannelInteractionExpert]


def build_expert_list(expert_classes, num_experts):
    experts = []
    for i in range(num_experts):
        experts.append(expert_classes[i % len(expert_classes)])
    return experts


class AblationMoEEnhancement(nn.Module):
    def __init__(self, channels, num_experts=3):
        super(AblationMoEEnhancement, self).__init__()
        self.num_experts = num_experts
        expert_list = build_expert_list(ENHANCEMENT_EXPERT_CLASSES, num_experts)
        self.experts = nn.ModuleList([cls(channels) for cls in expert_list])
        self.gate = GateNetwork(channels, num_experts)
        self._gate_weights = None
        self._gate_weights_for_loss = None

    def load_balancing_loss(self):
        if self._gate_weights_for_loss is None:
            return torch.tensor(0.0, device=next(self.parameters()).device)
        f = self._gate_weights_for_loss.mean(dim=0)
        return self.num_experts * (f * f).sum() - 1.0

    def forward(self, x, return_gate_weights=False):
        weights = self.gate(x)
        self._gate_weights = weights.detach()
        self._gate_weights_for_loss = weights
        expert_outputs = [expert(x) for expert in self.experts]
        B, C, H, W = x.shape
        out = torch.zeros(B, C, H, W, device=x.device)
        for i, expert_out in enumerate(expert_outputs):
            out = out + weights[:, i].view(B, 1, 1, 1) * expert_out
        if return_gate_weights:
            return out, weights
        return out


class AblationMoEFusion(nn.Module):
    def __init__(self, channels, num_experts=3):
        super(AblationMoEFusion, self).__init__()
        self.num_experts = num_experts
        expert_list = build_expert_list(FUSION_EXPERT_CLASSES, num_experts)
        self.experts = nn.ModuleList([cls(channels) for cls in expert_list])
        self.gate = FusionGateNetwork(channels, num_experts)
        self._gate_weights = None
        self._gate_weights_for_loss = None

    def load_balancing_loss(self):
        if self._gate_weights_for_loss is None:
            return torch.tensor(0.0, device=next(self.parameters()).device)
        f = self._gate_weights_for_loss.mean(dim=0)
        return self.num_experts * (f * f).sum() - 1.0

    def forward(self, f_p, f_v, return_gate_weights=False):
        weights = self.gate(f_p, f_v)
        self._gate_weights = weights.detach()
        self._gate_weights_for_loss = weights
        expert_outputs = [expert(f_p, f_v) for expert in self.experts]
        B, C, H, W = f_p.shape
        out = torch.zeros(B, C, H, W, device=f_p.device)
        for i, expert_out in enumerate(expert_outputs):
            out = out + weights[:, i].view(B, 1, 1, 1) * expert_out
        if return_gate_weights:
            return out, weights
        return out


class AblationDualStreamBackbone(nn.Module):
    def __init__(
        self,
        in_channels=1,
        feature_dim=256,
        out_stages=None,
        reducer_channels=64,
        moe_num_experts=3,
        enable_stage_enhancement=True,
        use_multiscale_extractor=False,
    ):
        super(AblationDualStreamBackbone, self).__init__()
        if out_stages is None:
            out_stages = [3, 4, 5]

        from models.backbone import LightweightBackbone, MultiScaleFeatureExtractor
        self.print_backbone = LightweightBackbone(in_channels, feature_dim, out_stages)
        self.vein_backbone = LightweightBackbone(in_channels, feature_dim, out_stages)

        stage_channels = {k: self.print_backbone.stage_channels[k] for k in out_stages}

        self.enable_stage_enhancement = enable_stage_enhancement
        self.out_stages = out_stages
        self.fusion_stage = max(out_stages)
        self.use_multiscale_extractor = use_multiscale_extractor

        if self.enable_stage_enhancement:
            self.print_stage_enhancers = nn.ModuleDict({
                str(k): AblationMoEEnhancement(stage_channels[k], num_experts=moe_num_experts)
                for k in out_stages
            })
            self.vein_stage_enhancers = nn.ModuleDict({
                str(k): AblationMoEEnhancement(stage_channels[k], num_experts=moe_num_experts)
                for k in out_stages
            })

        if self.use_multiscale_extractor:
            self.print_extractor = MultiScaleFeatureExtractor(stage_channels, reducer_channels, feature_dim)
            self.vein_extractor = MultiScaleFeatureExtractor(stage_channels, reducer_channels, feature_dim)
        else:
            self.print_extractor = None
            self.vein_extractor = None

        self.out_channels = feature_dim

    def load_balancing_loss(self):
        loss = torch.tensor(0.0, device=next(self.parameters()).device)
        if self.enable_stage_enhancement:
            for k in self.out_stages:
                loss = loss + self.print_stage_enhancers[str(k)].load_balancing_loss()
                loss = loss + self.vein_stage_enhancers[str(k)].load_balancing_loss()
        return loss

    def forward(self, print_img, vein_img, return_gate_weights=False):
        print_feats = self.print_backbone(print_img)
        vein_feats = self.vein_backbone(vein_img)

        stage_gate_weights = {'print': {}, 'vein': {}}

        if self.enable_stage_enhancement:
            for k in self.out_stages:
                if return_gate_weights:
                    print_feats[k], pw = self.print_stage_enhancers[str(k)](
                        print_feats[k], return_gate_weights=True
                    )
                    vein_feats[k], vw = self.vein_stage_enhancers[str(k)](
                        vein_feats[k], return_gate_weights=True
                    )
                    stage_gate_weights['print'][k] = pw
                    stage_gate_weights['vein'][k] = vw
                else:
                    print_feats[k] = self.print_stage_enhancers[str(k)](print_feats[k])
                    vein_feats[k] = self.vein_stage_enhancers[str(k)](vein_feats[k])

        if self.use_multiscale_extractor and self.print_extractor is not None:
            print_feat = self.print_extractor(print_feats)
            vein_feat = self.vein_extractor(vein_feats)
        else:
            print_feat = print_feats[self.fusion_stage]
            vein_feat = vein_feats[self.fusion_stage]

        if return_gate_weights:
            return print_feat, vein_feat, stage_gate_weights
        return print_feat, vein_feat


class AblationVIBENet(nn.Module):
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
        super(AblationVIBENet, self).__init__()

        self.backbone = AblationDualStreamBackbone(
            in_channels=3, feature_dim=feature_dim,
            out_stages=out_stages, reducer_channels=reducer_channels,
            moe_num_experts=moe_num_experts,
            enable_stage_enhancement=True,
            use_multiscale_extractor=use_multiscale_extractor,
        )

        self.fusion = AblationMoEFusion(feature_dim, num_experts=moe_num_experts)

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


def seed_everything(seed, deterministic=True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        try:
            torch.use_deterministic_algorithms(True)
        except Exception:
            pass
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = False


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


def train_and_evaluate(num_experts, dataset_name, epochs, batch_size, lr, device):
    seed_everything(config.SEED, deterministic=config.DETERMINISTIC)

    dataset_cfg = config.get_dataset_config(dataset_name)
    num_classes = dataset_cfg['num_classes']

    model = AblationVIBENet(
        num_classes=num_classes,
        feature_dim=config.FEATURE_DIM,
        out_stages=config.OUT_STAGES,
        reducer_channels=config.REDUCER_CHANNELS,
        moe_num_experts=num_experts,
        classifier_embed_dim=config.CLASSIFIER_EMBED_DIM,
        classifier_margin=config.ARC_MARGIN,
        classifier_scale=config.ARC_SCALE,
        classifier_dropout=config.CLASSIFIER_DROPOUT,
    )
    model = model.to(device)

    num_params = count_parameters(model)

    train_loader = get_dataloader(
        dataset_name, mode='train', batch_size=batch_size,
        num_workers=config.NUM_WORKERS, shuffle=True
    )
    test_loader = get_dataloader(
        dataset_name, mode='test', batch_size=batch_size,
        num_workers=config.NUM_WORKERS, shuffle=False
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=config.WEIGHT_DECAY)

    best_acc = 0.0
    best_epoch = -1

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(train_loader, desc=f'Experts={num_experts} Epoch {epoch+1}/{epochs}')

        for print_img, vein_img, labels in pbar:
            print_img = print_img.to(device)
            vein_img = vein_img.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(print_img, vein_img, labels=labels)
            ce_loss = criterion(outputs, labels)
            lb_loss = model.compute_load_balancing_loss()
            loss = ce_loss + config.LOAD_BALANCE_WEIGHT * lb_loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            pbar.set_postfix({
                'loss': f'{running_loss / len(pbar):.4f}',
                'acc': f'{100. * correct / total:.2f}%'
            })

        if config.WARMUP_EPOCHS > 0 and epoch < config.WARMUP_EPOCHS:
            current_lr = lr * float(epoch + 1) / float(config.WARMUP_EPOCHS)
        else:
            denom = max(1, epochs - config.WARMUP_EPOCHS)
            progress = float(epoch - config.WARMUP_EPOCHS) / float(denom)
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
            current_lr = config.MIN_LR + (lr - config.MIN_LR) * cosine

        for param_group in optimizer.param_groups:
            param_group['lr'] = current_lr

        model.eval()
        test_correct = 0
        test_total = 0

        with torch.no_grad():
            for print_img, vein_img, labels in test_loader:
                print_img = print_img.to(device)
                vein_img = vein_img.to(device)
                labels = labels.to(device)

                outputs = model(print_img, vein_img)
                _, predicted = outputs.max(1)
                test_total += labels.size(0)
                test_correct += predicted.eq(labels).sum().item()

        test_acc = 100. * test_correct / test_total

        if test_acc > best_acc:
            best_acc = test_acc
            best_epoch = epoch + 1

        print(f'  Epoch {epoch+1}/{epochs} - Test Acc: {test_acc:.2f}% | Best: {best_acc:.2f}% (Epoch {best_epoch})')

    return best_acc, num_params, best_epoch


def plot_accuracy_chart(expert_counts, accuracies, save_dir):
    plt = setup_matplotlib()

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(expert_counts, accuracies, 'o-', color='#2196F3', linewidth=2, markersize=8, markeredgecolor='white', markeredgewidth=1.5)

    for i, (x, y) in enumerate(zip(expert_counts, accuracies)):
        ax.annotate(f'{y:.2f}%', (x, y), textcoords="offset points", xytext=(0, 12), ha='center', fontsize=10)

    ax.set_xlabel('Number of Experts')
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Accuracy vs Number of Experts')
    ax.set_xticks(expert_counts)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    save_path = os.path.join(save_dir, 'accuracy_vs_experts.png')
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    return save_path


def plot_params_chart(expert_counts, param_counts, save_dir):
    plt = setup_matplotlib()

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = [EXPERT_COLORS[i % len(EXPERT_COLORS)] for i in range(len(expert_counts))]
    bars = ax.bar([str(x) for x in expert_counts], param_counts, color=colors, edgecolor='white', linewidth=1.2)

    for bar, val in zip(bars, param_counts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(param_counts) * 0.01,
                f'{val:,}', ha='center', va='bottom', fontsize=9)

    ax.set_xlabel('Number of Experts')
    ax.set_ylabel('Parameter Count')
    ax.set_title('Parameter Count vs Number of Experts')
    ax.grid(True, alpha=0.3, axis='y')
    fig.tight_layout()

    save_path = os.path.join(save_dir, 'params_vs_experts.png')
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    return save_path


def save_csv(expert_counts, accuracies, param_counts, best_epochs, save_dir):
    save_path = os.path.join(save_dir, 'ablation_results.csv')
    with open(save_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['num_experts', 'accuracy(%)', 'params', 'best_epoch'])
        for n, acc, params, ep in zip(expert_counts, accuracies, param_counts, best_epochs):
            writer.writerow([n, f'{acc:.2f}', params, ep])
    return save_path


def main():
    parser = argparse.ArgumentParser(description='Ablation Study: Number of Experts')
    parser.add_argument('--dataset', type=str, default='HandsData', choices=DATASETS)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--expert-range', type=str, default='1,2,3,4,5')
    args = parser.parse_args()

    expert_counts = [int(x.strip()) for x in args.expert_range.split(',')]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    save_dir = get_result_dir(args.dataset, 'ablation')

    print('=' * 60)
    print('Ablation Study: Number of Experts')
    print('=' * 60)
    print(f'Dataset: {args.dataset}')
    print(f'Expert counts: {expert_counts}')
    print(f'Epochs: {args.epochs}')
    print(f'Batch size: {args.batch_size}')
    print(f'Learning rate: {args.lr}')
    print(f'Device: {device}')
    print(f'Results dir: {save_dir}')
    print('=' * 60)

    accuracies = []
    param_counts = []
    best_epochs = []

    for num_experts in expert_counts:
        print(f'\n{"=" * 60}')
        print(f'Training with {num_experts} expert(s)')
        print(f'{"=" * 60}')

        enhancement_names = [ENHANCEMENT_EXPERT_CLASSES[i % len(ENHANCEMENT_EXPERT_CLASSES)].__name__ for i in range(num_experts)]
        fusion_names = [FUSION_EXPERT_CLASSES[i % len(FUSION_EXPERT_CLASSES)].__name__ for i in range(num_experts)]
        print(f'Enhancement experts: {enhancement_names}')
        print(f'Fusion experts: {fusion_names}')

        best_acc, num_params, best_epoch = train_and_evaluate(
            num_experts=num_experts,
            dataset_name=args.dataset,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            device=device,
        )

        accuracies.append(best_acc)
        param_counts.append(num_params)
        best_epochs.append(best_epoch)

        print(f'\nResults for {num_experts} expert(s):')
        print(f'  Best accuracy: {best_acc:.2f}% (Epoch {best_epoch})')
        print(f'  Parameters: {num_params:,}')

    print(f'\n{"=" * 60}')
    print('Ablation Study Complete')
    print(f'{"=" * 60}')
    print(f'{"Experts":>8} | {"Accuracy(%)":>12} | {"Params":>12} | {"Best Epoch":>10}')
    print('-' * 52)
    for n, acc, params, ep in zip(expert_counts, accuracies, param_counts, best_epochs):
        print(f'{n:>8} | {acc:>12.2f} | {params:>12,} | {ep:>10}')

    acc_chart = plot_accuracy_chart(expert_counts, accuracies, save_dir)
    print(f'\nAccuracy chart saved: {acc_chart}')

    params_chart = plot_params_chart(expert_counts, param_counts, save_dir)
    print(f'Params chart saved: {params_chart}')

    csv_path = save_csv(expert_counts, accuracies, param_counts, best_epochs, save_dir)
    print(f'CSV results saved: {csv_path}')


if __name__ == '__main__':
    main()
