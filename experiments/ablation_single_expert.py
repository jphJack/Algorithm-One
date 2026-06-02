import os
import sys
import argparse
import random
import math
import csv
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
from copy import deepcopy

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from models.moe_enhancement import MoEEnhancement, HighFreqExpert, MidFreqExpert, LowFreqExpert, GateNetwork
from models.moe_fusion import MoEFusion, CrossAttentionExpert, MultiScaleConvExpert, ChannelInteractionExpert, FusionGateNetwork
from models.vibe_net import VIBENet
from models.backbone import DualStreamBackbone
from dataset import get_dataloader
import config
from experiments.utils import (
    setup_matplotlib, get_result_dir, DATASETS,
    ENHANCEMENT_EXPERT_NAMES, FUSION_EXPERT_NAMES, EXPERT_COLORS
)


ENHANCEMENT_EXPERTS = [HighFreqExpert, MidFreqExpert, LowFreqExpert]
FUSION_EXPERTS = [CrossAttentionExpert, MultiScaleConvExpert, ChannelInteractionExpert]


class AblationMoEEnhancement(nn.Module):
    def __init__(self, channels, expert_classes):
        super(AblationMoEEnhancement, self).__init__()
        self.num_experts = len(expert_classes)
        self.experts = nn.ModuleList([cls(channels) for cls in expert_classes])
        self.gate = GateNetwork(channels, self.num_experts)
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
    def __init__(self, channels, expert_classes):
        super(AblationMoEFusion, self).__init__()
        self.num_experts = len(expert_classes)
        self.experts = nn.ModuleList([cls(channels) for cls in expert_classes])
        self.gate = FusionGateNetwork(channels, self.num_experts)
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


class AblationDualStreamBackbone(DualStreamBackbone):
    def __init__(self, in_channels=3, feature_dim=256, out_stages=None,
                 reducer_channels=64, enhancement_expert_classes=None):
        if out_stages is None:
            out_stages = [3, 4, 5]
        nn.Module.__init__(self)

        from models.backbone import LightweightBackbone
        self.print_backbone = LightweightBackbone(in_channels, feature_dim, out_stages)
        self.vein_backbone = LightweightBackbone(in_channels, feature_dim, out_stages)

        stage_channels = {k: self.print_backbone.stage_channels[k] for k in out_stages}

        self.enable_stage_enhancement = True
        self.out_stages = out_stages
        self.fusion_stage = max(out_stages)
        self.use_multiscale_extractor = False
        self.print_extractor = None
        self.vein_extractor = None
        self.out_channels = feature_dim

        if enhancement_expert_classes is not None:
            self.print_stage_enhancers = nn.ModuleDict({
                str(k): AblationMoEEnhancement(stage_channels[k], enhancement_expert_classes)
                for k in out_stages
            })
            self.vein_stage_enhancers = nn.ModuleDict({
                str(k): AblationMoEEnhancement(stage_channels[k], enhancement_expert_classes)
                for k in out_stages
            })
        else:
            self.print_stage_enhancers = nn.ModuleDict({
                str(k): MoEEnhancement(stage_channels[k])
                for k in out_stages
            })
            self.vein_stage_enhancers = nn.ModuleDict({
                str(k): MoEEnhancement(stage_channels[k])
                for k in out_stages
            })


class AblationVIBENet(nn.Module):
    def __init__(self, num_classes=290, feature_dim=256, out_stages=None,
                 reducer_channels=64, enhancement_expert_classes=None,
                 fusion_expert_classes=None,
                 classifier_embed_dim=256, classifier_margin=0.5,
                 classifier_scale=30.0, classifier_dropout=0.5):
        super(AblationVIBENet, self).__init__()

        self.backbone = AblationDualStreamBackbone(
            in_channels=3, feature_dim=feature_dim,
            out_stages=out_stages, reducer_channels=reducer_channels,
            enhancement_expert_classes=enhancement_expert_classes,
        )

        if fusion_expert_classes is not None:
            self.fusion = AblationMoEFusion(feature_dim, fusion_expert_classes)
        else:
            self.fusion = MoEFusion(feature_dim)

        from models.classifier import Classifier
        self.classifier = Classifier(
            feature_dim, num_classes,
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
        torch.backends.cudnn.benchmark = config.CUDNN_BENCHMARK


def build_enhancement_variants():
    variants = {}
    for i, name in enumerate(ENHANCEMENT_EXPERT_NAMES):
        remaining = [cls for j, cls in enumerate(ENHANCEMENT_EXPERTS) if j != i]
        variants[f'w/o {name}'] = remaining
    return variants


def build_fusion_variants():
    variants = {}
    for i, name in enumerate(FUSION_EXPERT_NAMES):
        remaining = [cls for j, cls in enumerate(FUSION_EXPERTS) if j != i]
        variants[f'w/o {name}'] = remaining
    return variants


def train_variant(model, train_loader, val_loader, device, epochs, lr, variant_name):
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=config.WEIGHT_DECAY)
    best_acc = 0.0

    for epoch in range(epochs):
        if config.WARMUP_EPOCHS > 0 and epoch < config.WARMUP_EPOCHS:
            current_lr = lr * float(epoch + 1) / float(config.WARMUP_EPOCHS)
        else:
            denom = max(1, epochs - config.WARMUP_EPOCHS)
            progress = float(epoch - config.WARMUP_EPOCHS) / float(denom)
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
            current_lr = config.MIN_LR + (lr - config.MIN_LR) * cosine
        for param_group in optimizer.param_groups:
            param_group['lr'] = current_lr

        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(train_loader, desc=f'[{variant_name}] Epoch {epoch+1}/{epochs}')
        for print_img, vein_img, labels in pbar:
            print_img = print_img.to(device)
            vein_img = vein_img.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(print_img, vein_img, labels=labels)

            if config.LABEL_SMOOTHING > 0:
                log_probs = F.log_softmax(outputs, dim=1)
                nll_loss = -log_probs.gather(1, labels.unsqueeze(1)).squeeze(1)
                smooth_loss = -log_probs.mean(dim=1)
                ce_loss = ((1.0 - config.LABEL_SMOOTHING) * nll_loss + config.LABEL_SMOOTHING * smooth_loss).mean()
            else:
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

            pbar.set_postfix({'loss': f'{running_loss/len(pbar):.4f}', 'acc': f'{100.*correct/total:.2f}%'})

        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for print_img, vein_img, labels in val_loader:
                print_img = print_img.to(device)
                vein_img = vein_img.to(device)
                labels = labels.to(device)
                outputs = model(print_img, vein_img)
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()

        val_acc = 100. * val_correct / val_total
        if val_acc > best_acc:
            best_acc = val_acc

        print(f'[{variant_name}] Epoch {epoch+1}/{epochs} - Val Acc: {val_acc:.2f}% (Best: {best_acc:.2f}%)')

    return best_acc


def evaluate_model(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for print_img, vein_img, labels in test_loader:
            print_img = print_img.to(device)
            vein_img = vein_img.to(device)
            labels = labels.to(device)
            outputs = model(print_img, vein_img)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    return 100. * correct / total


def plot_ablation_results(results, save_dir, dataset_name):
    plt = setup_matplotlib()

    enhancement_variants = results['enhancement']
    fusion_variants = results['fusion']

    enh_names = list(enhancement_variants.keys())
    enh_accs = list(enhancement_variants.values())
    fus_names = list(fusion_variants.keys())
    fus_accs = list(fusion_variants.values())

    all_names = enh_names + fus_names
    all_accs = enh_accs + fus_accs
    n_enh = len(enh_names)
    n_fus = len(fus_names)

    fig, ax = plt.subplots(figsize=(12, 6))

    x = np.arange(len(all_names))
    bar_width = 0.6

    colors = []
    for i in range(n_enh):
        idx = ENHANCEMENT_EXPERT_NAMES.index(enh_names[i].replace('w/o ', ''))
        colors.append(EXPERT_COLORS[idx])
    for i in range(n_fus):
        idx = FUSION_EXPERT_NAMES.index(fus_names[i].replace('w/o ', ''))
        colors.append(EXPERT_COLORS[idx])

    bars = ax.bar(x, all_accs, width=bar_width, color=colors, edgecolor='black', linewidth=0.8)

    for bar, acc in zip(bars, all_accs):
        ax.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 0.3,
                f'{acc:.2f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels(all_names, rotation=30, ha='right')
    ax.set_ylabel('Accuracy (%)')
    ax.set_title(f'Single Expert Ablation - {dataset_name}')

    ax.axvline(x=n_enh - 0.5, color='gray', linestyle='--', linewidth=1.0, alpha=0.7)

    ax.text(n_enh / 2 - 0.5, ax.get_ylim()[1] * 0.02, 'Enhancement MoE',
            ha='center', va='bottom', fontsize=11, fontstyle='italic', color='#555555')
    ax.text(n_enh + n_fus / 2 - 0.5, ax.get_ylim()[1] * 0.02, 'Fusion MoE',
            ha='center', va='bottom', fontsize=11, fontstyle='italic', color='#555555')

    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim(bottom=0)

    fig.tight_layout()
    save_path = os.path.join(save_dir, 'ablation_single_expert.png')
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved ablation chart to: {save_path}")


def save_csv(results, save_dir, dataset_name):
    save_path = os.path.join(save_dir, 'ablation_single_expert.csv')
    with open(save_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Module', 'Variant', 'Accuracy (%)'])
        for variant, acc in results['enhancement'].items():
            writer.writerow(['Enhancement MoE', variant, f'{acc:.2f}'])
        for variant, acc in results['fusion'].items():
            writer.writerow(['Fusion MoE', variant, f'{acc:.2f}'])
    print(f"Saved CSV to: {save_path}")


def main():
    parser = argparse.ArgumentParser(description='VIBE-Net Single Expert Ablation Study')
    parser.add_argument('--dataset', type=str, default='HandsData', choices=DATASETS,
                        help='Dataset for ablation')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of training epochs per variant')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')
    args = parser.parse_args()

    seed_everything(config.SEED, deterministic=config.DETERMINISTIC)

    dataset_cfg = config.get_dataset_config(args.dataset)
    num_classes = dataset_cfg['num_classes']
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print('=' * 60)
    print('VIBE-Net Single Expert Ablation Study')
    print('=' * 60)
    print(f'Dataset: {args.dataset}')
    print(f'Num classes: {num_classes}')
    print(f'Device: {device}')
    print(f'Epochs: {args.epochs}')
    print(f'Batch size: {args.batch_size}')
    print(f'Learning rate: {args.lr}')
    print('=' * 60)

    train_loader = get_dataloader(
        args.dataset, mode='train', batch_size=args.batch_size,
        num_workers=config.NUM_WORKERS, shuffle=True
    )
    test_loader = get_dataloader(
        args.dataset, mode='test', batch_size=args.batch_size,
        num_workers=config.NUM_WORKERS, shuffle=False
    )

    save_dir = get_result_dir(args.dataset, 'ablation')

    enhancement_variants = build_enhancement_variants()
    fusion_variants = build_fusion_variants()

    results = {'enhancement': {}, 'fusion': {}}

    model_kwargs = dict(
        num_classes=num_classes,
        feature_dim=config.FEATURE_DIM,
        out_stages=config.OUT_STAGES,
        reducer_channels=config.REDUCER_CHANNELS,
        classifier_embed_dim=config.CLASSIFIER_EMBED_DIM,
        classifier_margin=config.ARC_MARGIN,
        classifier_scale=config.ARC_SCALE,
        classifier_dropout=config.CLASSIFIER_DROPOUT,
    )

    print('\n--- Enhancement MoE Ablation ---')
    for variant_name, expert_classes in enhancement_variants.items():
        print(f'\nTraining variant: {variant_name}')
        model = AblationVIBENet(
            **model_kwargs,
            enhancement_expert_classes=expert_classes,
        )
        total_params = sum(p.numel() for p in model.parameters())
        print(f'Parameters: {total_params:,}')
        best_acc = train_variant(model, train_loader, test_loader, device,
                                 args.epochs, args.lr, variant_name)
        results['enhancement'][variant_name] = best_acc
        print(f'{variant_name} -> Best Accuracy: {best_acc:.2f}%')

    print('\n--- Fusion MoE Ablation ---')
    for variant_name, expert_classes in fusion_variants.items():
        print(f'\nTraining variant: {variant_name}')
        model = AblationVIBENet(
            **model_kwargs,
            fusion_expert_classes=expert_classes,
        )
        total_params = sum(p.numel() for p in model.parameters())
        print(f'Parameters: {total_params:,}')
        best_acc = train_variant(model, train_loader, test_loader, device,
                                 args.epochs, args.lr, variant_name)
        results['fusion'][variant_name] = best_acc
        print(f'{variant_name} -> Best Accuracy: {best_acc:.2f}%')

    plot_ablation_results(results, save_dir, args.dataset)
    save_csv(results, save_dir, args.dataset)

    print('\n' + '=' * 60)
    print('Ablation Results Summary')
    print('=' * 60)
    print(f'{"Module":<20} {"Variant":<25} {"Accuracy (%)":<15}')
    print('-' * 60)
    for variant, acc in results['enhancement'].items():
        print(f'{"Enhancement MoE":<20} {variant:<25} {acc:<15.2f}')
    for variant, acc in results['fusion'].items():
        print(f'{"Fusion MoE":<20} {variant:<25} {acc:<15.2f}')
    print('=' * 60)
    print('Done.')


if __name__ == '__main__':
    main()
