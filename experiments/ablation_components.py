import os
import sys
import math
import random
import argparse
import time
import csv

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from models.vibe_net import VIBENet
from models.backbone import DualStreamBackbone
from models.moe_fusion import MoEFusion
from models.classifier import Classifier, ArcMarginProduct
from dataset import get_dataloader
import config
from experiments.utils import setup_matplotlib, get_result_dir, DATASETS


class SimpleAddFusion(nn.Module):
    def __init__(self, channels, num_experts=3):
        super(SimpleAddFusion, self).__init__()
        self.channels = channels

    def load_balancing_loss(self):
        return torch.tensor(0.0, device=next(self.parameters()).device)

    def forward(self, f_p, f_v, return_gate_weights=False):
        out = f_p + f_v
        if return_gate_weights:
            return out, None
        return out


class SimpleClassifier(nn.Module):
    def __init__(self, in_channels, num_classes, embed_dim=256, margin=0.5, scale=30.0, dropout=0.5):
        super(SimpleClassifier, self).__init__()
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(in_channels, num_classes)

    def extract_embedding(self, x):
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        return x

    def forward(self, x, labels=None, return_embedding=False):
        embedding = self.extract_embedding(x)
        logits = self.fc(embedding)
        if return_embedding:
            return logits, embedding
        return logits


def create_full_model(num_classes, feature_dim, out_stages, reducer_channels,
                      classifier_embed_dim, classifier_margin, classifier_scale,
                      classifier_dropout, moe_num_experts):
    return VIBENet(
        num_classes=num_classes,
        feature_dim=feature_dim,
        out_stages=out_stages,
        reducer_channels=reducer_channels,
        moe_num_experts=moe_num_experts,
        classifier_embed_dim=classifier_embed_dim,
        classifier_margin=classifier_margin,
        classifier_scale=classifier_scale,
        classifier_dropout=classifier_dropout,
    )


def create_no_moe_enhancement_model(num_classes, feature_dim, out_stages, reducer_channels,
                                     classifier_embed_dim, classifier_margin, classifier_scale,
                                     classifier_dropout, moe_num_experts):
    model = VIBENet(
        num_classes=num_classes,
        feature_dim=feature_dim,
        out_stages=out_stages,
        reducer_channels=reducer_channels,
        moe_num_experts=moe_num_experts,
        classifier_embed_dim=classifier_embed_dim,
        classifier_margin=classifier_margin,
        classifier_scale=classifier_scale,
        classifier_dropout=classifier_dropout,
    )
    model.backbone = DualStreamBackbone(
        in_channels=3, feature_dim=feature_dim,
        out_stages=out_stages, reducer_channels=reducer_channels,
        moe_num_experts=moe_num_experts,
        enable_stage_enhancement=False,
        use_multiscale_extractor=False,
    )
    return model


def create_no_moe_fusion_model(num_classes, feature_dim, out_stages, reducer_channels,
                                classifier_embed_dim, classifier_margin, classifier_scale,
                                classifier_dropout, moe_num_experts):
    model = VIBENet(
        num_classes=num_classes,
        feature_dim=feature_dim,
        out_stages=out_stages,
        reducer_channels=reducer_channels,
        moe_num_experts=moe_num_experts,
        classifier_embed_dim=classifier_embed_dim,
        classifier_margin=classifier_margin,
        classifier_scale=classifier_scale,
        classifier_dropout=classifier_dropout,
    )
    model.fusion = SimpleAddFusion(feature_dim)
    return model


def create_no_arcface_model(num_classes, feature_dim, out_stages, reducer_channels,
                             classifier_embed_dim, classifier_margin, classifier_scale,
                             classifier_dropout, moe_num_experts):
    model = VIBENet(
        num_classes=num_classes,
        feature_dim=feature_dim,
        out_stages=out_stages,
        reducer_channels=reducer_channels,
        moe_num_experts=moe_num_experts,
        classifier_embed_dim=classifier_embed_dim,
        classifier_margin=classifier_margin,
        classifier_scale=classifier_scale,
        classifier_dropout=classifier_dropout,
    )
    model.classifier = SimpleClassifier(feature_dim, num_classes)
    return model


def create_no_lb_loss_model(num_classes, feature_dim, out_stages, reducer_channels,
                             classifier_embed_dim, classifier_margin, classifier_scale,
                             classifier_dropout, moe_num_experts):
    return VIBENet(
        num_classes=num_classes,
        feature_dim=feature_dim,
        out_stages=out_stages,
        reducer_channels=reducer_channels,
        moe_num_experts=moe_num_experts,
        classifier_embed_dim=classifier_embed_dim,
        classifier_margin=classifier_margin,
        classifier_scale=classifier_scale,
        classifier_dropout=classifier_dropout,
    )


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


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


def train_model(model, train_loader, val_loader, device, epochs, lr, lb_weight):
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=config.WEIGHT_DECAY)

    best_acc = 0.0
    best_state = None

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

        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}')
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
            loss = ce_loss + lb_weight * lb_loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            pbar.set_postfix({'loss': f'{running_loss/len(pbar):.4f}', 'acc': f'{100.*correct/total:.2f}%'})

        val_acc = evaluate(model, val_loader, device)

        if val_acc > best_acc:
            best_acc = val_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        print(f'Epoch {epoch+1}/{epochs} - Train Acc: {100.*correct/total:.2f}% - Val Acc: {val_acc:.2f}% - LR: {current_lr:.6f}')

    if best_state is not None:
        model.load_state_dict(best_state)
    model = model.to(device)

    final_acc = evaluate(model, val_loader, device)
    return model, final_acc


def evaluate(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for print_img, vein_img, labels in loader:
            print_img = print_img.to(device)
            vein_img = vein_img.to(device)
            labels = labels.to(device)
            outputs = model(print_img, vein_img)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    return 100. * correct / total


def plot_ablation_chart(variant_names, accuracies, param_counts, save_dir):
    plt = setup_matplotlib()

    fig, ax1 = plt.subplots(figsize=(10, 6))

    x = np.arange(len(variant_names))
    bar_width = 0.5

    bars = ax1.bar(x, accuracies, bar_width, color=['#2196F3', '#FF9800', '#4CAF50', '#E91E63', '#9C27B0'],
                   edgecolor='black', linewidth=0.8)

    for bar, acc, params in zip(bars, accuracies, param_counts):
        ax1.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 0.3,
                 f'{acc:.2f}%\n({params/1e6:.2f}M)', ha='center', va='bottom', fontsize=9)

    ax1.set_xlabel('Model Variant')
    ax1.set_ylabel('Accuracy (%)')
    ax1.set_title('Component Ablation Study')
    ax1.set_xticks(x)
    ax1.set_xticklabels(variant_names, rotation=15, ha='right')
    ax1.set_ylim(min(accuracies) - 5, max(accuracies) + 5)
    ax1.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(save_dir, 'ablation_components.png')
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'Bar chart saved to: {save_path}')


def save_csv(variant_names, accuracies, param_counts, save_dir):
    csv_path = os.path.join(save_dir, 'ablation_components.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Variant', 'Accuracy (%)', 'Parameters'])
        for name, acc, params in zip(variant_names, accuracies, param_counts):
            writer.writerow([name, f'{acc:.2f}', f'{params:,}'])
    print(f'CSV saved to: {csv_path}')


ABLATION_VARIANTS = {
    'Full Model': {
        'create_fn': create_full_model,
        'lb_weight': config.LOAD_BALANCE_WEIGHT,
    },
    'w/o MoE Enhancement': {
        'create_fn': create_no_moe_enhancement_model,
        'lb_weight': config.LOAD_BALANCE_WEIGHT,
    },
    'w/o MoE Fusion': {
        'create_fn': create_no_moe_fusion_model,
        'lb_weight': config.LOAD_BALANCE_WEIGHT,
    },
    'w/o ArcFace': {
        'create_fn': create_no_arcface_model,
        'lb_weight': config.LOAD_BALANCE_WEIGHT,
    },
    'w/o Load Balance Loss': {
        'create_fn': create_no_lb_loss_model,
        'lb_weight': 0.0,
    },
}


def main():
    parser = argparse.ArgumentParser(description='VIBE-Net Component Ablation Study')
    parser.add_argument('--dataset', type=str, default='HandsData', choices=DATASETS,
                        help='Dataset name')
    parser.add_argument('--epochs', type=int, default=None,
                        help='Number of training epochs (default: from config)')
    parser.add_argument('--batch-size', type=int, default=None,
                        help='Batch size (default: from config)')
    parser.add_argument('--lr', type=float, default=None,
                        help='Learning rate (default: from config)')
    parser.add_argument('--save-dir', type=str, default=None,
                        help='Directory to save results')
    args = parser.parse_args()

    seed_everything(config.SEED, deterministic=config.DETERMINISTIC)

    dataset_name = args.dataset
    epochs = args.epochs if args.epochs is not None else config.NUM_EPOCHS
    batch_size = args.batch_size if args.batch_size is not None else config.BATCH_SIZE
    lr = args.lr if args.lr is not None else config.LEARNING_RATE

    dataset_cfg = config.get_dataset_config(dataset_name)
    num_classes = dataset_cfg['num_classes']

    if args.save_dir is not None:
        save_dir = args.save_dir
    else:
        save_dir = get_result_dir(dataset_name, 'ablation')
    os.makedirs(save_dir, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')
    print(f'Dataset: {dataset_name} ({num_classes} classes)')
    print(f'Epochs: {epochs}, Batch size: {batch_size}, LR: {lr}')
    print(f'Results dir: {save_dir}')
    print('=' * 60)

    train_loader = get_dataloader(dataset_name, mode='train', batch_size=batch_size,
                                  num_workers=config.NUM_WORKERS, shuffle=True)
    val_loader = get_dataloader(dataset_name, mode='test', batch_size=batch_size,
                                num_workers=config.NUM_WORKERS, shuffle=False)

    model_kwargs = dict(
        num_classes=num_classes,
        feature_dim=config.FEATURE_DIM,
        out_stages=config.OUT_STAGES,
        reducer_channels=config.REDUCER_CHANNELS,
        classifier_embed_dim=config.CLASSIFIER_EMBED_DIM,
        classifier_margin=config.ARC_MARGIN,
        classifier_scale=config.ARC_SCALE,
        classifier_dropout=config.CLASSIFIER_DROPOUT,
        moe_num_experts=config.NUM_EXPERTS,
    )

    variant_names = []
    accuracies = []
    param_counts = []

    for variant_name, variant_cfg in ABLATION_VARIANTS.items():
        print(f'\n{"=" * 60}')
        print(f'Training variant: {variant_name}')
        print(f'{"=" * 60}')

        seed_everything(config.SEED, deterministic=config.DETERMINISTIC)

        model = variant_cfg['create_fn'](**model_kwargs)
        num_params = count_parameters(model)
        print(f'Parameters: {num_params:,}')

        lb_weight = variant_cfg['lb_weight']

        start_time = time.time()
        model, best_acc = train_model(model, train_loader, val_loader, device, epochs, lr, lb_weight)
        elapsed = time.time() - start_time

        print(f'\n{variant_name} - Best Val Acc: {best_acc:.2f}% - Params: {num_params:,} - Time: {elapsed:.1f}s')

        variant_names.append(variant_name)
        accuracies.append(best_acc)
        param_counts.append(num_params)

        ckpt_dir = os.path.join(save_dir, variant_name.replace(' ', '_').replace('/', ''))
        os.makedirs(ckpt_dir, exist_ok=True)
        torch.save(model.state_dict(), os.path.join(ckpt_dir, 'best_model.pth'))

    print(f'\n{"=" * 60}')
    print('Ablation Study Results Summary')
    print(f'{"=" * 60}')
    print(f'{"Variant":<25} {"Accuracy (%)":<15} {"Parameters":<15}')
    print('-' * 55)
    for name, acc, params in zip(variant_names, accuracies, param_counts):
        print(f'{name:<25} {acc:<15.2f} {params:<15,}')

    plot_ablation_chart(variant_names, accuracies, param_counts, save_dir)
    save_csv(variant_names, accuracies, param_counts, save_dir)

    print(f'\nAll results saved to: {save_dir}')


if __name__ == '__main__':
    main()
