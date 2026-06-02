import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import argparse
import math
import time
import csv
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from tqdm import tqdm

from models.vibe_net import VIBENet
from dataset import get_dataloader
import config
from experiments.utils import (
    setup_matplotlib, get_result_dir, get_test_loader, get_train_loader,
    collect_predictions, DATASETS,
)


ABLATION_CONFIGS = [
    {
        'name': 'w/o Label Smoothing',
        'label_smoothing': 0.0,
        'lb_weight': config.LOAD_BALANCE_WEIGHT,
        'arc_margin': config.ARC_MARGIN,
    },
    {
        'name': 'w/o Load Balance Loss',
        'label_smoothing': config.LABEL_SMOOTHING,
        'lb_weight': 0.0,
        'arc_margin': config.ARC_MARGIN,
    },
    {
        'name': 'ArcFace margin=0.3',
        'label_smoothing': config.LABEL_SMOOTHING,
        'lb_weight': config.LOAD_BALANCE_WEIGHT,
        'arc_margin': 0.3,
    },
    {
        'name': 'ArcFace margin=0.5',
        'label_smoothing': config.LABEL_SMOOTHING,
        'lb_weight': config.LOAD_BALANCE_WEIGHT,
        'arc_margin': 0.5,
    },
    {
        'name': 'ArcFace margin=0.7',
        'label_smoothing': config.LABEL_SMOOTHING,
        'lb_weight': config.LOAD_BALANCE_WEIGHT,
        'arc_margin': 0.7,
    },
    {
        'name': 'Full Model',
        'label_smoothing': config.LABEL_SMOOTHING,
        'lb_weight': config.LOAD_BALANCE_WEIGHT,
        'arc_margin': config.ARC_MARGIN,
    },
]


def compute_ce_loss(logits, labels, label_smoothing):
    if label_smoothing <= 0:
        return F.cross_entropy(logits, labels)
    log_probs = F.log_softmax(logits, dim=1)
    nll_loss = -log_probs.gather(1, labels.unsqueeze(1)).squeeze(1)
    smooth_loss = -log_probs.mean(dim=1)
    loss = (1.0 - label_smoothing) * nll_loss + label_smoothing * smooth_loss
    return loss.mean()


def set_lr(optimizer, epoch, base_lr, num_epochs, warmup_epochs, min_lr):
    if warmup_epochs > 0 and epoch < warmup_epochs:
        lr = base_lr * float(epoch + 1) / float(warmup_epochs)
    else:
        denom = max(1, num_epochs - warmup_epochs)
        progress = float(epoch - warmup_epochs) / float(denom)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        lr = min_lr + (base_lr - min_lr) * cosine
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def train_one_config(cfg, dataset_name, epochs, batch_size, lr, device):
    dataset_cfg = config.get_dataset_config(dataset_name)
    num_classes = dataset_cfg['num_classes']

    model = VIBENet(
        num_classes=num_classes,
        feature_dim=config.FEATURE_DIM,
        out_stages=config.OUT_STAGES,
        reducer_channels=config.REDUCER_CHANNELS,
        classifier_embed_dim=config.CLASSIFIER_EMBED_DIM,
        classifier_margin=cfg['arc_margin'],
        classifier_scale=config.ARC_SCALE,
        classifier_dropout=config.CLASSIFIER_DROPOUT,
    )
    model = model.to(device)

    train_loader = get_dataloader(
        dataset_name, mode='train', batch_size=batch_size,
        num_workers=0, shuffle=True,
    )
    test_loader = get_test_loader(dataset_name, batch_size=batch_size)

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=config.WEIGHT_DECAY)

    best_acc = 0.0
    for epoch in range(epochs):
        current_lr = set_lr(optimizer, epoch, lr, epochs, config.WARMUP_EPOCHS, config.MIN_LR)
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(train_loader, desc=f"[{cfg['name']}] Epoch {epoch+1}/{epochs}")
        for print_img, vein_img, labels in pbar:
            print_img = print_img.to(device)
            vein_img = vein_img.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(print_img, vein_img, labels=labels)
            ce_loss = compute_ce_loss(outputs, labels, cfg['label_smoothing'])
            lb_loss = model.compute_load_balancing_loss()
            loss = ce_loss + cfg['lb_weight'] * lb_loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            pbar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{100.*correct/total:.2f}%'})

        train_acc = 100. * correct / total

        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for print_img, vein_img, labels in test_loader:
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

        print(f"  [{cfg['name']}] Epoch {epoch+1}: train_acc={train_acc:.2f}%, val_acc={val_acc:.2f}%, best={best_acc:.2f}%")

    model.eval()
    final_acc, _, _, _ = collect_predictions(model, test_loader, device)
    return final_acc


def plot_comparison(results, save_dir):
    plt = setup_matplotlib()
    names = [r['name'] for r in results]
    accs = [r['accuracy'] for r in results]

    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#6C5CE7']

    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.bar(range(len(names)), accs, color=colors[:len(names)], edgecolor='black', linewidth=0.8)

    for bar, acc in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 0.1,
                f'{acc:.2f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')

    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=15, ha='right')
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Loss Function Ablation Study')
    ax.set_ylim(min(accs) - 2, max(accs) + 2)
    ax.grid(axis='y', alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, 'ablation_loss_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close(fig)


def save_csv(results, save_dir):
    csv_path = os.path.join(save_dir, 'ablation_loss_results.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Configuration', 'Label Smoothing', 'LB Weight', 'ArcFace Margin', 'Accuracy (%)'])
        for r in results:
            writer.writerow([r['name'], r['label_smoothing'], r['lb_weight'], r['arc_margin'], f"{r['accuracy']:.2f}"])
    return csv_path


def main():
    parser = argparse.ArgumentParser(description='VIBE-Net Loss Function Ablation Study')
    parser.add_argument('--dataset', type=str, default='HandsData', choices=DATASETS)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-4)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print(f"Dataset: {args.dataset}")
    print(f"Epochs: {args.epochs}, Batch Size: {args.batch_size}, LR: {args.lr}")
    print("=" * 60)

    results = []
    for cfg in ABLATION_CONFIGS:
        print(f"\n>>> Running: {cfg['name']}")
        print(f"    label_smoothing={cfg['label_smoothing']}, lb_weight={cfg['lb_weight']}, arc_margin={cfg['arc_margin']}")
        acc = train_one_config(cfg, args.dataset, args.epochs, args.batch_size, args.lr, device)
        results.append({
            'name': cfg['name'],
            'label_smoothing': cfg['label_smoothing'],
            'lb_weight': cfg['lb_weight'],
            'arc_margin': cfg['arc_margin'],
            'accuracy': acc,
        })
        print(f"    => Accuracy: {acc:.2f}%")

    save_dir = get_result_dir(args.dataset, 'ablation')

    plot_comparison(results, save_dir)
    print(f"\nBar chart saved to {os.path.join(save_dir, 'ablation_loss_comparison.png')}")

    csv_path = save_csv(results, save_dir)
    print(f"CSV saved to {csv_path}")

    print("\n" + "=" * 60)
    print("Ablation Results Summary:")
    print("-" * 60)
    print(f"{'Configuration':<25} {'Accuracy (%)':>12}")
    print("-" * 60)
    for r in results:
        print(f"{r['name']:<25} {r['accuracy']:>12.2f}")
    print("=" * 60)


if __name__ == '__main__':
    main()
