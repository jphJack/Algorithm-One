import os
import sys
import argparse
import csv
import random
import math
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

from models.vibe_net import VIBENet
from dataset import get_dataloader
import config
from utils import setup_matplotlib, get_result_dir, DATASETS

ABLATION_CONFIGS = [
    {'name': 'stages_3_4', 'out_stages': [3, 4], 'label': 'Stages [3,4]'},
    {'name': 'stages_4_5', 'out_stages': [4, 5], 'label': 'Stages [4,5]'},
    {'name': 'stages_3_4_5', 'out_stages': [3, 4, 5], 'label': 'Stages [3,4,5] (Default)'},
]


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
        torch.backends.cudnn.benchmark = True


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


def train_one_epoch(model, loader, optimizer, criterion, device, lb_weight):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for print_img, vein_img, labels in loader:
        print_img = print_img.to(device)
        vein_img = vein_img.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(print_img, vein_img, labels=labels)
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

    epoch_loss = running_loss / len(loader)
    epoch_acc = 100. * correct / total
    return epoch_loss, epoch_acc


def validate(model, loader, device):
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

    acc = 100. * correct / total
    return acc


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


def train_and_evaluate(out_stages, dataset_name, epochs, batch_size, lr, device):
    dataset_cfg = config.get_dataset_config(dataset_name)
    num_classes = dataset_cfg['num_classes']

    model = VIBENet(
        num_classes=num_classes,
        feature_dim=config.FEATURE_DIM,
        out_stages=out_stages,
        reducer_channels=config.REDUCER_CHANNELS,
        classifier_embed_dim=config.CLASSIFIER_EMBED_DIM,
        classifier_margin=config.ARC_MARGIN,
        classifier_scale=config.ARC_SCALE,
        classifier_dropout=config.CLASSIFIER_DROPOUT,
    )
    model = model.to(device)

    num_params = count_parameters(model)
    print(f"  Parameters: {num_params:,}")

    train_loader = get_dataloader(
        dataset_name, mode='train', batch_size=batch_size,
        num_workers=config.NUM_WORKERS, shuffle=True
    )
    val_loader = get_dataloader(
        dataset_name, mode='test', batch_size=batch_size,
        num_workers=config.NUM_WORKERS, shuffle=False
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=config.WEIGHT_DECAY)

    best_acc = 0.0
    best_epoch = -1

    for epoch in range(epochs):
        current_lr = set_lr(
            optimizer, epoch, lr, epochs,
            config.WARMUP_EPOCHS, config.MIN_LR
        )

        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, device,
            config.LOAD_BALANCE_WEIGHT
        )
        val_acc = validate(model, val_loader, device)

        if val_acc > best_acc:
            best_acc = val_acc
            best_epoch = epoch + 1

        print(f"  Epoch {epoch+1}/{epochs} | LR: {current_lr:.6f} | "
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
              f"Val Acc: {val_acc:.2f}% | Best: {best_acc:.2f}% (ep{best_epoch})")

    test_acc = validate(model, val_loader, device)

    return test_acc, num_params


def plot_comparison(results, save_dir):
    plt = setup_matplotlib()

    names = [r['label'] for r in results]
    accs = [r['accuracy'] for r in results]
    params = [r['params'] / 1e6 for r in results]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    colors = ['#2196F3', '#4CAF50', '#FF9800']

    bars1 = ax1.bar(names, accs, color=colors, edgecolor='black', linewidth=0.5)
    ax1.set_ylabel('Accuracy (%)')
    ax1.set_title('Stage Selection Ablation - Accuracy')
    ax1.set_ylim(min(accs) - 2, max(accs) + 1)
    for bar, acc in zip(bars1, accs):
        ax1.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 0.1,
                 f'{acc:.2f}%', ha='center', va='bottom', fontsize=10)
    ax1.grid(axis='y', alpha=0.3)

    bars2 = ax2.bar(names, params, color=colors, edgecolor='black', linewidth=0.5)
    ax2.set_ylabel('Parameters (M)')
    ax2.set_title('Stage Selection Ablation - Parameters')
    for bar, p in zip(bars2, params):
        ax2.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 0.01,
                 f'{p:.2f}M', ha='center', va='bottom', fontsize=10)
    ax2.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    fig.savefig(os.path.join(save_dir, 'ablation_stages_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close(fig)


def save_csv(results, save_dir):
    csv_path = os.path.join(save_dir, 'ablation_stages_results.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Config', 'out_stages', 'Accuracy (%)', 'Parameters'])
        for r in results:
            writer.writerow([r['name'], str(r['out_stages']), f"{r['accuracy']:.2f}", r['params']])
    return csv_path


def main():
    parser = argparse.ArgumentParser(description='VIBE-Net Stage Selection Ablation')
    parser.add_argument('--dataset', type=str, default='HandsData', choices=DATASETS)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-4)
    args = parser.parse_args()

    seed_everything(config.SEED, deterministic=config.DETERMINISTIC)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print(f"Dataset: {args.dataset}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    print("=" * 60)

    results = []

    for cfg in ABLATION_CONFIGS:
        print(f"\n{'=' * 60}")
        print(f"Config: {cfg['label']} (out_stages={cfg['out_stages']})")
        print(f"{'=' * 60}")

        start_time = time.time()
        test_acc, num_params = train_and_evaluate(
            out_stages=cfg['out_stages'],
            dataset_name=args.dataset,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            device=device,
        )
        elapsed = time.time() - start_time

        print(f"\n  Result: Accuracy={test_acc:.2f}%, Params={num_params:,}, Time={elapsed:.1f}s")

        results.append({
            'name': cfg['name'],
            'label': cfg['label'],
            'out_stages': cfg['out_stages'],
            'accuracy': test_acc,
            'params': num_params,
        })

    save_dir = get_result_dir(args.dataset, 'ablation')

    plot_comparison(results, save_dir)
    csv_path = save_csv(results, save_dir)

    print(f"\n{'=' * 60}")
    print("Ablation Study Summary - Stage Selection")
    print(f"{'=' * 60}")
    print(f"{'Config':<25} {'Accuracy (%)':<15} {'Parameters':<15}")
    print("-" * 55)
    for r in results:
        print(f"{r['label']:<25} {r['accuracy']:<15.2f} {r['params']:<15,}")
    print(f"\nResults saved to: {save_dir}")
    print(f"CSV: {csv_path}")
    print(f"Chart: {os.path.join(save_dir, 'ablation_stages_comparison.png')}")


if __name__ == '__main__':
    main()
