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

from models.vibe_net import VIBENet
from dataset import get_dataloader
import config
from experiments.utils import setup_matplotlib, get_result_dir, DATASETS

ABLATION_VARIANTS = {
    'print_only': 'Print-Only',
    'vein_only': 'Vein-Only',
    'dual_stream': 'Dual-Stream (Full)',
}

VARIANT_COLORS = {
    'print_only': '#1f77b4',
    'vein_only': '#ff7f0e',
    'dual_stream': '#2ca02c',
}


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


def apply_modality_mask(print_img, vein_img, variant):
    if variant == 'print_only':
        vein_img = torch.zeros_like(vein_img)
    elif variant == 'vein_only':
        print_img = torch.zeros_like(print_img)
    return print_img, vein_img


def train_one_epoch(model, train_loader, optimizer, device, variant, epoch, total_epochs):
    model.train()
    criterion = nn.CrossEntropyLoss()
    running_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{total_epochs}')
    for print_img, vein_img, labels in pbar:
        print_img, vein_img = apply_modality_mask(print_img, vein_img, variant)
        print_img = print_img.to(device)
        vein_img = vein_img.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(print_img, vein_img, labels=labels)

        if config.LABEL_SMOOTHING > 0:
            log_probs = F.log_softmax(outputs, dim=1)
            nll_loss = -log_probs.gather(1, labels.unsqueeze(1)).squeeze(1)
            smooth_loss = -log_probs.mean(dim=1)
            ce_loss = (1.0 - config.LABEL_SMOOTHING) * nll_loss + config.LABEL_SMOOTHING * smooth_loss
            ce_loss = ce_loss.mean()
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

    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100. * correct / total
    return epoch_loss, epoch_acc


def validate(model, val_loader, device, variant):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for print_img, vein_img, labels in tqdm(val_loader, desc='Validating'):
            print_img, vein_img = apply_modality_mask(print_img, vein_img, variant)
            print_img = print_img.to(device)
            vein_img = vein_img.to(device)
            labels = labels.to(device)

            outputs = model(print_img, vein_img)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    acc = 100. * correct / total
    return acc


def set_lr(optimizer, epoch, base_lr, num_epochs, warmup_epochs=5, min_lr=1e-6):
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


def train_variant(variant, dataset_name, epochs, batch_size, lr, device):
    dataset_cfg = config.get_dataset_config(dataset_name)
    num_classes = dataset_cfg['num_classes']

    train_loader = get_dataloader(dataset_name, mode='train', batch_size=batch_size,
                                  num_workers=config.NUM_WORKERS, shuffle=True)
    val_loader = get_dataloader(dataset_name, mode='test', batch_size=batch_size,
                                num_workers=config.NUM_WORKERS, shuffle=False)

    model = VIBENet(
        num_classes=num_classes, feature_dim=config.FEATURE_DIM,
        out_stages=config.OUT_STAGES, reducer_channels=config.REDUCER_CHANNELS,
        classifier_embed_dim=config.CLASSIFIER_EMBED_DIM,
        classifier_margin=config.ARC_MARGIN,
        classifier_scale=config.ARC_SCALE,
        classifier_dropout=config.CLASSIFIER_DROPOUT,
    )
    model = model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=config.WEIGHT_DECAY)

    best_acc = 0.0
    best_epoch = -1
    train_accs = []
    val_accs = []

    variant_label = ABLATION_VARIANTS[variant]
    print(f'\n{"="*60}')
    print(f'Training variant: {variant_label}')
    print(f'Dataset: {dataset_name} | Epochs: {epochs} | Batch size: {batch_size} | LR: {lr}')
    print(f'Total params: {total_params:,} | Trainable params: {trainable_params:,}')
    print(f'{"="*60}')

    start_time = time.time()

    for epoch in range(epochs):
        current_lr = set_lr(optimizer, epoch, lr, epochs,
                            warmup_epochs=config.WARMUP_EPOCHS, min_lr=config.MIN_LR)
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, device, variant, epoch, epochs)
        val_acc = validate(model, val_loader, device, variant)

        train_accs.append(train_acc)
        val_accs.append(val_acc)

        print(f'\nEpoch {epoch+1}/{epochs} | LR: {current_lr:.6f}')
        print(f'  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}%')

        if val_acc > best_acc:
            best_acc = val_acc
            best_epoch = epoch + 1

    elapsed = time.time() - start_time
    hours, rem = divmod(elapsed, 3600)
    minutes, seconds = divmod(rem, 60)

    print(f'\n{variant_label} training complete!')
    print(f'Time: {int(hours)}h {int(minutes)}m {seconds:.1f}s')
    print(f'Best Val Acc: {best_acc:.2f}% (Epoch {best_epoch})')

    return {
        'variant': variant,
        'label': variant_label,
        'best_val_acc': best_acc,
        'best_epoch': best_epoch,
        'final_train_acc': train_accs[-1],
        'total_params': total_params,
        'trainable_params': trainable_params,
        'train_accs': train_accs,
        'val_accs': val_accs,
    }


def plot_comparison(results, save_dir):
    plt = setup_matplotlib()

    variants = [r['variant'] for r in results]
    labels = [r['label'] for r in results]
    accuracies = [r['best_val_acc'] for r in results]
    colors = [VARIANT_COLORS[v] for v in variants]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(labels, accuracies, color=colors, width=0.5, edgecolor='black', linewidth=0.8)

    for bar, acc in zip(bars, accuracies):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                f'{acc:.2f}%', ha='center', va='bottom', fontsize=12, fontweight='bold')

    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Dual-Stream vs Single-Stream Ablation Study')
    ax.set_ylim(0, max(accuracies) * 1.1)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(save_dir, 'ablation_comparison.png')
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'Comparison bar chart saved to: {save_path}')


def plot_training_curves(results, save_dir):
    plt = setup_matplotlib()

    fig, ax = plt.subplots(figsize=(10, 6))
    for r in results:
        epochs_range = range(1, len(r['val_accs']) + 1)
        ax.plot(epochs_range, r['val_accs'], label=r['label'],
                color=VARIANT_COLORS[r['variant']], linewidth=2)

    ax.set_xlabel('Epoch')
    ax.set_ylabel('Validation Accuracy (%)')
    ax.set_title('Validation Accuracy Curves - Ablation Study')
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(save_dir, 'ablation_training_curves.png')
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'Training curves saved to: {save_path}')


def save_csv(results, save_dir):
    csv_path = os.path.join(save_dir, 'ablation_results.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Variant', 'Best Val Accuracy (%)', 'Best Epoch',
                         'Final Train Accuracy (%)', 'Total Params', 'Trainable Params'])
        for r in results:
            writer.writerow([
                r['label'],
                f"{r['best_val_acc']:.2f}",
                r['best_epoch'],
                f"{r['final_train_acc']:.2f}",
                r['total_params'],
                r['trainable_params'],
            ])
    print(f'CSV results saved to: {csv_path}')


def main():
    parser = argparse.ArgumentParser(description='VIBE-Net Dual-Stream Ablation Study')
    parser.add_argument('--dataset', type=str, default='HandsData', choices=DATASETS,
                        help='Dataset name')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    args = parser.parse_args()

    seed_everything(config.SEED, deterministic=config.DETERMINISTIC)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f'Device: {device}')
    print(f'Dataset: {args.dataset}')
    print(f'Epochs: {args.epochs} | Batch size: {args.batch_size} | LR: {args.lr}')

    save_dir = get_result_dir(args.dataset, 'ablation')

    variant_order = ['print_only', 'vein_only', 'dual_stream']
    results = []

    for variant in variant_order:
        result = train_variant(variant, args.dataset, args.epochs, args.batch_size, args.lr, device)
        results.append(result)

    plot_comparison(results, save_dir)
    plot_training_curves(results, save_dir)
    save_csv(results, save_dir)

    print(f'\n{"="*60}')
    print('Ablation Study Summary')
    print(f'{"="*60}')
    print(f'{"Variant":<25} {"Best Val Acc (%)":<20} {"Best Epoch":<12} {"Params":<15}')
    print('-' * 72)
    for r in results:
        print(f'{r["label"]:<25} {r["best_val_acc"]:<20.2f} {r["best_epoch"]:<12} {r["total_params"]:<15,}')
    print(f'{"="*60}')
    print(f'Results saved to: {save_dir}')


if __name__ == '__main__':
    main()
