import os
import sys
import argparse
import csv
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from models.vibe_net import VIBENet
from dataset import get_dataloader
import config
from utils import setup_matplotlib, get_result_dir, DATASETS


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


def train_and_evaluate(feature_dim, dataset_name, epochs, batch_size, lr, device):
    dataset_cfg = config.get_dataset_config(dataset_name)
    num_classes = dataset_cfg['num_classes']

    model = VIBENet(
        num_classes=num_classes,
        feature_dim=feature_dim,
        out_stages=config.OUT_STAGES,
        reducer_channels=config.REDUCER_CHANNELS,
        classifier_embed_dim=config.CLASSIFIER_EMBED_DIM,
        classifier_margin=config.ARC_MARGIN,
        classifier_scale=config.ARC_SCALE,
        classifier_dropout=config.CLASSIFIER_DROPOUT,
    )
    model = model.to(device)

    param_count = count_parameters(model)

    train_loader = get_dataloader(
        dataset_name, mode='train', batch_size=batch_size,
        num_workers=config.NUM_WORKERS, shuffle=True
    )
    val_loader = get_dataloader(
        dataset_name, mode='test', batch_size=batch_size,
        num_workers=config.NUM_WORKERS, shuffle=False
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    best_acc = 0.0

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(train_loader, desc=f'FeatureDim={feature_dim} Epoch {epoch+1}/{epochs}')
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
                'loss': f'{running_loss / (pbar.n + 1):.4f}',
                'acc': f'{100. * correct / total:.2f}%'
            })

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

        print(f'  FeatureDim={feature_dim} Epoch {epoch+1}: Val Acc={val_acc:.2f}% (Best={best_acc:.2f}%)')

    return best_acc, param_count


def plot_accuracy_vs_dim(dims, accuracies, save_dir):
    plt = setup_matplotlib()
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(dims, accuracies, 'o-', color='#1f77b4', linewidth=2, markersize=8)
    for dim, acc in zip(dims, accuracies):
        ax.annotate(f'{acc:.2f}%', (dim, acc), textcoords="offset points",
                    xytext=(0, 12), ha='center', fontsize=10)
    ax.set_xlabel('Feature Dimension')
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Accuracy vs Feature Dimension')
    ax.set_xticks(dims)
    ax.grid(True, linestyle='--', alpha=0.7)
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, 'accuracy_vs_feature_dim.png'), dpi=300, bbox_inches='tight')
    plt.close(fig)


def plot_params_vs_dim(dims, param_counts, save_dir):
    plt = setup_matplotlib()
    fig, ax = plt.subplots(figsize=(8, 5))
    param_millions = [p / 1e6 for p in param_counts]
    bars = ax.bar([str(d) for d in dims], param_millions, color=['#2196F3', '#4CAF50', '#FF9800'], width=0.5)
    for bar, pm in zip(bars, param_millions):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f'{pm:.2f}M', ha='center', va='bottom', fontsize=10)
    ax.set_xlabel('Feature Dimension')
    ax.set_ylabel('Parameter Count (M)')
    ax.set_title('Parameter Count vs Feature Dimension')
    ax.grid(True, linestyle='--', alpha=0.7, axis='y')
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, 'params_vs_feature_dim.png'), dpi=300, bbox_inches='tight')
    plt.close(fig)


def save_csv(dims, accuracies, param_counts, save_dir):
    csv_path = os.path.join(save_dir, 'ablation_feature_dim.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['feature_dim', 'accuracy(%)', 'param_count'])
        for dim, acc, params in zip(dims, accuracies, param_counts):
            writer.writerow([dim, f'{acc:.2f}', params])
    return csv_path


def main():
    parser = argparse.ArgumentParser(description='Feature Dimension Ablation for VIBE-Net')
    parser.add_argument('--dataset', type=str, default='HandsData', choices=DATASETS,
                        help='Dataset name')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--dims', type=str, default='128,256,512',
                        help='Comma-separated feature dimensions to test')
    args = parser.parse_args()

    dims = [int(d.strip()) for d in args.dims.split(',')]

    seed_everything(config.SEED, deterministic=config.DETERMINISTIC)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print('=' * 60)
    print('Feature Dimension Ablation Study')
    print('=' * 60)
    print(f'Dataset: {args.dataset}')
    print(f'Epochs: {args.epochs}')
    print(f'Batch size: {args.batch_size}')
    print(f'Learning rate: {args.lr}')
    print(f'Feature dimensions: {dims}')
    print(f'Device: {device}')
    print('=' * 60)

    accuracies = []
    param_counts = []

    for dim in dims:
        print(f'\n--- Training with feature_dim={dim} ---')
        acc, params = train_and_evaluate(
            feature_dim=dim,
            dataset_name=args.dataset,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            device=device,
        )
        accuracies.append(acc)
        param_counts.append(params)
        print(f'  Result: feature_dim={dim}, accuracy={acc:.2f}%, params={params:,}')

    save_dir = get_result_dir(args.dataset, 'ablation')

    plot_accuracy_vs_dim(dims, accuracies, save_dir)
    print(f'Saved accuracy chart to {save_dir}/accuracy_vs_feature_dim.png')

    plot_params_vs_dim(dims, param_counts, save_dir)
    print(f'Saved parameter count chart to {save_dir}/params_vs_feature_dim.png')

    csv_path = save_csv(dims, accuracies, param_counts, save_dir)
    print(f'Saved CSV to {csv_path}')

    print('\n' + '=' * 60)
    print('Ablation Results Summary')
    print('=' * 60)
    print(f'{"Feature Dim":>12} | {"Accuracy (%)":>12} | {"Param Count":>12}')
    print('-' * 42)
    for dim, acc, params in zip(dims, accuracies, param_counts):
        print(f'{dim:>12} | {acc:>12.2f} | {params:>12,}')
    print('=' * 60)


if __name__ == '__main__':
    main()
