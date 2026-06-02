import sys
import os
import argparse
import csv
import random
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from models.vibe_net import VIBENet
from dataset import get_dataloader
import config
from experiments.utils import setup_matplotlib, get_result_dir, DATASETS


class AddFusion(nn.Module):
    def __init__(self, channels, **kwargs):
        super(AddFusion, self).__init__()

    def load_balancing_loss(self):
        return 0.0

    def forward(self, f_p, f_v, return_gate_weights=False):
        out = f_p + f_v
        if return_gate_weights:
            return out, None
        return out


class ConcatFusion(nn.Module):
    def __init__(self, channels, **kwargs):
        super(ConcatFusion, self).__init__()
        self.conv = nn.Conv2d(channels * 2, channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(channels)

    def load_balancing_loss(self):
        return 0.0

    def forward(self, f_p, f_v, return_gate_weights=False):
        out = torch.cat([f_p, f_v], dim=1)
        out = self.conv(out)
        out = self.bn(out)
        if return_gate_weights:
            return out, None
        return out


class WeightedAvgFusion(nn.Module):
    def __init__(self, channels, **kwargs):
        super(WeightedAvgFusion, self).__init__()
        self.raw_alpha = nn.Parameter(torch.tensor(0.0))

    def load_balancing_loss(self):
        return 0.0

    def forward(self, f_p, f_v, return_gate_weights=False):
        alpha = torch.sigmoid(self.raw_alpha)
        out = alpha * f_p + (1 - alpha) * f_v
        if return_gate_weights:
            return out, None
        return out


FUSION_STRATEGIES = {
    'Add': AddFusion,
    'Concat': ConcatFusion,
    'WeightedAvg': WeightedAvgFusion,
}

STRATEGY_COLORS = {
    'Add': '#2196F3',
    'Concat': '#4CAF50',
    'WeightedAvg': '#FF9800',
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


def train_and_evaluate(fusion_cls, fusion_name, dataset_name, epochs, batch_size, lr, device):
    dataset_cfg = config.get_dataset_config(dataset_name)
    num_classes = dataset_cfg['num_classes']

    train_loader = get_dataloader(
        dataset_name, mode='train', batch_size=batch_size,
        num_workers=config.NUM_WORKERS, shuffle=True
    )
    test_loader = get_dataloader(
        dataset_name, mode='test', batch_size=batch_size,
        num_workers=config.NUM_WORKERS, shuffle=False
    )

    model = VIBENet(
        num_classes=num_classes,
        feature_dim=config.FEATURE_DIM,
        out_stages=config.OUT_STAGES,
        reducer_channels=config.REDUCER_CHANNELS,
        classifier_embed_dim=config.CLASSIFIER_EMBED_DIM,
        classifier_margin=config.ARC_MARGIN,
        classifier_scale=config.ARC_SCALE,
        classifier_dropout=config.CLASSIFIER_DROPOUT,
    )
    model.fusion = fusion_cls(config.FEATURE_DIM)
    model = model.to(device)

    save_dir = os.path.join(
        config.BASE_DIR, 'experiment_results', dataset_name,
        'ablation', f'ckpt_{fusion_name}'
    )
    os.makedirs(save_dir, exist_ok=True)

    original_epochs = config.NUM_EPOCHS
    original_lr = config.LEARNING_RATE
    config.NUM_EPOCHS = epochs
    config.LEARNING_RATE = lr

    from train import Trainer
    trainer = Trainer(model, train_loader, test_loader, device, save_dir)
    trainer.train()

    config.NUM_EPOCHS = original_epochs
    config.LEARNING_RATE = original_lr

    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for print_img, vein_img, labels in tqdm(test_loader, desc=f'Evaluating {fusion_name}'):
            print_img = print_img.to(device)
            vein_img = vein_img.to(device)
            outputs = model(print_img, vein_img)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels.to(device)).sum().item()

    accuracy = 100. * correct / total
    return accuracy


def plot_comparison(results, save_dir):
    plt = setup_matplotlib()
    names = list(results.keys())
    accs = list(results.values())
    colors = [STRATEGY_COLORS[n] for n in names]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(names, accs, color=colors, width=0.5, edgecolor='black', linewidth=0.8)

    for bar, acc in zip(bars, accs):
        ax.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
            f'{acc:.2f}%', ha='center', va='bottom', fontsize=11, fontweight='bold'
        )

    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Fusion Strategy Ablation Study')
    ax.set_ylim(min(accs) - 5, max(accs) + 3)
    ax.grid(axis='y', alpha=0.3)

    fig.savefig(os.path.join(save_dir, 'fusion_strategy_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()


def save_csv(results, save_dir):
    csv_path = os.path.join(save_dir, 'fusion_strategy_results.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Fusion Strategy', 'Accuracy (%)'])
        for name, acc in results.items():
            writer.writerow([name, f'{acc:.2f}'])
    return csv_path


def main():
    parser = argparse.ArgumentParser(description='Fusion Strategy Ablation Study for VIBE-Net')
    parser.add_argument('--dataset', type=str, default='HandsData', choices=DATASETS)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-4)
    args = parser.parse_args()

    seed_everything(config.SEED, deterministic=config.DETERMINISTIC)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f'Dataset: {args.dataset}')
    print(f'Epochs: {args.epochs}')
    print(f'Batch Size: {args.batch_size}')
    print(f'Learning Rate: {args.lr}')
    print(f'Device: {device}')
    print('=' * 60)

    results = {}
    for name, cls in FUSION_STRATEGIES.items():
        print(f'\n{"=" * 60}')
        print(f'Training with {name} Fusion')
        print(f'{"=" * 60}')
        acc = train_and_evaluate(
            cls, name, args.dataset, args.epochs, args.batch_size, args.lr, device
        )
        results[name] = acc
        print(f'{name} Fusion Accuracy: {acc:.2f}%')

    save_dir = get_result_dir(args.dataset, 'ablation')
    plot_comparison(results, save_dir)
    csv_path = save_csv(results, save_dir)

    print(f'\n{"=" * 60}')
    print('Ablation Study Results')
    print(f'{"=" * 60}')
    for name, acc in results.items():
        print(f'  {name}: {acc:.2f}%')
    print(f'\nResults saved to: {save_dir}')
    print(f'CSV saved to: {csv_path}')


if __name__ == '__main__':
    main()
