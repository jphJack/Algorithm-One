import os
import sys
import argparse
import torch
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from experiments.utils import setup_matplotlib, get_result_dir, load_model, DATASETS, DATASET_COLORS
import config


def load_training_history(dataset_name, checkpoint_dir=None):
    if checkpoint_dir is None:
        checkpoint_dir = config.get_save_dir(dataset_name)
    checkpoint_path = os.path.join(checkpoint_dir, 'best_model.pth')
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    return {
        'train_losses': checkpoint['train_losses'],
        'train_accs': checkpoint['train_accs'],
        'val_accs': checkpoint['val_accs'],
        'train_ce_losses': checkpoint['train_ce_losses'],
        'train_lb_losses': checkpoint['train_lb_losses'],
    }


def plot_total_loss(history, save_dir, dataset_name):
    plt = setup_matplotlib()
    fig, ax = plt.subplots(figsize=(8, 5))
    epochs = np.arange(1, len(history['train_losses']) + 1)
    ax.plot(epochs, history['train_losses'], color='#1f77b4', linewidth=1.5)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Total Loss')
    ax.set_title(f'Total Loss - {dataset_name}')
    ax.grid(True, alpha=0.3)
    fig.savefig(os.path.join(save_dir, 'total_loss.png'), dpi=300, bbox_inches='tight')
    plt.close(fig)


def plot_ce_lb_loss(history, save_dir, dataset_name):
    plt = setup_matplotlib()
    epochs = np.arange(1, len(history['train_ce_losses']) + 1)
    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.plot(epochs, history['train_ce_losses'], color='#1f77b4', linewidth=1.5, label='CE Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('CE Loss', color='#1f77b4')
    ax1.tick_params(axis='y', labelcolor='#1f77b4')
    ax2 = ax1.twinx()
    ax2.plot(epochs, history['train_lb_losses'], color='#ff7f0e', linewidth=1.5, label='LB Loss')
    ax2.set_ylabel('LB Loss', color='#ff7f0e')
    ax2.tick_params(axis='y', labelcolor='#ff7f0e')
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
    ax1.set_title(f'CE Loss & LB Loss - {dataset_name}')
    ax1.grid(True, alpha=0.3)
    fig.savefig(os.path.join(save_dir, 'ce_lb_loss.png'), dpi=300, bbox_inches='tight')
    plt.close(fig)


def plot_accuracy(history, save_dir, dataset_name):
    plt = setup_matplotlib()
    fig, ax = plt.subplots(figsize=(8, 5))
    epochs = np.arange(1, len(history['train_accs']) + 1)
    ax.plot(epochs, history['train_accs'], color='#1f77b4', linewidth=1.5, label='Train Accuracy')
    ax.plot(epochs, history['val_accs'], color='#d62728', linewidth=1.5, label='Val Accuracy')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy (%)')
    ax.set_title(f'Train & Val Accuracy - {dataset_name}')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    fig.savefig(os.path.join(save_dir, 'accuracy.png'), dpi=300, bbox_inches='tight')
    plt.close(fig)


def plot_multi_dataset_comparison(datasets, checkpoint_dir=None):
    plt = setup_matplotlib()
    all_histories = {}
    for ds in datasets:
        try:
            all_histories[ds] = load_training_history(ds, checkpoint_dir)
        except FileNotFoundError as e:
            print(f"Skipping {ds}: {e}")

    if not all_histories:
        print("No checkpoint data found for any dataset.")
        return

    fig, axes = plt.subplots(1, 3, figsize=(20, 5))

    ax_loss = axes[0]
    for ds, hist in all_histories.items():
        epochs = np.arange(1, len(hist['train_losses']) + 1)
        ax_loss.plot(epochs, hist['train_losses'], color=DATASET_COLORS[ds], linewidth=1.5, label=ds)
    ax_loss.set_xlabel('Epoch')
    ax_loss.set_ylabel('Total Loss')
    ax_loss.set_title('Total Loss Comparison')
    ax_loss.legend(loc='upper right')
    ax_loss.grid(True, alpha=0.3)

    ax_ce = axes[1]
    for ds, hist in all_histories.items():
        epochs = np.arange(1, len(hist['train_ce_losses']) + 1)
        ax_ce.plot(epochs, hist['train_ce_losses'], color=DATASET_COLORS[ds], linewidth=1.5, label=ds)
    ax_ce.set_xlabel('Epoch')
    ax_ce.set_ylabel('CE Loss')
    ax_ce.set_title('CE Loss Comparison')
    ax_ce.legend(loc='upper right')
    ax_ce.grid(True, alpha=0.3)

    ax_acc = axes[2]
    for ds, hist in all_histories.items():
        epochs = np.arange(1, len(hist['val_accs']) + 1)
        ax_acc.plot(epochs, hist['val_accs'], color=DATASET_COLORS[ds], linewidth=1.5, label=ds)
    ax_acc.set_xlabel('Epoch')
    ax_acc.set_ylabel('Val Accuracy (%)')
    ax_acc.set_title('Validation Accuracy Comparison')
    ax_acc.legend(loc='lower right')
    ax_acc.grid(True, alpha=0.3)

    plt.tight_layout()
    comparison_dir = get_result_dir(datasets[0], 'training_curves')
    save_path = os.path.join(os.path.dirname(comparison_dir), 'multi_dataset_comparison.png')
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved multi-dataset comparison to: {save_path}")


def main():
    parser = argparse.ArgumentParser(description='VIBE-Net Training Curves Visualization')
    parser.add_argument('--datasets', nargs='+', default=DATASETS, choices=DATASETS,
                        help='Datasets to visualize')
    parser.add_argument('--checkpoint-dir', type=str, default=None,
                        help='Checkpoint directory override')
    args = parser.parse_args()

    for ds in args.datasets:
        print(f"Processing dataset: {ds}")
        try:
            history = load_training_history(ds, args.checkpoint_dir)
        except FileNotFoundError as e:
            print(f"Skipping {ds}: {e}")
            continue

        save_dir = get_result_dir(ds, 'training_curves')

        plot_total_loss(history, save_dir, ds)
        print(f"  Saved total_loss.png")

        plot_ce_lb_loss(history, save_dir, ds)
        print(f"  Saved ce_lb_loss.png")

        plot_accuracy(history, save_dir, ds)
        print(f"  Saved accuracy.png")

    if len(args.datasets) > 1:
        plot_multi_dataset_comparison(args.datasets, args.checkpoint_dir)

    print("Done.")


if __name__ == '__main__':
    main()
