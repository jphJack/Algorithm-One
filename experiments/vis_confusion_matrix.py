import sys
import os
import argparse

import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sns

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from utils import setup_matplotlib, get_result_dir, load_model, get_test_loader, collect_predictions, DATASETS


def plot_full_confusion_matrix(cm_normalized, save_dir, dataset_name):
    plt = setup_matplotlib()
    fig, ax = plt.subplots(figsize=(14, 12))
    sns.heatmap(
        cm_normalized,
        annot=False,
        fmt='.2f',
        cmap='Blues',
        ax=ax,
        cbar=True,
        square=True,
        linewidths=0,
    )
    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')
    ax.set_title(f'Normalized Confusion Matrix - {dataset_name}')
    fig.tight_layout()
    save_path = os.path.join(save_dir, 'confusion_matrix_full.png')
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved full confusion matrix to: {save_path}")


def plot_subset_confusion_matrix(cm_normalized, per_class_acc, save_dir, dataset_name, num_classes_show, top_k, bottom_k):
    plt = setup_matplotlib()
    sorted_indices = np.argsort(per_class_acc)

    top_k_indices = sorted_indices[-top_k:][::-1]
    bottom_k_indices = sorted_indices[:bottom_k]
    selected_indices = np.concatenate([top_k_indices, bottom_k_indices])
    selected_indices = np.sort(np.unique(selected_indices))

    if len(selected_indices) > num_classes_show:
        selected_indices = selected_indices[:num_classes_show]

    subset_cm = cm_normalized[np.ix_(selected_indices, selected_indices)]
    tick_labels = [str(i) for i in selected_indices]

    fig, ax = plt.subplots(figsize=(max(10, len(selected_indices) * 0.6), max(8, len(selected_indices) * 0.5)))
    sns.heatmap(
        subset_cm,
        annot=len(selected_indices) <= 30,
        fmt='.2f' if len(selected_indices) <= 30 else '',
        cmap='Blues',
        ax=ax,
        cbar=True,
        square=True,
        linewidths=0.5 if len(selected_indices) <= 30 else 0,
        xticklabels=tick_labels,
        yticklabels=tick_labels,
    )
    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')
    ax.set_title(f'Subset Confusion Matrix (Top-{top_k} & Bottom-{bottom_k}) - {dataset_name}')
    fig.tight_layout()
    save_path = os.path.join(save_dir, 'confusion_matrix_subset.png')
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved subset confusion matrix to: {save_path}")


def plot_per_class_accuracy(per_class_acc, save_dir, dataset_name):
    plt = setup_matplotlib()
    fig, ax = plt.subplots(figsize=(14, 5))
    num_classes = len(per_class_acc)
    x = np.arange(num_classes)
    sorted_acc = np.sort(per_class_acc)
    colors = ['#e74c3c' if a < 0.5 else '#f39c12' if a < 0.8 else '#27ae60' for a in sorted_acc]

    ax.bar(x, sorted_acc, color=colors, width=1.0, edgecolor='none')
    ax.set_xlabel('Class Index (sorted by accuracy)')
    ax.set_ylabel('Accuracy')
    ax.set_title(f'Per-Class Accuracy Distribution - {dataset_name}')
    ax.set_ylim(0, 1.05)
    ax.axhline(y=0.5, color='gray', linestyle='--', linewidth=0.8, alpha=0.6)
    ax.axhline(y=0.8, color='gray', linestyle='--', linewidth=0.8, alpha=0.6)
    mean_acc = np.mean(per_class_acc)
    ax.axhline(y=mean_acc, color='#2c3e50', linestyle='-', linewidth=1.2, alpha=0.8)
    ax.text(num_classes * 0.02, mean_acc + 0.02, f'Mean: {mean_acc:.3f}', color='#2c3e50', fontsize=10)
    fig.tight_layout()
    save_path = os.path.join(save_dir, 'per_class_accuracy.png')
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved per-class accuracy histogram to: {save_path}")


def main():
    parser = argparse.ArgumentParser(description='Confusion Matrix Visualization for VIBE-Net')
    parser.add_argument('--dataset', type=str, default='CUMT2', choices=DATASETS)
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--num-classes-show', type=int, default=40)
    parser.add_argument('--top-k', type=int, default=10)
    parser.add_argument('--bottom-k', type=int, default=10)
    args = parser.parse_args()

    model, checkpoint, device = load_model(args.dataset, checkpoint_path=args.checkpoint)
    loader = get_test_loader(args.dataset, batch_size=args.batch_size)

    accuracy, all_preds, all_labels, _ = collect_predictions(model, loader, device)
    print(f"Overall accuracy: {accuracy:.2f}%")

    cm = confusion_matrix(all_labels, all_preds)
    cm_normalized = cm.astype(np.float64)
    row_sums = cm_normalized.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    cm_normalized = cm_normalized / row_sums

    per_class_acc = np.diag(cm_normalized)

    save_dir = get_result_dir(args.dataset, 'confusion_matrix')

    plot_full_confusion_matrix(cm_normalized, save_dir, args.dataset)
    plot_subset_confusion_matrix(cm_normalized, per_class_acc, save_dir, args.dataset, args.num_classes_show, args.top_k, args.bottom_k)
    plot_per_class_accuracy(per_class_acc, save_dir, args.dataset)

    np.savez(
        os.path.join(save_dir, 'confusion_matrix_data.npz'),
        confusion_matrix=cm,
        confusion_matrix_normalized=cm_normalized,
        per_class_accuracy=per_class_acc,
        overall_accuracy=accuracy,
    )
    print(f"Saved confusion matrix data to: {os.path.join(save_dir, 'confusion_matrix_data.npz')}")


if __name__ == '__main__':
    main()
