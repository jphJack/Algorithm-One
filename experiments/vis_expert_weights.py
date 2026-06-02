import os
import sys
import argparse
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from utils import (
    setup_matplotlib, get_result_dir, load_model, get_test_loader,
    save_results, DATASETS, ENHANCEMENT_EXPERT_NAMES, FUSION_EXPERT_NAMES,
    EXPERT_COLORS,
)


def collect_gate_weights(model, loader, device):
    model.eval()
    all_gate_weights = {
        'print_stage_enhancement': {3: [], 4: [], 5: []},
        'vein_stage_enhancement': {3: [], 4: [], 5: []},
        'fusion': [],
    }
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for print_img, vein_img, labels in loader:
            print_img = print_img.to(device)
            vein_img = vein_img.to(device)

            output, gate_weights_dict = model(
                print_img, vein_img, return_gate_weights=True
            )
            preds = output.argmax(dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

            for stage_key in ['print_stage_enhancement', 'vein_stage_enhancement']:
                for stage_id, weights in gate_weights_dict[stage_key].items():
                    all_gate_weights[stage_key][stage_id].append(
                        weights.cpu().numpy()
                    )

            all_gate_weights['fusion'].append(
                gate_weights_dict['fusion'].cpu().numpy()
            )

    for stage_key in ['print_stage_enhancement', 'vein_stage_enhancement']:
        for stage_id in all_gate_weights[stage_key]:
            all_gate_weights[stage_key][stage_id] = np.concatenate(
                all_gate_weights[stage_key][stage_id], axis=0
            )
    all_gate_weights['fusion'] = np.concatenate(
        all_gate_weights['fusion'], axis=0
    )

    return all_gate_weights, np.array(all_preds), np.array(all_labels)


def plot_mean_expert_weights(gate_weights, save_dir, plt):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    stage_labels = ['Stage 3', 'Stage 4', 'Stage 5']

    for idx, (stage_key, title) in enumerate([
        ('print_stage_enhancement', 'Palmprint Enhancement'),
        ('vein_stage_enhancement', 'Palmvein Enhancement'),
    ]):
        ax = axes[idx]
        means = []
        stds = []
        for stage_id in [3, 4, 5]:
            w = gate_weights[stage_key][stage_id]
            means.append(w.mean(axis=0))
            stds.append(w.std(axis=0))

        means = np.array(means)
        stds = np.array(stds)
        x = np.arange(len(ENHANCEMENT_EXPERT_NAMES))
        width = 0.25

        for i, stage_label in enumerate(stage_labels):
            ax.bar(
                x + i * width, means[i], width, yerr=stds[i],
                label=stage_label, color=EXPERT_COLORS[i], capsize=3,
                alpha=0.85,
            )

        ax.set_xlabel('Expert')
        ax.set_ylabel('Weight')
        ax.set_title(title)
        ax.set_xticks(x + width)
        ax.set_xticklabels(ENHANCEMENT_EXPERT_NAMES)
        ax.legend()
        ax.set_ylim(0, 1)

    ax = axes[2]
    w = gate_weights['fusion']
    means = w.mean(axis=0)
    stds = w.std(axis=0)
    x = np.arange(len(FUSION_EXPERT_NAMES))
    ax.bar(x, means, yerr=stds, color=EXPERT_COLORS[:len(FUSION_EXPERT_NAMES)],
           capsize=3, alpha=0.85)
    ax.set_xlabel('Expert')
    ax.set_ylabel('Weight')
    ax.set_title('Fusion')
    ax.set_xticks(x)
    ax.set_xticklabels(FUSION_EXPERT_NAMES)
    ax.set_ylim(0, 1)

    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, 'mean_expert_weights.png'), dpi=300, bbox_inches='tight')
    plt.close(fig)


def plot_weight_distributions(gate_weights, save_dir, plt):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for idx, (stage_key, title) in enumerate([
        ('print_stage_enhancement', 'Palmprint Enhancement'),
        ('vein_stage_enhancement', 'Palmvein Enhancement'),
    ]):
        ax = axes[idx]
        data = []
        labels = []
        for stage_id in [3, 4, 5]:
            w = gate_weights[stage_key][stage_id]
            for j, name in enumerate(ENHANCEMENT_EXPERT_NAMES):
                data.append(w[:, j])
                labels.append(f'S{stage_id}-{name}')

        bp = ax.boxplot(data, patch_artist=True, labels=labels)
        for i, patch in enumerate(bp['boxes']):
            patch.set_facecolor(EXPERT_COLORS[i % len(ENHANCEMENT_EXPERT_NAMES)])
            patch.set_alpha(0.7)

        ax.set_xlabel('Stage-Expert')
        ax.set_ylabel('Weight')
        ax.set_title(title)
        ax.tick_params(axis='x', rotation=45)

    ax = axes[2]
    w = gate_weights['fusion']
    data = [w[:, j] for j in range(w.shape[1])]
    bp = ax.boxplot(data, patch_artist=True, labels=FUSION_EXPERT_NAMES)
    for i, patch in enumerate(bp['boxes']):
        patch.set_facecolor(EXPERT_COLORS[i])
        patch.set_alpha(0.7)

    ax.set_xlabel('Expert')
    ax.set_ylabel('Weight')
    ax.set_title('Fusion')

    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, 'weight_distributions.png'), dpi=300, bbox_inches='tight')
    plt.close(fig)


def plot_correct_vs_incorrect(gate_weights, preds, labels, save_dir, plt):
    correct_mask = preds == labels
    incorrect_mask = ~correct_mask

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for idx, (stage_key, title) in enumerate([
        ('print_stage_enhancement', 'Palmprint Enhancement'),
        ('vein_stage_enhancement', 'Palmvein Enhancement'),
    ]):
        ax = axes[idx]
        correct_means = []
        incorrect_means = []
        x_labels = []

        for stage_id in [3, 4, 5]:
            w = gate_weights[stage_key][stage_id]
            for j, name in enumerate(ENHANCEMENT_EXPERT_NAMES):
                correct_means.append(w[correct_mask, j].mean() if correct_mask.any() else 0)
                incorrect_means.append(w[incorrect_mask, j].mean() if incorrect_mask.any() else 0)
                x_labels.append(f'S{stage_id}-{name}')

        x = np.arange(len(x_labels))
        width = 0.35
        ax.bar(x - width / 2, correct_means, width, label='Correct',
               color='#4CAF50', alpha=0.85)
        ax.bar(x + width / 2, incorrect_means, width, label='Incorrect',
               color='#F44336', alpha=0.85)

        ax.set_xlabel('Stage-Expert')
        ax.set_ylabel('Mean Weight')
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(x_labels, rotation=45)
        ax.legend()

    ax = axes[2]
    w = gate_weights['fusion']
    correct_means = [w[correct_mask, j].mean() if correct_mask.any() else 0
                     for j in range(w.shape[1])]
    incorrect_means = [w[incorrect_mask, j].mean() if incorrect_mask.any() else 0
                       for j in range(w.shape[1])]

    x = np.arange(len(FUSION_EXPERT_NAMES))
    width = 0.35
    ax.bar(x - width / 2, correct_means, width, label='Correct',
           color='#4CAF50', alpha=0.85)
    ax.bar(x + width / 2, incorrect_means, width, label='Incorrect',
           color='#F44336', alpha=0.85)

    ax.set_xlabel('Expert')
    ax.set_ylabel('Mean Weight')
    ax.set_title('Fusion')
    ax.set_xticks(x)
    ax.set_xticklabels(FUSION_EXPERT_NAMES)
    ax.legend()

    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, 'correct_vs_incorrect.png'), dpi=300, bbox_inches='tight')
    plt.close(fig)


def plot_weight_heatmap(gate_weights, labels, save_dir, plt, max_classes=None):
    unique_labels = np.unique(labels)
    if max_classes is not None and len(unique_labels) > max_classes:
        unique_labels = unique_labels[:max_classes]

    fig, axes = plt.subplots(1, 3, figsize=(20, 8))

    for idx, (stage_key, title) in enumerate([
        ('print_stage_enhancement', 'Palmprint Enhancement'),
        ('vein_stage_enhancement', 'Palmvein Enhancement'),
    ]):
        ax = axes[idx]
        all_expert_names = []
        heatmap_data = []

        for stage_id in [3, 4, 5]:
            w = gate_weights[stage_key][stage_id]
            for j, name in enumerate(ENHANCEMENT_EXPERT_NAMES):
                all_expert_names.append(f'S{stage_id}-{name}')
                row = []
                for cls in unique_labels:
                    mask = labels == cls
                    row.append(w[mask, j].mean())
                heatmap_data.append(row)

        heatmap_data = np.array(heatmap_data)
        im = ax.imshow(heatmap_data, aspect='auto', cmap='YlOrRd')
        ax.set_yticks(np.arange(len(all_expert_names)))
        ax.set_yticklabels(all_expert_names)
        ax.set_xticks(np.arange(len(unique_labels)))
        ax.set_xticklabels([str(c) for c in unique_labels])
        ax.set_xlabel('Class')
        ax.set_ylabel('Expert')
        ax.set_title(title)
        fig.colorbar(im, ax=ax)

    ax = axes[2]
    w = gate_weights['fusion']
    heatmap_data = []
    for j, name in enumerate(FUSION_EXPERT_NAMES):
        row = []
        for cls in unique_labels:
            mask = labels == cls
            row.append(w[mask, j].mean())
        heatmap_data.append(row)

    heatmap_data = np.array(heatmap_data)
    im = ax.imshow(heatmap_data, aspect='auto', cmap='YlOrRd')
    ax.set_yticks(np.arange(len(FUSION_EXPERT_NAMES)))
    ax.set_yticklabels(FUSION_EXPERT_NAMES)
    ax.set_xticks(np.arange(len(unique_labels)))
    ax.set_xticklabels([str(c) for c in unique_labels])
    ax.set_xlabel('Class')
    ax.set_ylabel('Expert')
    ax.set_title('Fusion')
    fig.colorbar(im, ax=ax)

    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, 'weight_heatmap.png'), dpi=300, bbox_inches='tight')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description='Expert Weight Visualization')
    parser.add_argument('--dataset', type=str, default='QH', choices=DATASETS)
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--max-classes', type=int, default=None)
    args = parser.parse_args()

    plt = setup_matplotlib()
    save_dir = get_result_dir(args.dataset, 'expert_weights')

    model, checkpoint, device = load_model(
        args.dataset, checkpoint_path=args.checkpoint
    )
    loader = get_test_loader(args.dataset, batch_size=args.batch_size)

    gate_weights, preds, labels = collect_gate_weights(model, loader, device)

    plot_mean_expert_weights(gate_weights, save_dir, plt)
    plot_weight_distributions(gate_weights, save_dir, plt)
    plot_correct_vs_incorrect(gate_weights, preds, labels, save_dir, plt)
    plot_weight_heatmap(gate_weights, labels, save_dir, plt, max_classes=args.max_classes)

    np_data = {}
    for stage_key in ['print_stage_enhancement', 'vein_stage_enhancement']:
        for stage_id in [3, 4, 5]:
            np_data[f'{stage_key}_{stage_id}'] = gate_weights[stage_key][stage_id]
    np_data['fusion'] = gate_weights['fusion']
    np_data['preds'] = preds
    np_data['labels'] = labels
    save_results(np_data, save_dir, filename='expert_weights.npz')

    print(f'Results saved to {save_dir}')


if __name__ == '__main__':
    main()
