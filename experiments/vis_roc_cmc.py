import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from utils import (
    setup_matplotlib,
    get_result_dir,
    load_model,
    get_test_loader,
    collect_predictions,
    DATASETS,
    DATASET_COLORS,
)


def cosine_similarity_matrix(embeddings):
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    normed = embeddings / norms
    sim = normed @ normed.T
    return sim


def compute_roc(sim_matrix, labels):
    n = len(labels)
    label_eq = labels[:, None] == labels[None, :]

    genuine_mask = label_eq & ~np.eye(n, dtype=bool)
    impostor_mask = ~label_eq & ~np.eye(n, dtype=bool)

    genuine_scores = sim_matrix[genuine_mask]
    impostor_scores = sim_matrix[impostor_mask]

    thresholds = np.linspace(sim_matrix.min(), sim_matrix.max(), 1000)

    far_list = []
    frr_list = []

    for thr in thresholds:
        far = np.mean(impostor_scores >= thr)
        frr = np.mean(genuine_scores < thr)
        far_list.append(far)
        frr_list.append(frr)

    far_arr = np.array(far_list)
    frr_arr = np.array(frr_list)
    gar_arr = 1.0 - frr_arr

    eer_idx = np.argmin(np.abs(far_arr - frr_arr))
    eer = (far_arr[eer_idx] + frr_arr[eer_idx]) / 2.0

    return far_arr, gar_arr, eer


def compute_cmc(sim_matrix, labels, max_rank):
    n = len(labels)
    label_eq = labels[:, None] == labels[None, :]

    np.fill_diagonal(sim_matrix, -np.inf)

    ranks = np.zeros(max_rank)

    for i in range(n):
        sorted_idx = np.argsort(-sim_matrix[i])
        sorted_labels = labels[sorted_idx]

        gallery_same = label_eq[i, sorted_idx]

        first_match = np.where(gallery_same)[0]
        if len(first_match) > 0:
            rank = first_match[0] + 1
            if rank <= max_rank:
                ranks[rank - 1:] += 1

    ranks = ranks / n
    return ranks


def plot_roc(far_arr, gar_arr, eer, dataset_name, color, ax):
    ax.plot(far_arr, gar_arr, color=color, linewidth=1.5,
            label=f'{dataset_name} (EER={eer*100:.2f}%)')
    ax.set_xlabel('False Accept Rate (FAR)')
    ax.set_ylabel('Genuine Accept Rate (GAR)')
    ax.set_title('ROC Curve')
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.05])
    ax.grid(True, alpha=0.3)
    ax.legend(loc='lower right')


def plot_cmc(cmc_ranks, dataset_name, color, max_rank, ax):
    x = np.arange(1, max_rank + 1)
    ax.plot(x, cmc_ranks * 100, color=color, linewidth=1.5, marker='o',
            markersize=3, label=f'{dataset_name}')
    ax.set_xlabel('Rank')
    ax.set_ylabel('Identification Rate (%)')
    ax.set_title('CMC Curve')
    ax.set_xlim([1, max_rank])
    ax.set_ylim([0, 105])
    ax.grid(True, alpha=0.3)
    ax.legend(loc='lower right')


def process_dataset(dataset_name, checkpoint, batch_size, max_rank, max_samples):
    model, checkpoint_data, device = load_model(dataset_name, checkpoint_path=checkpoint)
    loader = get_test_loader(dataset_name, batch_size=batch_size)

    accuracy, preds, labels, embeddings = collect_predictions(model, loader, device)

    if max_samples is not None and max_samples < len(embeddings):
        indices = np.random.choice(len(embeddings), max_samples, replace=False)
        embeddings = embeddings[indices]
        labels = labels[indices]

    sim_matrix = cosine_similarity_matrix(embeddings)

    far, gar, eer = compute_roc(sim_matrix, labels)
    cmc = compute_cmc(sim_matrix, labels, max_rank)

    print(f"[{dataset_name}] Accuracy: {accuracy:.2f}%, EER: {eer*100:.2f}%, "
          f"Rank-1: {cmc[0]*100:.2f}%, Rank-5: {cmc[min(4, max_rank-1)]*100:.2f}%")

    return far, gar, eer, cmc


def main():
    parser = argparse.ArgumentParser(description='ROC and CMC Curve Visualization for VIBE-Net')
    parser.add_argument('--dataset', type=str, nargs='+', default=['QH'],
                        choices=DATASETS,
                        help='Dataset name(s) for evaluation')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to checkpoint (default: best_model.pth)')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size for inference')
    parser.add_argument('--max-rank', type=int, default=20,
                        help='Maximum rank for CMC curve')
    parser.add_argument('--max-samples', type=int, default=None,
                        help='Maximum number of samples to use')

    args = parser.parse_args()

    plt = setup_matplotlib()

    results = {}
    for ds in args.dataset:
        print(f"\nProcessing dataset: {ds}")
        far, gar, eer, cmc = process_dataset(
            ds, args.checkpoint, args.batch_size, args.max_rank, args.max_samples
        )
        results[ds] = {'far': far, 'gar': gar, 'eer': eer, 'cmc': cmc}

    fig, (ax_roc, ax_cmc) = plt.subplots(1, 2, figsize=(14, 6))

    for ds in args.dataset:
        r = results[ds]
        color = DATASET_COLORS.get(ds, None)
        plot_roc(r['far'], r['gar'], r['eer'], ds, color, ax_roc)
        plot_cmc(r['cmc'], ds, color, args.max_rank, ax_cmc)

    plt.tight_layout()

    primary_ds = args.dataset[0]
    save_dir = get_result_dir(primary_ds, 'roc_cmc')
    save_path = os.path.join(save_dir, 'roc_cmc_curve.png')
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"\nSaved ROC/CMC plot to: {save_path}")

    for ds in args.dataset:
        r = results[ds]
        ds_dir = get_result_dir(ds, 'roc_cmc')
        np.savez(
            os.path.join(ds_dir, 'roc_cmc_data.npz'),
            far=r['far'],
            gar=r['gar'],
            eer=np.array([r['eer']]),
            cmc=r['cmc'],
        )
        print(f"Saved data for {ds} to: {ds_dir}")


if __name__ == '__main__':
    main()
