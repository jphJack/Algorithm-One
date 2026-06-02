import sys
import os
import argparse
from collections import defaultdict

import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from experiments.utils import (
    setup_matplotlib,
    get_result_dir,
    load_model,
    get_test_loader,
    save_results,
    DATASETS,
)


def collect_embeddings(model, loader, device, max_samples, max_classes, samples_per_class):
    embeddings = []
    labels = []
    preds = []
    class_counts = defaultdict(int)

    if max_classes is not None and max_classes > 0:
        allowed_classes = set(range(max_classes))
    else:
        allowed_classes = None

    stop = False
    with torch.no_grad():
        for print_img, vein_img, batch_labels in loader:
            print_img = print_img.to(device)
            vein_img = vein_img.to(device)

            logits, batch_embeddings = model(print_img, vein_img, return_embedding=True)
            batch_preds = logits.argmax(dim=1)

            for i in range(batch_labels.size(0)):
                label = int(batch_labels[i].item())
                if allowed_classes is not None and label not in allowed_classes:
                    continue
                if samples_per_class is not None and class_counts[label] >= samples_per_class:
                    continue

                embeddings.append(batch_embeddings[i].cpu().numpy())
                labels.append(label)
                preds.append(int(batch_preds[i].item()))
                class_counts[label] += 1

                if max_samples is not None and len(embeddings) >= max_samples:
                    stop = True
                    break

            if stop:
                break

    if len(embeddings) == 0:
        raise RuntimeError("No embeddings collected. Check max_classes/samples_per_class settings.")

    return np.array(embeddings), np.array(labels), np.array(preds)


def run_pca(embeddings, seed):
    pca = PCA(n_components=2, random_state=seed)
    coords = pca.fit_transform(embeddings)
    explained = pca.explained_variance_ratio_
    return coords, explained


def run_tsne(embeddings, seed, perplexity, n_iter):
    if embeddings.shape[1] > 50:
        pca = PCA(n_components=50, random_state=seed)
        embeddings = pca.fit_transform(embeddings)

    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        n_iter=n_iter,
        init='pca',
        random_state=seed,
        learning_rate='auto',
    )
    coords = tsne.fit_transform(embeddings)
    return coords


def plot_embeddings(coords, labels, preds, title, save_path, dataset_name):
    plt = setup_matplotlib()

    fig, ax = plt.subplots(figsize=(8, 6))

    unique_labels = sorted(np.unique(labels).tolist())
    n_classes = len(unique_labels)

    cmap = plt.cm.get_cmap('tab20' if n_classes <= 20 else 'nipy_spectral', n_classes)
    label_to_idx = {l: i for i, l in enumerate(unique_labels)}

    correct_mask = labels == preds
    incorrect_mask = ~correct_mask

    for label in unique_labels:
        idx = label_to_idx[label]
        color = cmap(idx / max(n_classes - 1, 1))
        mask_c = correct_mask & (labels == label)
        mask_i = incorrect_mask & (labels == label)

        if mask_c.any():
            ax.scatter(
                coords[mask_c, 0], coords[mask_c, 1],
                c=[color], marker='o', s=12, alpha=0.7,
                label=f'Class {label}' if n_classes <= 15 else None,
                edgecolors='none',
            )
        if mask_i.any():
            ax.scatter(
                coords[mask_i, 0], coords[mask_i, 1],
                c=[color], marker='x', s=30, alpha=0.9,
                edgecolors='none',
            )

    correct_handle = plt.Line2D(
        [], [], marker='o', color='gray', linestyle='None',
        markersize=6, label='Correct',
    )
    incorrect_handle = plt.Line2D(
        [], [], marker='x', color='gray', linestyle='None',
        markersize=6, label='Incorrect',
    )

    handles = [correct_handle, incorrect_handle]
    if n_classes <= 15:
        for label in unique_labels:
            idx = label_to_idx[label]
            color = cmap(idx / max(n_classes - 1, 1))
            handles.append(plt.Line2D(
                [], [], marker='o', color=color, linestyle='None',
                markersize=6, label=f'Class {label}',
            ))

    ax.legend(handles=handles, loc='best', fontsize=8, ncol=2)

    ax.set_title(title)
    ax.set_xlabel('Dimension 1')
    ax.set_ylabel('Dimension 2')

    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

    return save_path


def run_single_dataset(dataset_name, args):
    print(f"\n{'='*60}")
    print(f"Dataset: {dataset_name}")
    print(f"{'='*60}")

    model, checkpoint, device = load_model(dataset_name, checkpoint_path=args.checkpoint)
    loader = get_test_loader(dataset_name, batch_size=args.batch_size)

    embeddings, labels, preds = collect_embeddings(
        model, loader, device,
        max_samples=args.max_samples,
        max_classes=args.max_classes,
        samples_per_class=args.samples_per_class,
    )

    accuracy = 100.0 * np.sum(labels == preds) / len(labels)
    print(f"Samples: {len(embeddings)}, Classes: {len(np.unique(labels))}, Accuracy: {accuracy:.2f}%")

    save_dir = get_result_dir(dataset_name, 'embeddings')

    save_results({
        'embeddings': embeddings,
        'labels': labels,
        'preds': preds,
    }, save_dir, filename='embedding_data.npz')

    results = {}

    if args.method in ('pca', 'both'):
        print("Running PCA...")
        pca_coords, explained = run_pca(embeddings, args.seed)
        pca_path = os.path.join(save_dir, f'{dataset_name}_pca.png')
        plot_embeddings(
            pca_coords, labels, preds,
            f'PCA Embedding Visualization ({dataset_name})',
            pca_path, dataset_name,
        )
        np.save(os.path.join(save_dir, 'pca_coords.npy'), pca_coords)
        print(f"PCA explained variance: PC1={explained[0]:.4f}, PC2={explained[1]:.4f}")
        print(f"Saved PCA plot to: {pca_path}")
        results['pca'] = pca_path

    if args.method in ('tsne', 'both'):
        print("Running t-SNE...")
        tsne_coords = run_tsne(embeddings, args.seed, args.perplexity, args.tsne_iter)
        tsne_path = os.path.join(save_dir, f'{dataset_name}_tsne.png')
        plot_embeddings(
            tsne_coords, labels, preds,
            f't-SNE Embedding Visualization ({dataset_name})',
            tsne_path, dataset_name,
        )
        np.save(os.path.join(save_dir, 'tsne_coords.npy'), tsne_coords)
        print(f"Saved t-SNE plot to: {tsne_path}")
        results['tsne'] = tsne_path

    return results


def main():
    parser = argparse.ArgumentParser(description='PCA/t-SNE Embedding Visualization for VIBE-Net')
    parser.add_argument('--dataset', type=str, default='QH',
                        choices=DATASETS + ['all'],
                        help='Dataset name or "all" for all 4 datasets')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to checkpoint (default: best_model.pth)')
    parser.add_argument('--method', type=str, default='both',
                        choices=['pca', 'tsne', 'both'],
                        help='Dimensionality reduction method')
    parser.add_argument('--max-samples', type=int, default=2000,
                        help='Maximum number of samples to visualize')
    parser.add_argument('--max-classes', type=int, default=None,
                        help='Only keep classes with index < max_classes')
    parser.add_argument('--samples-per-class', type=int, default=None,
                        help='Maximum samples per class')
    parser.add_argument('--batch-size', type=int, default=64,
                        help='Batch size for inference')
    parser.add_argument('--perplexity', type=float, default=30.0,
                        help='t-SNE perplexity')
    parser.add_argument('--tsne-iter', type=int, default=1000,
                        help='t-SNE iterations')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')

    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    if args.dataset == 'all':
        datasets = DATASETS
    else:
        datasets = [args.dataset]

    all_results = {}
    for ds in datasets:
        try:
            results = run_single_dataset(ds, args)
            all_results[ds] = results
        except Exception as e:
            print(f"Error processing {ds}: {e}")
            all_results[ds] = {'error': str(e)}

    print(f"\n{'='*60}")
    print("Summary")
    print(f"{'='*60}")
    for ds, res in all_results.items():
        if 'error' in res:
            print(f"  {ds}: ERROR - {res['error']}")
        else:
            methods = [k for k in res.keys()]
            print(f"  {ds}: {methods}")

    print("\nDone.")


if __name__ == '__main__':
    main()
