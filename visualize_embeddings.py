import argparse
import os
from collections import defaultdict

import numpy as np
import torch

from dataset import get_dataloader
from models.vibe_net import VIBENet
import config


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_model(dataset_name, checkpoint_path, device, use_multiscale):
    dataset_cfg = config.get_dataset_config(dataset_name)
    num_classes = dataset_cfg['num_classes']

    model = VIBENet(
        num_classes=num_classes,
        feature_dim=config.FEATURE_DIM,
        out_stages=config.OUT_STAGES,
        reducer_channels=config.REDUCER_CHANNELS,
        classifier_embed_dim=config.CLASSIFIER_EMBED_DIM,
        classifier_margin=config.ARC_MARGIN,
        classifier_scale=config.ARC_SCALE,
        classifier_dropout=config.CLASSIFIER_DROPOUT,
        use_multiscale_extractor=use_multiscale,
    )

    if checkpoint_path is None:
        checkpoint_path = os.path.join(config.get_save_dir(dataset_name), 'best_model.pth')

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    return model, checkpoint_path


def collect_embeddings(
    model,
    loader,
    device,
    max_samples,
    max_classes,
    samples_per_class,
):
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
    try:
        from sklearn.decomposition import PCA
    except ImportError as exc:
        raise ImportError("scikit-learn is required for PCA. Install with: pip install scikit-learn") from exc

    pca = PCA(n_components=2, random_state=seed)
    coords = pca.fit_transform(embeddings)
    return coords


def run_tsne(embeddings, seed, perplexity, n_iter, pca_dim):
    try:
        from sklearn.decomposition import PCA
        from sklearn.manifold import TSNE
    except ImportError as exc:
        raise ImportError("scikit-learn is required for t-SNE. Install with: pip install scikit-learn") from exc

    if pca_dim is not None and embeddings.shape[1] > pca_dim:
        pca = PCA(n_components=pca_dim, random_state=seed)
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


def plot_scatter(coords, labels, title, save_path):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError("matplotlib is required for plotting. Install with: pip install matplotlib") from exc

    plt.figure(figsize=(8, 6))
    unique_labels = np.unique(labels)

    scatter = plt.scatter(
        coords[:, 0],
        coords[:, 1],
        c=labels,
        cmap='tab20',
        s=8,
        alpha=0.8,
    )

    plt.title(title)
    plt.xlabel('Dim 1')
    plt.ylabel('Dim 2')

    if len(unique_labels) <= 15:
        handles, _ = scatter.legend_elements()
        plt.legend(handles, unique_labels, title='Class', bbox_to_anchor=(1.02, 1), loc='upper left')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Embedding visualization with PCA/t-SNE')
    parser.add_argument('--dataset', type=str, default=config.DEFAULT_DATASET,
                        choices=list(config.DATASET_CONFIG.keys()))
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to checkpoint (default: best_model.pth in save dir)')
    parser.add_argument('--method', type=str, default='both', choices=['pca', 'tsne', 'both'])
    parser.add_argument('--max-samples', type=int, default=2000,
                        help='Max number of samples to visualize')
    parser.add_argument('--max-classes', type=int, default=None,
                        help='Only keep classes [0, max_classes-1]')
    parser.add_argument('--samples-per-class', type=int, default=None,
                        help='Limit samples per class')
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--num-workers', type=int, default=0)
    parser.add_argument('--perplexity', type=float, default=30.0)
    parser.add_argument('--tsne-iter', type=int, default=1000)
    parser.add_argument('--pca-dim', type=int, default=50,
                        help='PCA dimension before t-SNE')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--save-dir', type=str, default=None)
    parser.add_argument('--use-multiscale', action='store_true',
                        help='Use multiscale extractor before fusion')

    args = parser.parse_args()

    set_seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print(f"Dataset: {args.dataset}")

    model, checkpoint_path = load_model(args.dataset, args.checkpoint, device, args.use_multiscale)
    print(f"Loaded checkpoint: {checkpoint_path}")

    loader = get_dataloader(
        args.dataset,
        mode='test',
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
    )

    embeddings, labels, preds = collect_embeddings(
        model,
        loader,
        device,
        max_samples=args.max_samples,
        max_classes=args.max_classes,
        samples_per_class=args.samples_per_class,
    )

    if args.save_dir:
        save_dir = args.save_dir
    else:
        save_dir = os.path.join(config.get_save_dir(args.dataset), 'embedding_vis')
    os.makedirs(save_dir, exist_ok=True)

    np.savez(
        os.path.join(save_dir, 'embedding_data.npz'),
        embeddings=embeddings,
        labels=labels,
        preds=preds,
    )

    if args.method in ('pca', 'both'):
        pca_coords = run_pca(embeddings, args.seed)
        pca_path = os.path.join(save_dir, 'pca_scatter.png')
        plot_scatter(pca_coords, labels, 'PCA Embedding Visualization', pca_path)
        np.save(os.path.join(save_dir, 'pca_coords.npy'), pca_coords)
        print(f"Saved PCA plot to: {pca_path}")

    if args.method in ('tsne', 'both'):
        tsne_coords = run_tsne(embeddings, args.seed, args.perplexity, args.tsne_iter, args.pca_dim)
        tsne_path = os.path.join(save_dir, 'tsne_scatter.png')
        plot_scatter(tsne_coords, labels, 't-SNE Embedding Visualization', tsne_path)
        np.save(os.path.join(save_dir, 'tsne_coords.npy'), tsne_coords)
        print(f"Saved t-SNE plot to: {tsne_path}")


if __name__ == '__main__':
    main()
