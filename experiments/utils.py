import os
import sys
import torch
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from models.vibe_net import VIBENet
from dataset import get_dataloader
import config

DATASETS = ['QH', 'TJ', 'CUMT2', 'HandsData']

RESULT_SUBDIRS = [
    'embeddings', 'feature_maps', 'gradcam', 'expert_weights',
    'cross_attention', 'training_curves', 'confusion_matrix',
    'roc_cmc', 'ablation'
]

ENHANCEMENT_EXPERT_NAMES = ['HighFreq', 'MidFreq', 'LowFreq']
FUSION_EXPERT_NAMES = ['CrossAttention', 'MultiScaleConv', 'ChannelInteraction']

PAPER_RC_PARAMS = {
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.unicode_minus': False,
}

EXPERT_COLORS = ['#2196F3', '#4CAF50', '#FF9800', '#E91E63', '#9C27B0']
DATASET_COLORS = {
    'QH': '#1f77b4',
    'TJ': '#ff7f0e',
    'CUMT2': '#2ca02c',
    'HandsData': '#d62728',
}


def setup_matplotlib():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update(PAPER_RC_PARAMS)
    return plt


def get_result_dir(dataset_name, subdir):
    base = os.path.join(os.path.dirname(__file__), '..', 'experiment_results', dataset_name, subdir)
    os.makedirs(base, exist_ok=True)
    return os.path.abspath(base)


def load_model(dataset_name, checkpoint_path=None, device=None, **model_kwargs):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dataset_cfg = config.get_dataset_config(dataset_name)
    num_classes = dataset_cfg['num_classes']

    defaults = dict(
        num_classes=num_classes,
        feature_dim=config.FEATURE_DIM,
        out_stages=config.OUT_STAGES,
        reducer_channels=config.REDUCER_CHANNELS,
        classifier_embed_dim=config.CLASSIFIER_EMBED_DIM,
        classifier_margin=config.ARC_MARGIN,
        classifier_scale=config.ARC_SCALE,
        classifier_dropout=config.CLASSIFIER_DROPOUT,
    )
    defaults.update(model_kwargs)

    model = VIBENet(**defaults)

    if checkpoint_path is None:
        checkpoint_path = os.path.join(config.get_save_dir(dataset_name), 'best_model.pth')

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    return model, checkpoint, device


def get_test_loader(dataset_name, batch_size=None, num_workers=0):
    if batch_size is None:
        batch_size = config.BATCH_SIZE
    return get_dataloader(dataset_name, mode='test', batch_size=batch_size,
                          num_workers=num_workers, shuffle=False)


def get_train_loader(dataset_name, batch_size=None, num_workers=0):
    if batch_size is None:
        batch_size = config.BATCH_SIZE
    return get_dataloader(dataset_name, mode='train', batch_size=batch_size,
                          num_workers=num_workers, shuffle=False)


def collect_predictions(model, loader, device):
    model.eval()
    all_preds = []
    all_labels = []
    all_embeddings = []
    correct = 0
    total = 0

    with torch.no_grad():
        for print_img, vein_img, labels in loader:
            print_img = print_img.to(device)
            vein_img = vein_img.to(device)

            logits, embeddings = model(print_img, vein_img, return_embedding=True)
            preds = logits.argmax(dim=1)

            total += labels.size(0)
            correct += preds.eq(labels.to(device)).sum().item()

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_embeddings.append(embeddings.cpu().numpy())

    accuracy = 100. * correct / total
    return accuracy, np.array(all_preds), np.array(all_labels), np.concatenate(all_embeddings, axis=0)


def save_results(data, save_dir, filename='results.npz'):
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, filename)
    np.savez(path, **data)
    return path


def save_figure(fig, save_dir, filename):
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, filename)
    fig.savefig(path, dpi=300, bbox_inches='tight')
    return path
