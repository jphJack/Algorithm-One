import os
import sys
import argparse
import torch
import torch.nn.functional as F
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from utils import setup_matplotlib, get_result_dir, load_model, get_test_loader, DATASETS


def register_attention_hooks(expert):
    captured = {}

    def make_hook(name):
        def hook(module, input, output):
            captured[name] = output.detach()
        return hook

    hooks = [
        expert.query_conv_p.register_forward_hook(make_hook('q_p')),
        expert.key_conv_v.register_forward_hook(make_hook('k_v')),
        expert.query_conv_v.register_forward_hook(make_hook('q_v')),
        expert.key_conv_p.register_forward_hook(make_hook('k_p')),
    ]

    return captured, hooks


def compute_attention_maps(captured):
    q_p = captured['q_p']
    k_v = captured['k_v']
    q_v = captured['q_v']
    k_p = captured['k_p']

    B, C8, H, W = q_p.shape

    q_p_flat = q_p.view(B, C8, H * W).permute(0, 2, 1)
    k_v_flat = k_v.view(B, C8, H * W)
    attn_p2v = torch.bmm(q_p_flat, k_v_flat)
    attn_p2v = F.softmax(attn_p2v, dim=-1)

    q_v_flat = q_v.view(B, C8, H * W).permute(0, 2, 1)
    k_p_flat = k_p.view(B, C8, H * W)
    attn_v2p = torch.bmm(q_v_flat, k_p_flat)
    attn_v2p = F.softmax(attn_v2p, dim=-1)

    attn_p2v_spatial = attn_p2v.mean(dim=1).view(B, H, W)
    attn_v2p_spatial = attn_v2p.mean(dim=1).view(B, H, W)

    return attn_p2v_spatial.cpu().numpy(), attn_v2p_spatial.cpu().numpy()


def visualize_cross_attention(attn_p2v_list, attn_v2p_list, save_dir, num_samples):
    plt = setup_matplotlib()

    ncols = 2
    nrows = num_samples

    fig, axes = plt.subplots(nrows, ncols, figsize=(8, 3.5 * nrows))
    if nrows == 1:
        axes = axes[np.newaxis, :]

    for i in range(num_samples):
        p2v = attn_p2v_list[i]
        v2p = attn_v2p_list[i]

        p2v_norm = (p2v - p2v.min()) / (p2v.max() - p2v.min() + 1e-8)
        v2p_norm = (v2p - v2p.min()) / (v2p.max() - v2p.min() + 1e-8)

        ax_p2v = axes[i, 0]
        im1 = ax_p2v.imshow(p2v_norm, cmap='hot', interpolation='bilinear')
        ax_p2v.set_title(f'Sample {i + 1}: Print→Vein', fontsize=11)
        ax_p2v.axis('off')
        fig.colorbar(im1, ax=ax_p2v, fraction=0.046, pad=0.04)

        ax_v2p = axes[i, 1]
        im2 = ax_v2p.imshow(v2p_norm, cmap='hot', interpolation='bilinear')
        ax_v2p.set_title(f'Sample {i + 1}: Vein→Print', fontsize=11)
        ax_v2p.axis('off')
        fig.colorbar(im2, ax=ax_v2p, fraction=0.046, pad=0.04)

    fig.suptitle('Cross-Attention Spatial Maps', fontsize=14, y=1.0)
    plt.tight_layout()

    save_path = os.path.join(save_dir, 'cross_attention_heatmaps.png')
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

    return save_path


def main():
    parser = argparse.ArgumentParser(description='Cross-Attention Visualization for VIBE-Net')
    parser.add_argument('--dataset', type=str, default='HandsData', choices=DATASETS)
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--num-samples', type=int, default=8)
    parser.add_argument('--batch-size', type=int, default=1)
    args = parser.parse_args()

    model, checkpoint, device = load_model(args.dataset, args.checkpoint)
    test_loader = get_test_loader(args.dataset, batch_size=args.batch_size)

    expert = model.fusion.experts[0]
    captured, hooks = register_attention_hooks(expert)

    attn_p2v_list = []
    attn_v2p_list = []
    collected = 0

    with torch.no_grad():
        for print_img, vein_img, labels in test_loader:
            if collected >= args.num_samples:
                break

            print_img = print_img.to(device)
            vein_img = vein_img.to(device)

            _ = model(print_img, vein_img)

            attn_p2v, attn_v2p = compute_attention_maps(captured)

            batch_size = attn_p2v.shape[0]
            for j in range(batch_size):
                if collected >= args.num_samples:
                    break
                attn_p2v_list.append(attn_p2v[j])
                attn_v2p_list.append(attn_v2p[j])
                collected += 1

    for h in hooks:
        h.remove()

    save_dir = get_result_dir(args.dataset, 'cross_attention')
    save_path = visualize_cross_attention(attn_p2v_list, attn_v2p_list, save_dir, collected)

    print(f'Cross-attention visualizations saved to: {save_path}')


if __name__ == '__main__':
    main()
