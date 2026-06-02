import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import argparse
import torch
import numpy as np

from utils import setup_matplotlib, get_result_dir, load_model, get_test_loader, DATASETS


def register_hooks(model, feature_maps):
    hooks = []

    def make_hook(name):
        def hook_fn(module, input, output):
            if isinstance(output, torch.Tensor):
                feature_maps[name] = output.detach().cpu()
            elif isinstance(output, tuple):
                feature_maps[name] = output[0].detach().cpu()
        return hook_fn

    for stream_name, backbone_attr in [('print', 'print_backbone'), ('vein', 'vein_backbone')]:
        backbone = getattr(model.backbone, backbone_attr)
        for stage in [3, 4, 5]:
            stage_module = getattr(backbone, f'stage{stage}')
            hooks.append(
                stage_module.register_forward_hook(
                    make_hook(f'{stream_name}_stage{stage}_before')
                )
            )

    for stream_name, enhancers_attr in [('print', 'print_stage_enhancers'), ('vein', 'vein_stage_enhancers')]:
        enhancers = getattr(model.backbone, enhancers_attr)
        for stage_key in ['3', '4', '5']:
            if stage_key in enhancers:
                hooks.append(
                    enhancers[stage_key].register_forward_hook(
                        make_hook(f'{stream_name}_stage{stage_key}_after')
                    )
                )

    hooks.append(
        model.fusion.register_forward_hook(make_hook('fusion_output'))
    )

    return hooks


def select_top_k_channels(feature_map, top_k):
    k = min(top_k, feature_map.shape[1])
    variances = feature_map.var(dim=(2, 3)).mean(dim=0)
    _, top_indices = variances.topk(k)
    return top_indices.sort().values


def visualize_feature_grid(feature_map, top_k, title, save_path, plt):
    top_indices = select_top_k_channels(feature_map, top_k)
    n_channels = len(top_indices)
    ncols = min(8, n_channels)
    nrows = (n_channels + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 2, nrows * 2))
    if nrows == 1 and ncols == 1:
        axes = np.array([[axes]])
    elif nrows == 1:
        axes = axes[np.newaxis, :]
    elif ncols == 1:
        axes = axes[:, np.newaxis]

    fig.suptitle(title, fontsize=14)

    for i in range(nrows * ncols):
        r, c = divmod(i, ncols)
        if i < n_channels:
            ch = top_indices[i].item()
            ax = axes[r, c]
            ax.imshow(feature_map[0, ch].numpy(), cmap='viridis')
            ax.set_title(f'Ch {ch}', fontsize=8)
            ax.axis('off')
        else:
            axes[r, c].axis('off')

    plt.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)


def visualize_before_after(before_map, after_map, top_k, title, save_path, plt):
    top_indices = select_top_k_channels(after_map, top_k)
    n_channels = len(top_indices)
    ncols = n_channels
    nrows = 3

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 2, nrows * 2.5))
    if ncols == 1:
        axes = axes[:, np.newaxis]

    fig.suptitle(title, fontsize=14)

    for i, idx in enumerate(top_indices):
        ch = idx.item()
        before_ch = before_map[0, ch].numpy()
        after_ch = after_map[0, ch].numpy()
        diff_ch = after_ch - before_ch

        axes[0, i].imshow(before_ch, cmap='viridis')
        axes[0, i].set_title(f'Ch {ch}', fontsize=8)
        axes[0, i].axis('off')

        axes[1, i].imshow(after_ch, cmap='viridis')
        axes[1, i].set_title(f'Ch {ch}', fontsize=8)
        axes[1, i].axis('off')

        vmax = max(abs(diff_ch.min()), abs(diff_ch.max())) or 1.0
        axes[2, i].imshow(diff_ch, cmap='RdBu_r', vmin=-vmax, vmax=vmax)
        axes[2, i].set_title(f'Ch {ch}', fontsize=8)
        axes[2, i].axis('off')

    axes[0, 0].set_ylabel('Before MoE', fontsize=10)
    axes[1, 0].set_ylabel('After MoE', fontsize=10)
    axes[2, 0].set_ylabel('Difference', fontsize=10)

    plt.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description='Feature Map Visualization for VIBE-Net')
    parser.add_argument('--dataset', type=str, default='HandsData', choices=DATASETS)
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--num-samples', type=int, default=4)
    parser.add_argument('--top-k-channels', type=int, default=8)
    parser.add_argument('--batch-size', type=int, default=1)
    args = parser.parse_args()

    plt = setup_matplotlib()
    save_dir = get_result_dir(args.dataset, 'feature_maps')

    model, checkpoint, device = load_model(args.dataset, args.checkpoint)
    loader = get_test_loader(args.dataset, batch_size=args.batch_size)

    data_iter = iter(loader)

    for sample_idx in range(args.num_samples):
        try:
            print_img, vein_img, labels = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            print_img, vein_img, labels = next(data_iter)

        print_img = print_img.to(device)
        vein_img = vein_img.to(device)

        feature_maps = {}
        hooks = register_hooks(model, feature_maps)

        with torch.no_grad():
            model(print_img, vein_img)

        for hook in hooks:
            hook.remove()

        for stream_name in ['print', 'vein']:
            for stage in [3, 4, 5]:
                before_key = f'{stream_name}_stage{stage}_before'
                after_key = f'{stream_name}_stage{stage}_after'

                if before_key in feature_maps and after_key in feature_maps:
                    stream_label = stream_name.capitalize()
                    title = f'Sample {sample_idx} - {stream_label} Stage {stage} MoE Enhancement'
                    save_path = os.path.join(
                        save_dir,
                        f'sample{sample_idx}_{stream_name}_stage{stage}_moe_comparison.png'
                    )
                    visualize_before_after(
                        feature_maps[before_key], feature_maps[after_key],
                        args.top_k_channels, title, save_path, plt
                    )

        if 'fusion_output' in feature_maps:
            title = f'Sample {sample_idx} - Fusion Output'
            save_path = os.path.join(save_dir, f'sample{sample_idx}_fusion_output.png')
            visualize_feature_grid(
                feature_maps['fusion_output'], args.top_k_channels,
                title, save_path, plt
            )

        print(f'Sample {sample_idx} visualizations saved to {save_dir}')

    print(f'All feature map visualizations saved to {save_dir}')


if __name__ == '__main__':
    main()
