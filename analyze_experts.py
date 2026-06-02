import os
import torch
import numpy as np
from tqdm import tqdm
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.vibe_net import VIBENet
from dataset import get_dataloader
import config

ENHANCEMENT_EXPERT_NAMES = ['HighFreq', 'MidFreq', 'LowFreq']
FUSION_EXPERT_NAMES = ['CrossAttention', 'MultiScaleConv', 'ChannelInteraction']


def analyze_expert_weights(dataset_name=None, save_dir=None):
    if dataset_name is None:
        dataset_name = config.DEFAULT_DATASET
    
    dataset_cfg = config.get_dataset_config(dataset_name)
    num_classes = dataset_cfg['num_classes']
    if save_dir is None:
        save_dir = config.get_save_dir(dataset_name)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'使用设备: {device}')
    print(f'数据集: {dataset_name}')
    print(f'类别数: {num_classes}')
    
    model = VIBENet(
        num_classes=num_classes,
        feature_dim=config.FEATURE_DIM,
        classifier_embed_dim=config.CLASSIFIER_EMBED_DIM,
        classifier_margin=config.ARC_MARGIN,
        classifier_scale=config.ARC_SCALE,
        classifier_dropout=config.CLASSIFIER_DROPOUT,
    )
    
    checkpoint_path = os.path.join(save_dir, 'best_model.pth')
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f'已加载模型: {checkpoint_path}')
        if 'best_acc' in checkpoint:
            print(f'训练时最佳准确率: {checkpoint["best_acc"]:.2f}%')
    else:
        print('警告: 未找到训练好的模型，使用随机初始化的模型')
        return
    
    model = model.to(device)
    model.eval()
    
    test_loader = get_dataloader(
        dataset_name,
        mode='test',
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
        shuffle=False
    )
    
    stage_keys = sorted(config.OUT_STAGES)

    all_print_gate_weights = {stage: [] for stage in stage_keys}
    all_vein_gate_weights = {stage: [] for stage in stage_keys}
    all_fusion_gate_weights = []
    all_labels = []
    all_preds = []
    
    correct = 0
    total = 0
    
    print(f'\n分析测试集中每个样本的专家权重...')
    with torch.no_grad():
        for print_img, vein_img, labels in tqdm(test_loader, desc='Analyzing'):
            print_img = print_img.to(device)
            vein_img = vein_img.to(device)
            
            outputs, gate_weights = model(print_img, vein_img, return_gate_weights=True)
            _, predicted = outputs.max(1)
            
            total += labels.size(0)
            correct += predicted.eq(labels.to(device)).sum().item()
            
            print_stage_weights = gate_weights['print_stage_enhancement']
            vein_stage_weights = gate_weights['vein_stage_enhancement']
            for stage in stage_keys:
                all_print_gate_weights[stage].append(print_stage_weights[stage].cpu().numpy())
                all_vein_gate_weights[stage].append(vein_stage_weights[stage].cpu().numpy())
            all_fusion_gate_weights.append(gate_weights['fusion'].cpu().numpy())
            all_labels.extend(labels.numpy())
            all_preds.extend(predicted.cpu().numpy())
    
    print_gate = {stage: np.concatenate(all_print_gate_weights[stage], axis=0) for stage in stage_keys}
    vein_gate = {stage: np.concatenate(all_vein_gate_weights[stage], axis=0) for stage in stage_keys}
    fusion_gate = np.concatenate(all_fusion_gate_weights, axis=0)
    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    
    accuracy = 100. * correct / total
    print(f'\n测试准确率: {accuracy:.2f}%')
    
    print('\n' + '=' * 70)
    print('MoE专家权重分析结果')
    print('=' * 70)
    
    for stage in stage_keys:
        print(f'\n--- 掌纹MoE特征增强模块 (Stage {stage}) ---')
        _print_gate_stats(print_gate[stage], ENHANCEMENT_EXPERT_NAMES, all_labels, all_preds)

    for stage in stage_keys:
        print(f'\n--- 掌静脉MoE特征增强模块 (Stage {stage}) ---')
        _print_gate_stats(vein_gate[stage], ENHANCEMENT_EXPERT_NAMES, all_labels, all_preds)
    
    print('\n--- MoE融合模块 (Fusion) ---')
    _print_gate_stats(fusion_gate, FUSION_EXPERT_NAMES, all_labels, all_preds)
    
    _plot_expert_weights(print_gate, vein_gate, fusion_gate, save_dir, stage_keys)

    _save_weight_data(print_gate, vein_gate, fusion_gate, all_labels, all_preds, save_dir, stage_keys)
    
    print(f'\n分析完成！结果已保存到: {save_dir}')


def _print_gate_stats(gate_weights, expert_names, labels, preds):
    mean_weights = gate_weights.mean(axis=0)
    std_weights = gate_weights.std(axis=0)
    
    print(f'  专家平均权重:')
    for i, name in enumerate(expert_names):
        print(f'    {name}: {mean_weights[i]:.4f} ± {std_weights[i]:.4f}')
    
    dominant_expert = np.argmax(gate_weights, axis=1)
    print(f'\n  主导专家分布 (权重最大的专家):')
    for i, name in enumerate(expert_names):
        count = (dominant_expert == i).sum()
        print(f'    {name}: {count}/{len(dominant_expert)} ({100.*count/len(dominant_expert):.1f}%)')
    
    correct_mask = labels == preds
    if correct_mask.sum() > 0 and (~correct_mask).sum() > 0:
        correct_weights = gate_weights[correct_mask].mean(axis=0)
        wrong_weights = gate_weights[~correct_mask].mean(axis=0)
        print(f'\n  正确预测样本的平均权重:')
        for i, name in enumerate(expert_names):
            print(f'    {name}: {correct_weights[i]:.4f}')
        print(f'  错误预测样本的平均权重:')
        for i, name in enumerate(expert_names):
            print(f'    {name}: {wrong_weights[i]:.4f}')


def _plot_expert_weights(print_gate, vein_gate, fusion_gate, save_dir, stage_keys):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import warnings
        warnings.filterwarnings('ignore', category=UserWarning)
        
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        
        _plot_stage_gate_grid(
            print_gate, stage_keys, ENHANCEMENT_EXPERT_NAMES,
            title_prefix='Print Enhancement',
            save_path=os.path.join(save_dir, 'expert_weights_print.png')
        )
        _plot_stage_gate_grid(
            vein_gate, stage_keys, ENHANCEMENT_EXPERT_NAMES,
            title_prefix='Vein Enhancement',
            save_path=os.path.join(save_dir, 'expert_weights_vein.png')
        )
        _plot_single_gate(
            fusion_gate, FUSION_EXPERT_NAMES,
            title='Fusion',
            save_path=os.path.join(save_dir, 'expert_weights_fusion.png')
        )
    except ImportError:
        print('matplotlib未安装，跳过绘图')


def _plot_stage_gate_grid(gate_by_stage, stage_keys, names, title_prefix, save_path):
    import matplotlib.pyplot as plt

    num_cols = len(stage_keys)
    fig, axes = plt.subplots(2, num_cols, figsize=(6 * num_cols, 8))
    if num_cols == 1:
        axes = axes.reshape(2, 1)

    colors = ['#2196F3', '#4CAF50', '#FF9800']
    for col, stage in enumerate(stage_keys):
        gate = gate_by_stage[stage]
        mean_w = gate.mean(axis=0)
        std_w = gate.std(axis=0)

        ax = axes[0, col]
        bars = ax.bar(names, mean_w, yerr=std_w, capsize=5, alpha=0.7, color=colors)
        ax.set_ylabel('Weight')
        ax.set_title(f'{title_prefix} - Stage {stage} Mean Weights')
        ax.set_ylim(0, max(mean_w + std_w) * 1.3)
        for bar, w in zip(bars, mean_w):
            ax.text(bar.get_x() + bar.get_width() / 2.0, bar.get_height() + 0.01,
                    f'{w:.3f}', ha='center', va='bottom', fontsize=9)

        ax2 = axes[1, col]
        ax2.boxplot([gate[:, i] for i in range(len(names))], labels=names, patch_artist=True)
        for patch, color in zip(ax2.patches, colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        ax2.set_ylabel('Weight')
        ax2.set_title(f'{title_prefix} - Stage {stage} Distribution')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f'\n专家权重分析图已保存到: {save_path}')


def _plot_single_gate(gate, names, title, save_path):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    mean_w = gate.mean(axis=0)
    std_w = gate.std(axis=0)
    colors = ['#2196F3', '#4CAF50', '#FF9800']

    bars = axes[0].bar(names, mean_w, yerr=std_w, capsize=5, alpha=0.7, color=colors)
    axes[0].set_ylabel('Weight')
    axes[0].set_title(f'{title} - Mean Expert Weights')
    axes[0].set_ylim(0, max(mean_w + std_w) * 1.3)
    for bar, w in zip(bars, mean_w):
        axes[0].text(bar.get_x() + bar.get_width() / 2.0, bar.get_height() + 0.01,
                     f'{w:.3f}', ha='center', va='bottom', fontsize=9)

    axes[1].boxplot([gate[:, i] for i in range(len(names))], labels=names, patch_artist=True)
    for patch, color in zip(axes[1].patches, colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    axes[1].set_ylabel('Weight')
    axes[1].set_title(f'{title} - Weight Distribution')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f'\n专家权重分析图已保存到: {save_path}')


def _save_weight_data(print_gate, vein_gate, fusion_gate, labels, preds, save_dir, stage_keys):
    save_path = os.path.join(save_dir, 'expert_weights.npz')
    data = {
        'fusion': fusion_gate,
        'labels': labels,
        'preds': preds,
        'enhancement_expert_names': ENHANCEMENT_EXPERT_NAMES,
        'fusion_expert_names': FUSION_EXPERT_NAMES,
        'stages': np.array(stage_keys),
    }
    for stage in stage_keys:
        data[f'print_enhancement_stage{stage}'] = print_gate[stage]
        data[f'vein_enhancement_stage{stage}'] = vein_gate[stage]

    np.savez(save_path, **data)
    print(f'专家权重数据已保存到: {save_path}')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='MoE专家权重分析')
    parser.add_argument('--dataset', type=str, default=config.DEFAULT_DATASET,
                        choices=list(config.DATASET_CONFIG.keys()),
                        help=f'数据集选择 (默认: {config.DEFAULT_DATASET})')
    parser.add_argument('--save-dir', type=str, default=None,
                        help='结果保存目录 (默认: checkpoints/<dataset_name>)')
    args = parser.parse_args()
    
    if args.save_dir:
        save_dir = args.save_dir
    else:
        save_dir = config.get_save_dir(args.dataset)
    
    analyze_expert_weights(args.dataset, save_dir=save_dir)
