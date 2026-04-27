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
    
    model = VIBENet(num_classes=num_classes, feature_dim=config.FEATURE_DIM)
    
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
    
    all_print_gate_weights = []
    all_vein_gate_weights = []
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
            
            all_print_gate_weights.append(gate_weights['print_enhancement'].cpu().numpy())
            all_vein_gate_weights.append(gate_weights['vein_enhancement'].cpu().numpy())
            all_fusion_gate_weights.append(gate_weights['fusion'].cpu().numpy())
            all_labels.extend(labels.numpy())
            all_preds.extend(predicted.cpu().numpy())
    
    print_gate = np.concatenate(all_print_gate_weights, axis=0)
    vein_gate = np.concatenate(all_vein_gate_weights, axis=0)
    fusion_gate = np.concatenate(all_fusion_gate_weights, axis=0)
    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    
    accuracy = 100. * correct / total
    print(f'\n测试准确率: {accuracy:.2f}%')
    
    print('\n' + '=' * 70)
    print('MoE专家权重分析结果')
    print('=' * 70)
    
    print('\n--- 掌纹MoE特征增强模块 (Print Enhancement) ---')
    _print_gate_stats(print_gate, ENHANCEMENT_EXPERT_NAMES, all_labels, all_preds)
    
    print('\n--- 掌静脉MoE特征增强模块 (Vein Enhancement) ---')
    _print_gate_stats(vein_gate, ENHANCEMENT_EXPERT_NAMES, all_labels, all_preds)
    
    print('\n--- MoE融合模块 (Fusion) ---')
    _print_gate_stats(fusion_gate, FUSION_EXPERT_NAMES, all_labels, all_preds)
    
    _plot_expert_weights(print_gate, vein_gate, fusion_gate, save_dir)
    
    _save_weight_data(print_gate, vein_gate, fusion_gate, all_labels, all_preds, save_dir)
    
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


def _plot_expert_weights(print_gate, vein_gate, fusion_gate, save_dir):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import warnings
        warnings.filterwarnings('ignore', category=UserWarning)
        
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        
        gate_data = [
            ('Print Enhancement', print_gate, ENHANCEMENT_EXPERT_NAMES),
            ('Vein Enhancement', vein_gate, ENHANCEMENT_EXPERT_NAMES),
            ('Fusion', fusion_gate, FUSION_EXPERT_NAMES),
        ]
        
        for col, (title, gate, names) in enumerate(gate_data):
            ax = axes[0, col]
            mean_w = gate.mean(axis=0)
            std_w = gate.std(axis=0)
            bars = ax.bar(names, mean_w, yerr=std_w, capsize=5, alpha=0.7,
                          color=['#2196F3', '#4CAF50', '#FF9800'])
            ax.set_ylabel('Weight')
            ax.set_title(f'{title} - Mean Expert Weights')
            ax.set_ylim(0, max(mean_w + std_w) * 1.3)
            for bar, w in zip(bars, mean_w):
                ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                        f'{w:.3f}', ha='center', va='bottom', fontsize=9)
            
            ax2 = axes[1, col]
            ax2.boxplot([gate[:, i] for i in range(len(names))],
                        labels=names, patch_artist=True)
            colors = ['#2196F3', '#4CAF50', '#FF9800']
            for patch, color in zip(ax2.patches, colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
            ax2.set_ylabel('Weight')
            ax2.set_title(f'{title} - Weight Distribution')
        
        plt.tight_layout()
        save_path = os.path.join(save_dir, 'expert_weights_analysis.png')
        plt.savefig(save_path, dpi=150)
        plt.close()
        print(f'\n专家权重分析图已保存到: {save_path}')
    except ImportError:
        print('matplotlib未安装，跳过绘图')


def _save_weight_data(print_gate, vein_gate, fusion_gate, labels, preds, save_dir):
    save_path = os.path.join(save_dir, 'expert_weights.npz')
    np.savez(save_path,
             print_enhancement=print_gate,
             vein_enhancement=vein_gate,
             fusion=fusion_gate,
             labels=labels,
             preds=preds,
             enhancement_expert_names=ENHANCEMENT_EXPERT_NAMES,
             fusion_expert_names=FUSION_EXPERT_NAMES)
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
