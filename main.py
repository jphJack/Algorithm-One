import os
import argparse
import torch
import sys
import random
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.vibe_net import VIBENet
from dataset import get_dataloader
import config


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def demo_inference():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'使用设备: {device}')
    
    model = VIBENet(num_classes=config.NUM_CLASSES, feature_dim=config.FEATURE_DIM)
    model = model.to(device)
    model.eval()
    
    print(f'模型参数量: {count_parameters(model):,}')
    
    print_img = torch.randn(1, 1, 217, 190).to(device)
    vein_img = torch.randn(1, 1, 180, 180).to(device)
    
    with torch.no_grad():
        output = model(print_img, vein_img)
        pred = output.argmax(dim=1)
    
    print(f'输入掌纹图像: {print_img.shape}')
    print(f'输入掌静脉图像: {vein_img.shape}')
    print(f'输出logits: {output.shape}')
    print(f'预测类别: {pred.item()}')


def main():
    parser = argparse.ArgumentParser(description='VIBE双模态生物特征识别网络')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test', 'demo'],
                        help='运行模式: train(训练), test(测试), demo(演示)')
    parser.add_argument('--epochs', type=int, default=config.NUM_EPOCHS,
                        help=f'训练轮数 (默认: {config.NUM_EPOCHS})')
    parser.add_argument('--batch-size', type=int, default=config.BATCH_SIZE,
                        help=f'批次大小 (默认: {config.BATCH_SIZE})')
    parser.add_argument('--lr', type=float, default=config.LEARNING_RATE,
                        help=f'学习率 (默认: {config.LEARNING_RATE})')
    parser.add_argument('--feature-dim', type=int, default=config.FEATURE_DIM,
                        help=f'特征维度 (默认: {config.FEATURE_DIM})')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='加载的检查点路径')
    parser.add_argument('--num-workers', type=int, default=config.NUM_WORKERS,
                        help=f'数据加载线程数 (默认: {config.NUM_WORKERS})')
    parser.add_argument('--seed', type=int, default=config.SEED,
                        help=f'随机种子 (默认: {config.SEED})')
    
    args = parser.parse_args()
    
    config.NUM_EPOCHS = args.epochs
    config.BATCH_SIZE = args.batch_size
    config.LEARNING_RATE = args.lr
    config.FEATURE_DIM = args.feature_dim
    config.NUM_WORKERS = args.num_workers
    config.SEED = args.seed

    set_seed(config.SEED)
    
    print('=' * 60)
    print('VIBE双模态生物特征识别网络')
    print('=' * 60)
    print(f'模式: {args.mode}')
    print(f'类别数: {config.NUM_CLASSES}')
    print(f'批次大小: {config.BATCH_SIZE}')
    print(f'特征维度: {config.FEATURE_DIM}')
    print(f'随机种子: {config.SEED}')
    print('=' * 60)
    
    if args.mode == 'train':
        from train import main as train_main
        train_main()
    
    elif args.mode == 'test':
        from test import main as test_main
        test_main()
    
    elif args.mode == 'demo':
        demo_inference()


if __name__ == '__main__':
    main()
