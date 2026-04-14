import os
import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.vibe_net import VIBENet
from dataset import get_dataloader
import config


class Tester:
    def __init__(self, model, test_loader, device):
        self.model = model.to(device)
        self.test_loader = test_loader
        self.device = device
        self.model.eval()
    
    def test(self):
        correct = 0
        total = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for print_img, vein_img, labels in tqdm(self.test_loader, desc='Testing'):
                print_img = print_img.to(self.device)
                vein_img = vein_img.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(print_img, vein_img)
                _, predicted = outputs.max(1)
                
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        accuracy = 100. * correct / total
        
        return accuracy, np.array(all_preds), np.array(all_labels)
    
    def evaluate(self):
        print(f'测试样本数: {len(self.test_loader.dataset)}')
        print('-' * 50)
        
        accuracy, preds, labels = self.test()
        
        print(f'\n测试准确率: {accuracy:.2f}%')
        
        return accuracy, preds, labels
    
    def load_checkpoint(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        print(f'已加载模型: {path}')
        if 'best_acc' in checkpoint:
            print(f'训练时最佳准确率: {checkpoint["best_acc"]:.2f}%')
        if 'best_epoch' in checkpoint and checkpoint['best_epoch'] > 0:
            print(f'最佳模型来自训练第 {checkpoint["best_epoch"]} 轮')


def plot_confusion_matrix(cm, save_path='confusion_matrix.png', num_classes_to_show=50):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import seaborn as sns
        import warnings
        warnings.filterwarnings('ignore', category=UserWarning)
        
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        
        plt.figure(figsize=(12, 10))
        
        cm_subset = cm[:num_classes_to_show, :num_classes_to_show]
        
        sns.heatmap(cm_subset, annot=False, fmt='d', cmap='Blues')
        plt.xlabel('Predicted Class')
        plt.ylabel('True Class')
        plt.title(f'Confusion Matrix (First {num_classes_to_show} Classes)')
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.close()
        print(f'混淆矩阵已保存到: {save_path}')
    except ImportError:
        print('matplotlib或seaborn未安装，跳过混淆矩阵绘制')


def analyze_results(preds, labels, num_classes):
    print('\n' + '=' * 50)
    print('详细分析结果')
    print('=' * 50)
    
    cm = confusion_matrix(labels, preds)
    
    class_correct = cm.diagonal()
    class_total = cm.sum(axis=1)
    
    class_acc = class_correct / (class_total + 1e-10) * 100
    
    print(f'\n各类别准确率统计:')
    print(f'  最高准确率: {class_acc.max():.2f}% (类别 {class_acc.argmax()})')
    print(f'  最低准确率: {class_acc.min():.2f}% (类别 {class_acc.argmin()})')
    print(f'  平均准确率: {class_acc.mean():.2f}%')
    print(f'  准确率标准差: {class_acc.std():.2f}%')
    
    perfect_classes = np.where(class_acc == 100)[0]
    if len(perfect_classes) > 0:
        print(f'\n  完美识别的类别数: {len(perfect_classes)}/{num_classes}')
    
    low_acc_classes = np.where(class_acc < 50)[0]
    if len(low_acc_classes) > 0:
        print(f'  准确率低于50%的类别: {low_acc_classes.tolist()}')
    
    return cm, class_acc


def main(dataset_name=None):
    if dataset_name is None:
        dataset_name = config.DEFAULT_DATASET
    
    dataset_cfg = config.get_dataset_config(dataset_name)
    num_classes = dataset_cfg['num_classes']
    save_dir = config.get_save_dir(dataset_name)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'使用设备: {device}')
    print(f'数据集: {dataset_name}')
    print(f'类别数: {num_classes}')
    
    test_loader = get_dataloader(
        dataset_name,
        mode='test',
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
        shuffle=False
    )
    
    model = VIBENet(num_classes=num_classes, feature_dim=config.FEATURE_DIM)
    
    tester = Tester(model, test_loader, device)
    
    checkpoint_path = os.path.join(save_dir, 'best_model.pth')
    if os.path.exists(checkpoint_path):
        tester.load_checkpoint(checkpoint_path)
    else:
        print('警告: 未找到训练好的模型，使用随机初始化的模型进行测试')
        print(f'请先运行 train.py 进行训练')
    
    accuracy, preds, labels = tester.evaluate()
    
    cm, class_acc = analyze_results(preds, labels, num_classes)
    
    plot_confusion_matrix(cm, save_path=os.path.join(save_dir, 'confusion_matrix.png'), num_classes_to_show=50)
    
    print('\n' + '=' * 50)
    print('测试完成!')
    print('=' * 50)


if __name__ == '__main__':
    main()
