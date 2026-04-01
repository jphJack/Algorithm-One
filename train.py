import os
import time
import math
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.vibe_net import VIBENet
from dataset import get_dataloader
import config


class Trainer:
    def __init__(self, model, train_loader, val_loader, device, save_dir='checkpoints'):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.save_dir = save_dir
        
        os.makedirs(save_dir, exist_ok=True)
        
        self.criterion = nn.CrossEntropyLoss(label_smoothing=config.LABEL_SMOOTHING)
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=config.LEARNING_RATE,
            weight_decay=config.WEIGHT_DECAY
        )
        self.scheduler = self._build_scheduler()
        
        self.best_acc = 0.0
        self.best_val_loss = float('inf')
        self.best_epoch = -1  # 记录最佳模型的epoch
        self.early_stop_counter = 0
        self.train_losses = []
        self.train_accs = []
        self.val_losses = []
        self.val_accs = []

    def _build_scheduler(self):
        warmup_epochs = max(0, min(config.WARMUP_EPOCHS, config.NUM_EPOCHS - 1))
        total_epochs = config.NUM_EPOCHS
        min_lr_ratio = 1e-2

        def lr_lambda(epoch):
            if warmup_epochs > 0 and epoch < warmup_epochs:
                return float(epoch + 1) / float(warmup_epochs)

            progress_denom = max(1, total_epochs - warmup_epochs)
            progress = (epoch - warmup_epochs) / progress_denom
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
            return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

        return LambdaLR(self.optimizer, lr_lambda=lr_lambda)
    
    def train_epoch(self, epoch):
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch+1}/{config.NUM_EPOCHS}')
        
        for batch_idx, (print_img, vein_img, labels) in enumerate(pbar, start=1):
            print_img = print_img.to(self.device)
            vein_img = vein_img.to(self.device)
            labels = labels.to(self.device)
            
            self.optimizer.zero_grad()
            
            outputs = self.model(print_img, vein_img)
            loss = self.criterion(outputs, labels)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            pbar.set_postfix({
                'loss': f'{running_loss/batch_idx:.4f}',
                'acc': f'{100.*correct/total:.2f}%'
            })
        
        epoch_loss = running_loss / len(self.train_loader)
        epoch_acc = 100. * correct / total
        
        return epoch_loss, epoch_acc
    
    def validate(self):
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for print_img, vein_img, labels in tqdm(self.val_loader, desc='Validating'):
                print_img = print_img.to(self.device)
                vein_img = vein_img.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(print_img, vein_img)
                loss = self.criterion(outputs, labels)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                running_loss += loss.item()
        
        avg_loss = running_loss / len(self.val_loader)
        acc = 100. * correct / total
        return avg_loss, acc
    
    def save_checkpoint(self, epoch, is_best=False):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_acc': self.best_acc,
            'best_val_loss': self.best_val_loss,
            'best_epoch': self.best_epoch,
            'train_losses': self.train_losses,
            'train_accs': self.train_accs,
            'val_losses': self.val_losses,
            'val_accs': self.val_accs
        }
        
        path = os.path.join(self.save_dir, f'checkpoint_epoch_{epoch+1}.pth')
        torch.save(checkpoint, path)
        
        if is_best:
            best_path = os.path.join(self.save_dir, 'best_model.pth')
            torch.save(checkpoint, best_path)
            print(f'保存最佳模型，准确率: {self.best_acc:.2f}%')
    
    def load_checkpoint(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.best_acc = checkpoint['best_acc']
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        self.best_epoch = checkpoint.get('best_epoch', -1)
        self.train_losses = checkpoint['train_losses']
        self.train_accs = checkpoint['train_accs']
        self.val_losses = checkpoint.get('val_losses', [])
        self.val_accs = checkpoint['val_accs']
        return checkpoint['epoch']
    
    def train(self, start_epoch=0):
        total_start_time = time.time()
        
        print(f'开始训练，设备: {self.device}')
        print(f'训练样本数: {len(self.train_loader.dataset)}')
        print(f'验证样本数: {len(self.val_loader.dataset)}')
        print('-' * 50)
        
        for epoch in range(start_epoch, config.NUM_EPOCHS):
            epoch_start_time = time.time()
            
            train_loss, train_acc = self.train_epoch(epoch)
            val_loss, val_acc = self.validate()
            
            self.scheduler.step()
            
            self.train_losses.append(train_loss)
            self.train_accs.append(train_acc)
            self.val_losses.append(val_loss)
            self.val_accs.append(val_acc)
            
            epoch_time = time.time() - epoch_start_time
            
            print(f'\nEpoch {epoch+1}/{config.NUM_EPOCHS} - {epoch_time:.1f}s')
            print(f'  训练损失: {train_loss:.4f}, 训练准确率: {train_acc:.2f}%')
            print(f'  验证损失: {val_loss:.4f}, 验证准确率: {val_acc:.2f}%')
            print(f'  学习率: {self.optimizer.param_groups[0]["lr"]:.6f}')
            
            improve_acc = val_acc > (self.best_acc + config.EARLY_STOPPING_MIN_DELTA)
            improve_tie_break = (
                abs(val_acc - self.best_acc) <= config.EARLY_STOPPING_MIN_DELTA
                and val_loss < self.best_val_loss
            )
            is_best = improve_acc or improve_tie_break
            if is_best:
                self.best_acc = val_acc
                self.best_val_loss = val_loss
                self.best_epoch = epoch + 1  # 记录最佳模型的epoch（从1开始计数）
                self.early_stop_counter = 0
            else:
                self.early_stop_counter += 1
            
            if (epoch + 1) % 10 == 0 or is_best:
                self.save_checkpoint(epoch, is_best)

            if self.early_stop_counter >= config.EARLY_STOPPING_PATIENCE:
                print(
                    f'触发Early Stopping: 连续{config.EARLY_STOPPING_PATIENCE}轮无验证集提升，提前结束训练。'
                )
                break
        
        total_time = time.time() - total_start_time
        hours, rem = divmod(total_time, 3600)
        minutes, seconds = divmod(rem, 60)
        
        print('\n训练完成!')
        print(f'总训练时长: {int(hours)}小时 {int(minutes)}分钟 {seconds:.1f}秒')
        print(f'最佳验证准确率: {self.best_acc:.2f}%')
        if self.best_epoch > 0:
            print(f'最佳模型(best_model.pth)来自训练第 {self.best_epoch} 轮')
        
        return self.train_losses, self.train_accs, self.val_accs


def plot_training_curves(train_losses, train_accs, val_accs, save_path='training_curves.png'):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import warnings
        warnings.filterwarnings('ignore', category=UserWarning)
        
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        epochs = range(1, len(train_losses) + 1)
        
        ax1.plot(epochs, train_losses, 'b-', label='Train Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training Loss Curve')
        ax1.legend()
        ax1.grid(True)
        
        ax2.plot(epochs, train_accs, 'b-', label='Train Acc')
        ax2.plot(epochs, val_accs, 'r-', label='Val Acc')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.set_title('Accuracy Curve')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.close()
        print(f'训练曲线已保存到: {save_path}')
    except ImportError:
        print('matplotlib未安装，跳过绘图')


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'使用设备: {device}')
    
    train_loader = get_dataloader(
        config.DATA_DIR,
        mode='train',
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
        shuffle=True
    )
    
    val_loader = get_dataloader(
        config.DATA_DIR,
        mode='test',
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
        shuffle=False
    )
    
    model = VIBENet(num_classes=config.NUM_CLASSES, feature_dim=config.FEATURE_DIM)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f'模型参数量: {total_params:,}')
    
    trainer = Trainer(model, train_loader, val_loader, device)
    
    train_losses, train_accs, val_accs = trainer.train()
    
    plot_training_curves(train_losses, train_accs, val_accs)


if __name__ == '__main__':
    main()
