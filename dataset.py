import os
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms
import config


class HandsDataset(Dataset):
    def __init__(self, data_dir, mode='train', transform=None):
        self.mode = mode
        self.transform = transform
        
        if mode == 'train':
            self.print_dir = os.path.join(data_dir, 'print-train')
            self.vein_dir = os.path.join(data_dir, 'vein-train')
        else:
            self.print_dir = os.path.join(data_dir, 'print-test')
            self.vein_dir = os.path.join(data_dir, 'vein-test')
        
        self.classes = sorted([d for d in os.listdir(self.print_dir) 
                              if os.path.isdir(os.path.join(self.print_dir, d))])
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        
        self.samples = []
        for cls in self.classes:
            print_cls_dir = os.path.join(self.print_dir, cls)
            vein_cls_dir = os.path.join(self.vein_dir, cls)
            
            print_files = sorted([
                f for f in os.listdir(print_cls_dir)
                if os.path.isfile(os.path.join(print_cls_dir, f))
            ])
            vein_files = sorted([
                f for f in os.listdir(vein_cls_dir)
                if os.path.isfile(os.path.join(vein_cls_dir, f))
            ])

            print_map = {os.path.splitext(f)[0]: f for f in print_files}
            vein_map = {os.path.splitext(f)[0]: f for f in vein_files}
            common_stems = sorted(set(print_map.keys()) & set(vein_map.keys()))

            missing_in_vein = len(print_map) - len(common_stems)
            missing_in_print = len(vein_map) - len(common_stems)
            if missing_in_vein > 0 or missing_in_print > 0:
                print(
                    f"[警告] 类别 {cls} 模态文件不完全匹配: "
                    f"print缺失{missing_in_print}, vein缺失{missing_in_vein}"
                )

            for stem in common_stems:
                p_file = print_map[stem]
                v_file = vein_map[stem]
                self.samples.append((
                    os.path.join(print_cls_dir, p_file),
                    os.path.join(vein_cls_dir, v_file),
                    self.class_to_idx[cls]
                ))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        print_path, vein_path, label = self.samples[idx]
        
        print_img = Image.open(print_path).convert('L')
        vein_img = Image.open(vein_path).convert('L')
        
        if self.transform:
            print_img = self.transform['print'](print_img)
            vein_img = self.transform['vein'](vein_img)
        
        return print_img, vein_img, label


def get_transforms(mode='train'):
    if mode == 'train':
        print_transform = transforms.Compose([
            transforms.Resize(config.PRINT_IMAGE_SIZE),
            transforms.RandomAffine(
                degrees=config.TRAIN_ROTATION_DEGREES,
                translate=(config.TRAIN_TRANSLATE, config.TRAIN_TRANSLATE),
                scale=(config.TRAIN_SCALE_MIN, config.TRAIN_SCALE_MAX)
            ),
            transforms.ColorJitter(
                brightness=config.TRAIN_BRIGHTNESS,
                contrast=config.TRAIN_CONTRAST
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5]),
            transforms.RandomErasing(p=config.TRAIN_RANDOM_ERASE_P, scale=(0.01, 0.08), ratio=(0.3, 3.3))
        ])

        vein_transform = transforms.Compose([
            transforms.Resize(config.VEIN_IMAGE_SIZE),
            transforms.RandomAffine(
                degrees=config.TRAIN_ROTATION_DEGREES,
                translate=(config.TRAIN_TRANSLATE, config.TRAIN_TRANSLATE),
                scale=(config.TRAIN_SCALE_MIN, config.TRAIN_SCALE_MAX)
            ),
            transforms.ColorJitter(
                brightness=config.TRAIN_BRIGHTNESS,
                contrast=config.TRAIN_CONTRAST
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5]),
            transforms.RandomErasing(p=config.TRAIN_RANDOM_ERASE_P, scale=(0.01, 0.08), ratio=(0.3, 3.3))
        ])
    else:
        print_transform = transforms.Compose([
            transforms.Resize(config.PRINT_IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])

        vein_transform = transforms.Compose([
            transforms.Resize(config.VEIN_IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])
    
    return {'print': print_transform, 'vein': vein_transform}


def get_dataloader(data_dir, mode='train', batch_size=32, num_workers=4, shuffle=True):
    transform = get_transforms(mode=mode)
    dataset = HandsDataset(data_dir, mode=mode, transform=transform)
    if mode != 'train':
        shuffle = False
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True
    )
    return dataloader


if __name__ == '__main__':
    train_loader = get_dataloader(config.DATA_DIR, mode='train', batch_size=4)
    print(f"训练集样本数: {len(train_loader.dataset)}")
    
    for print_img, vein_img, label in train_loader:
        print(f"掌纹图像形状: {print_img.shape}")
        print(f"掌静脉图像形状: {vein_img.shape}")
        print(f"标签: {label}")
        break
