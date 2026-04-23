import os
import random
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision.transforms import functional as F
import config


class TrainTransform:
    def __init__(self, img_size=(128, 128)):
        self.ToPILImage = transforms.ToPILImage()
        self.resize = transforms.Resize(img_size)
        self.rotate_degrees = 15
        self.color_jitter = transforms.ColorJitter(
            brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1
        )
        self.normalize = transforms.Normalize(
            [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
        )

    def __call__(self, img1, img2):
        img1 = self.ToPILImage(img1)
        img2 = self.ToPILImage(img2)

        img1 = self.resize(img1)
        img2 = self.resize(img2)

        i, j, h, w = transforms.RandomCrop.get_params(
            img1, output_size=(img1.height, img1.width)
        )
        img1 = F.crop(img1, i, j, h, w)
        img2 = F.crop(img2, i, j, h, w)

        angle = random.uniform(-self.rotate_degrees, self.rotate_degrees)
        img1 = F.rotate(img1, angle)
        img2 = F.rotate(img2, angle)

        img1 = self.color_jitter(img1)
        img2 = self.color_jitter(img2)

        img1 = F.to_tensor(img1)
        img2 = F.to_tensor(img2)
        img1 = self.normalize(img1)
        img2 = self.normalize(img2)

        return img1, img2


class TestTransform:
    def __init__(self, img_size=(128, 128)):
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(img_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def __call__(self, img):
        return self.transform(img)


class TrainDataset(Dataset):
    def __init__(self, print_root_dir, vein_root_dir, img_size=(128, 128)):
        self.print_root_dir = print_root_dir
        self.vein_root_dir = vein_root_dir

        self.person_path = sorted(os.listdir(self.print_root_dir))
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.person_path)}

        self.transform = TrainTransform(img_size)

        self.samples_per_person = []
        for person_name in self.person_path:
            print_dir = os.path.join(self.print_root_dir, person_name)
            n_imgs = len(os.listdir(print_dir))
            self.samples_per_person.append(n_imgs)

    def __getitem__(self, idx):
        person_idx = 0
        cumulative = 0
        for i, n in enumerate(self.samples_per_person):
            if idx < cumulative + n:
                person_idx = i
                break
            cumulative += n

        person_name = self.person_path[person_idx]
        local_idx = idx - cumulative

        print_imgs_path = sorted(
            os.listdir(os.path.join(self.print_root_dir, person_name))
        )
        vein_imgs_path = sorted(
            os.listdir(os.path.join(self.vein_root_dir, person_name))
        )

        print_img_path = print_imgs_path[local_idx]
        vein_img_path = vein_imgs_path[local_idx]

        p_img_item_path = os.path.join(
            self.print_root_dir, person_name, print_img_path
        )
        v_img_item_path = os.path.join(
            self.vein_root_dir, person_name, vein_img_path
        )

        p_img = cv2.imread(p_img_item_path)
        p_img = torch.tensor(p_img).to(torch.float).permute(2, 0, 1)
        v_img = cv2.imread(v_img_item_path)
        v_img = torch.tensor(v_img).to(torch.float).permute(2, 0, 1)
        p_img, v_img = self.transform(p_img, v_img)

        label = self.class_to_idx[person_name]
        return p_img, v_img, label

    def __len__(self):
        return sum(self.samples_per_person)


class TestDataset(Dataset):
    def __init__(self, print_root_dir, vein_root_dir, img_size=(128, 128)):
        self.print_root_dir = print_root_dir
        self.vein_root_dir = vein_root_dir

        self.person_path = sorted(os.listdir(self.print_root_dir))
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.person_path)}

        self.transform = TestTransform(img_size)

        self.samples = []
        for person_name in self.person_path:
            print_dir = os.path.join(self.print_root_dir, person_name)
            print_imgs_path = sorted(os.listdir(print_dir))
            vein_dir = os.path.join(self.vein_root_dir, person_name)
            vein_imgs_path = sorted(os.listdir(vein_dir))

            for p_file, v_file in zip(print_imgs_path, vein_imgs_path):
                self.samples.append((
                    os.path.join(print_dir, p_file),
                    os.path.join(vein_dir, v_file),
                    self.class_to_idx[person_name]
                ))

    def __getitem__(self, idx):
        p_img_item_path, v_img_item_path, label = self.samples[idx]

        p_img = cv2.imread(p_img_item_path)
        p_img = torch.tensor(p_img).to(torch.float).permute(2, 0, 1)
        p_img = self.transform(p_img)

        v_img = cv2.imread(v_img_item_path)
        v_img = torch.tensor(v_img).to(torch.float).permute(2, 0, 1)
        v_img = self.transform(v_img)

        return p_img, v_img, label

    def __len__(self):
        return len(self.samples)


def get_dataloader(dataset_name, mode='train', batch_size=32, num_workers=4, shuffle=True):
    dataset_cfg = config.get_dataset_config(dataset_name)
    img_size = dataset_cfg['img_size']

    if mode == 'train':
        print_dir = dataset_cfg['print_train_dir']
        vein_dir = dataset_cfg['vein_train_dir']
        dataset = TrainDataset(print_dir, vein_dir, img_size=img_size)
    else:
        print_dir = dataset_cfg['print_test_dir']
        vein_dir = dataset_cfg['vein_test_dir']
        dataset = TestDataset(print_dir, vein_dir, img_size=img_size)

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True
    )
    return dataloader


if __name__ == '__main__':
    for dataset_name in ['HandsData', 'CASIA', 'QH', 'TJ']:
        print(f"\n测试数据集: {dataset_name}")
        try:
            train_loader = get_dataloader(
                dataset_name, mode='train', batch_size=4, num_workers=0
            )
            print(f"  训练集样本数: {len(train_loader.dataset)}")
            print(f"  类别数: {len(train_loader.dataset.class_to_idx)}")

            for print_img, vein_img, label in train_loader:
                print(f"  掌纹图像形状: {print_img.shape}")
                print(f"  掌静脉图像形状: {vein_img.shape}")
                print(f"  标签: {label}")
                break
        except Exception as e:
            print(f"  错误: {e}")
