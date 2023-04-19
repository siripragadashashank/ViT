import torch
import pytorch_lightning as pl

# Torchvision
from torchvision.datasets import CIFAR10
from torchvision import transforms

from torch.utils.data import DataLoader


class ViTDataLoader:
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path

    def get_train_loader(self):

        train_transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        train_dataset = CIFAR10(
            root=self.dataset_path,
            train=True,
            transform=train_transform,
            download=True
        )

        pl.seed_everything(42)
        train_set, _ = torch.utils.data.random_split(train_dataset, [45000, 5000])

        train_loader = DataLoader(
            train_set,
            batch_size=4,
            shuffle=True,
            drop_last=True,
            pin_memory=True,
            num_workers=4
        )

        return train_loader

    def get_val_loader(self):

        val_transform = transforms.Compose(
            transforms.ToTensor(),
        )

        val_dataset = CIFAR10(
            root=self.dataset_path,
            train=True,
            transform=val_transform,
            download=True
        )

        pl.seed_everything(42)
        _, val_set = torch.utils.data.random_split(val_dataset, [45000, 5000])
        val_loader = DataLoader(
            val_set,
            batch_size=4,
            shuffle=False,
            drop_last=False,
            num_workers=4
        )

        return val_loader

    def get_test_loader(self):
        test_transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        test_dataset = CIFAR10(
            root=self.dataset_path,
            train=False,
            transform=test_transform,
            download=True
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=4,
            shuffle=False,
            drop_last=False,
            num_workers=4
        )

        return test_loader


if __name__ == '__main__':
    vit_loader = ViTDataLoader(dataset_path='data')
    train = vit_loader.get_train_loader()
    val = vit_loader.get_val_loader()
    print(next(iter(train))[0].shape)


