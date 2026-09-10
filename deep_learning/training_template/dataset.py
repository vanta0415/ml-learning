from pathlib import Path
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

import config


def set_seed(seed=0):
    torch.manual_seed(seed)


def get_loaders():
    data_dir = Path(__file__).resolve().parent.parent / config.DATA_DIR

    train_tf = [transforms.ToTensor()]
    if config.USE_AUG:
        train_tf = [
            transforms.RandomRotation(10),
            transforms.ToTensor(),
        ]
    test_tf = transforms.ToTensor()

    train_full = datasets.MNIST(
        root=str(data_dir), train=True, download=False,
        transform=transforms.Compose(train_tf),
    )
    # 验证/测试不做增强
    base_full = datasets.MNIST(
        root=str(data_dir), train=True, download=False,
        transform=test_tf,
    )
    test_full = datasets.MNIST(
        root=str(data_dir), train=False, download=False,
        transform=test_tf,
    )

    train_set = Subset(train_full, list(range(config.TRAIN_SIZE)))
    val_set = Subset(base_full, list(range(config.TRAIN_SIZE, config.TRAIN_SIZE + config.VAL_SIZE)))
    test_set = Subset(test_full, list(range(config.TEST_SIZE)))

    train_loader = DataLoader(train_set, batch_size=config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=256, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=256, shuffle=False)
    return train_loader, val_loader, test_loader
