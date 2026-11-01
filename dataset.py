"""Utilities for downloading and loading CIFAR-10 and CIFAR-100."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from torch.utils.data import DataLoader
from torchvision import datasets, transforms


CifarName = Literal["cifar10", "cifar100"]


def get_cifar_datasets(
    name: CifarName,
    data_dir: str | Path = "data",
    *,
    download: bool = True,
    transform: transforms.Compose | None = None,
    train_transform: transforms.Compose | None = None,
    test_transform: transforms.Compose | None = None,
) -> tuple[datasets.VisionDataset, datasets.VisionDataset]:
    """Return the training and test splits for a CIFAR dataset.

    Args:
        name: Either ``"cifar10"`` or ``"cifar100"``.
        data_dir: Directory in which the dataset is stored.
        download: Download the dataset when it is not already present.
        transform: Transform applied to both splits. Cannot be combined with
            ``train_transform`` or ``test_transform``.
        train_transform: Transform applied only to the training split.
        test_transform: Transform applied only to the test split.
    """
    if name not in ("cifar10", "cifar100"):
        raise ValueError("name must be either 'cifar10' or 'cifar100'")
    if transform is not None and (train_transform is not None or test_transform is not None):
        raise ValueError("transform cannot be combined with split-specific transforms")

    dataset_type = datasets.CIFAR10 if name == "cifar10" else datasets.CIFAR100
    train_transform = train_transform or transform or transforms.ToTensor()
    test_transform = test_transform or transform or transforms.ToTensor()
    root = Path(data_dir)

    train_dataset = dataset_type(
        root=str(root),
        train=True,
        transform=train_transform,
        download=download,
    )
    test_dataset = dataset_type(
        root=str(root),
        train=False,
        transform=test_transform,
        download=download,
    )
    return train_dataset, test_dataset


def get_cifar_dataloaders(
    name: CifarName,
    data_dir: str | Path = "data",
    *,
    batch_size: int = 128,
    num_workers: int = 0,
    download: bool = True,
    transform: transforms.Compose | None = None,
) -> tuple[DataLoader, DataLoader]:
    """Return training and test DataLoaders for CIFAR-10 or CIFAR-100."""
    if batch_size <= 0:
        raise ValueError("batch_size must be greater than zero")
    if num_workers < 0:
        raise ValueError("num_workers must not be negative")

    train_dataset, test_dataset = get_cifar_datasets(
        name,
        data_dir,
        download=download,
        transform=transform,
    )
    return (
        DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
        ),
        DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        ),
    )


__all__ = ["CifarName", "get_cifar_datasets", "get_cifar_dataloaders"]