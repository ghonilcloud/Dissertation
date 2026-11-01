"""Shared image preprocessing pipelines for CIFAR datasets."""

from __future__ import annotations

from typing import Literal

from torchvision import transforms


CifarName = Literal["cifar10", "cifar100"]

_CIFAR_STATS = {
    "cifar10": (
        (0.4914, 0.4822, 0.4465),
        (0.2470, 0.2435, 0.2616),
    ),
    "cifar100": (
        (0.5071, 0.4867, 0.4408),
        (0.2675, 0.2565, 0.2761),
    ),
}


def get_cifar_transforms(
    name: CifarName,
    *,
    augment_training: bool = True,
) -> tuple[transforms.Compose, transforms.Compose]:
    """Return separate training and evaluation transforms for CIFAR."""
    if name not in _CIFAR_STATS:
        raise ValueError("name must be either 'cifar10' or 'cifar100'")

    mean, standard_deviation = _CIFAR_STATS[name]
    training_steps = []
    if augment_training:
        training_steps.extend(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
            ]
        )
    training_steps.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean, standard_deviation),
        ]
    )
    evaluation_steps = [
        transforms.ToTensor(),
        transforms.Normalize(mean, standard_deviation),
    ]
    return transforms.Compose(training_steps), transforms.Compose(evaluation_steps)


__all__ = ["CifarName", "get_cifar_transforms"]
