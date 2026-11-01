"""Definitions and dataset splitting utilities for continual-learning tasks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset


CIFAR10_CLASSES = (
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
)
CIFAR10_TASKS = (
    (0, 1),
    (2, 3),
    (4, 5),
    (6, 7),
    (8, 9),
)


@dataclass(frozen=True)
class Task:
    """A class-incremental task and its CIFAR class labels."""

    number: int
    classes: tuple[int, ...]


def build_task_loaders(
    dataset: Dataset,
    tasks: Sequence[Task],
    *,
    batch_size: int,
    num_workers: int = 0,
) -> list[DataLoader]:
    """Filter a labelled dataset into one DataLoader per task."""
    if batch_size <= 0:
        raise ValueError("batch_size must be greater than zero")
    if num_workers < 0:
        raise ValueError("num_workers must not be negative")
    if not hasattr(dataset, "targets"):
        raise ValueError("dataset must expose a targets attribute")

    targets = np.asarray(dataset.targets)
    loaders = []
    for task in tasks:
        class_indices = np.flatnonzero(np.isin(targets, task.classes)).tolist()
        loaders.append(
            DataLoader(
                Subset(dataset, class_indices),
                batch_size=batch_size,
                shuffle=True,
                num_workers=num_workers,
                pin_memory=torch.cuda.is_available(),
            )
        )
    return loaders


__all__ = [
    "CIFAR10_CLASSES",
    "CIFAR10_TASKS",
    "Task",
    "build_task_loaders",
]
