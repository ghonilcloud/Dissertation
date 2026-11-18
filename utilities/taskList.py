"""Definitions and dataset splitting utilities for continual-learning tasks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import utilities.windowsTorchRuntime  # noqa: F401
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
    remap_labels: bool = False,
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
        subset = Subset(dataset, class_indices)
        if remap_labels:
            label_map = {label: index for index, label in enumerate(task.classes)}
            subset = _RemappedSubset(subset, label_map)
        loaders.append(
            DataLoader(
                subset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=num_workers,
                pin_memory=torch.cuda.is_available(),
            )
        )
    return loaders


class _RemappedSubset(Dataset):
    """Subset view that maps global class labels to local task labels."""

    def __init__(self, subset: Subset, label_map: dict[int, int]) -> None:
        self.subset = subset
        self.label_map = label_map

    def __len__(self) -> int:
        return len(self.subset)

    def __getitem__(self, index: int):
        inputs, label = self.subset[index]
        return inputs, self.label_map[int(label)]


__all__ = [
    "CIFAR10_CLASSES",
    "CIFAR10_TASKS",
    "Task",
    "build_task_loaders",
]
