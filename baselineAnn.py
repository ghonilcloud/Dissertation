"""Baseline ANN and class-incremental CIFAR-10 experiment."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from dataset import get_cifar_datasets
from preprocessing import get_cifar_transforms
from taskList import CIFAR10_CLASSES, CIFAR10_TASKS, Task, build_task_loaders


class BaselineANN(nn.Module):
    """A fully connected ANN for 32x32 RGB CIFAR images."""

    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Flatten(),
            nn.Linear(3 * 32 * 32, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.network(inputs)


def set_seed(seed: int) -> None:
    """Make task ordering and model initialisation reproducible."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    class_labels: Sequence[int] | None = None,
) -> float:
    """Return accuracy, optionally restricted to selected class labels."""
    model.eval()
    correct = total = 0
    allowed = set(class_labels) if class_labels is not None else None
    with torch.inference_mode():
        for inputs, labels in loader:
            if allowed is not None:
                mask = torch.tensor(
                    [int(label) in allowed for label in labels],
                    dtype=torch.bool,
                )
                inputs, labels = inputs[mask], labels[mask]
                if labels.numel() == 0:
                    continue
            predictions = model(inputs.to(device)).argmax(dim=1).cpu()
            correct += int((predictions == labels).sum())
            total += labels.numel()
    return correct / total if total else 0.0


def train_task(
    model: nn.Module,
    loader: DataLoader,
    *,
    device: torch.device,
    epochs: int,
    learning_rate: float,
) -> None:
    """Train the model on one task using cross-entropy and Adam."""
    if epochs <= 0:
        raise ValueError("epochs must be greater than zero")
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    for _ in range(epochs):
        for inputs, labels in loader:
            optimizer.zero_grad()
            loss = criterion(model(inputs.to(device)), labels.to(device))
            loss.backward()
            optimizer.step()


def run_experiment(
    *,
    data_dir: str | Path = "data",
    batch_size: int = 128,
    epochs: int = 5,
    learning_rate: float = 1e-3,
    seed: int = 42,
    num_workers: int = 0,
    download: bool = True,
    max_tasks: int | None = None,
) -> list[dict[str, object]]:
    """Train sequentially on CIFAR-10 tasks.

    Set ``max_tasks=1`` to run only the first task.
    """
    if max_tasks is not None and not 1 <= max_tasks <= len(CIFAR10_TASKS):
        raise ValueError(f"max_tasks must be between 1 and {len(CIFAR10_TASKS)}")
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_transform, test_transform = get_cifar_transforms("cifar10")
    train_dataset, test_dataset = get_cifar_datasets(
        "cifar10",
        data_dir,
        download=download,
        train_transform=train_transform,
        test_transform=test_transform,
    )
    tasks = [
        Task(index + 1, classes)
        for index, classes in enumerate(CIFAR10_TASKS[:max_tasks])
    ]
    train_loaders = build_task_loaders(
        train_dataset, tasks, batch_size=batch_size, num_workers=num_workers
    )
    test_loaders = build_task_loaders(
        test_dataset, tasks, batch_size=batch_size, num_workers=num_workers
    )
    full_test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    model = BaselineANN().to(device)
    results: list[dict[str, object]] = []
    seen_classes: list[int] = []
    for task, train_loader in zip(tasks, train_loaders):
        train_task(
            model,
            train_loader,
            device=device,
            epochs=epochs,
            learning_rate=learning_rate,
        )
        seen_classes.extend(task.classes)
        results.append(
            {
                "task": task.number,
                "classes": [CIFAR10_CLASSES[index] for index in task.classes],
                "seen_accuracy": evaluate(
                    model, full_test_loader, device, seen_classes
                ),
                "task_accuracy": evaluate(
                    model, test_loaders[task.number - 1], device, task.classes
                ),
            }
        )
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--max-tasks",
        type=int,
        default=None,
        help="Number of sequential tasks to run (use 1 for Task 1 only).",
    )
    parser.add_argument("--no-download", action="store_true")
    args = parser.parse_args()
    print(
        json.dumps(
            run_experiment(
                data_dir=args.data_dir,
                batch_size=args.batch_size,
                epochs=args.epochs,
                learning_rate=args.learning_rate,
                seed=args.seed,
                num_workers=args.num_workers,
                download=not args.no_download,
                max_tasks=args.max_tasks,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()