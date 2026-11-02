"""SpikingJelly baseline for class-incremental CIFAR-10 learning."""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from spikingjelly.activation_based import functional, neuron, surrogate
from torch import nn
from torch.utils.data import DataLoader

from dataset import get_cifar_datasets
from preprocessing import get_cifar_transforms
from taskList import CIFAR10_CLASSES, CIFAR10_TASKS, Task, build_task_loaders

DEFAULT_SEEDS = (42, 123, 456, 789, 1011)


class BaselineSNN(nn.Module):
    """A fully connected LIF SNN for 32x32 RGB CIFAR images."""

    def __init__(self, num_classes: int = 10, time_steps: int = 10) -> None:
        super().__init__()
        if time_steps <= 0:
            raise ValueError("time_steps must be greater than zero")
        self.time_steps = time_steps
        self.network = nn.Sequential(
            nn.Flatten(),
            nn.Linear(3 * 32 * 32, 512),
            neuron.LIFNode(surrogate_function=surrogate.Sigmoid()),
            nn.Linear(512, 256),
            neuron.LIFNode(surrogate_function=surrogate.Sigmoid()),
            nn.Linear(256, num_classes),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        functional.reset_net(self)
        output = torch.zeros(
            inputs.shape[0],
            self.network[-1].out_features,
            device=inputs.device,
        )
        for _ in range(self.time_steps):
            output += self.network(inputs)
        return output / self.time_steps


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
    task_number: int,
    progress: bool = False,
) -> tuple[list[float], float]:
    """Train the SNN on one task using surrogate-gradient backpropagation."""
    if epochs <= 0:
        raise ValueError("epochs must be greater than zero")
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    losses = []
    start_time = time.perf_counter()
    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        batches = 0
        for inputs, labels in loader:
            optimizer.zero_grad()
            loss = criterion(model(inputs.to(device)), labels.to(device))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            batches += 1
        if progress:
            print(
                f"Task {task_number}, epoch {epoch}/{epochs}, "
                f"loss: {total_loss / batches:.4f}"
            )
        losses.append(total_loss / batches)
    return losses, time.perf_counter() - start_time


def run_experiment(
    *,
    data_dir: str | Path = "data",
    batch_size: int = 128,
    epochs: int = 5,
    learning_rate: float = 1e-3,
    time_steps: int = 10,
    seed: int = 42,
    num_workers: int = 0,
    download: bool = True,
    max_tasks: int | None = None,
    progress: bool = False,
    seeds: Sequence[int] = DEFAULT_SEEDS,
) -> list[dict[str, object]]:
    """Train sequentially on CIFAR-10 tasks using a SpikingJelly SNN."""
    if max_tasks is not None and not 1 <= max_tasks <= len(CIFAR10_TASKS):
        raise ValueError(f"max_tasks must be between 1 and {len(CIFAR10_TASKS)}")
    if seeds == DEFAULT_SEEDS and seed != 42:
        seeds = (seed,)
    all_results = []
    for run_seed in seeds:
        all_results.append(
            _run_single_experiment(
                data_dir=data_dir,
                batch_size=batch_size,
                epochs=epochs,
                learning_rate=learning_rate,
                time_steps=time_steps,
                seed=run_seed,
                num_workers=num_workers,
                download=download,
                max_tasks=max_tasks,
                progress=progress,
            )
        )
    return all_results


def _run_single_experiment(
    *,
    data_dir: str | Path,
    batch_size: int,
    epochs: int,
    learning_rate: float,
    time_steps: int,
    seed: int,
    num_workers: int,
    download: bool,
    max_tasks: int | None,
    progress: bool,
) -> dict[str, object]:
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

    model = BaselineSNN(time_steps=time_steps).to(device)
    seen_classes: list[int] = []
    task_results = []
    best_task_accuracies: list[float] = []
    total_training_time = 0.0
    for task, train_loader in zip(tasks, train_loaders):
        losses, training_time = train_task(
            model,
            train_loader,
            device=device,
            epochs=epochs,
            learning_rate=learning_rate,
            task_number=task.number,
            progress=progress,
        )
        total_training_time += training_time
        seen_classes.extend(task.classes)
        task_accuracies = [
            evaluate(model, test_loaders[index], device, tasks[index].classes)
            for index in range(task.number)
        ]
        previous_forgetting = {
            f"task_{index + 1}": best_task_accuracies[index]
            - task_accuracies[index]
            for index in range(task.number - 1)
        }
        best_task_accuracies = [
            max(previous, current)
            for previous, current in zip(
                best_task_accuracies + [0.0],
                task_accuracies,
            )
        ]
        task_results.append(
            {
                "task": task.number,
                "classes": [CIFAR10_CLASSES[index] for index in task.classes],
                "current_task_accuracy": task_accuracies[-1],
                "seen_accuracy": evaluate(
                    model, full_test_loader, device, seen_classes
                ),
                "previous_task_accuracy": {
                    f"task_{index + 1}": task_accuracies[index]
                    for index in range(task.number - 1)
                },
                "average_accuracy": sum(task_accuracies) / len(task_accuracies),
                "forgetting": previous_forgetting,
                "training_loss": losses,
                "training_time_seconds": training_time,
                "task_accuracies": task_accuracies,
            }
        )
    return {
        "seed": seed,
        "tasks": task_results,
        "final_accuracy": task_results[-1]["average_accuracy"],
        "final_seen_accuracy": task_results[-1]["seen_accuracy"],
        "total_training_time_seconds": total_training_time,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--time-steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-tasks", type=int, default=None)
    parser.add_argument("--no-download", action="store_true")
    parser.add_argument("--progress", action="store_true")
    args = parser.parse_args()
    print(
        json.dumps(
            run_experiment(
                data_dir=args.data_dir,
                batch_size=args.batch_size,
                epochs=args.epochs,
                learning_rate=args.learning_rate,
                time_steps=args.time_steps,
                seed=args.seed,
                num_workers=args.num_workers,
                download=not args.no_download,
                max_tasks=args.max_tasks,
                progress=args.progress,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
