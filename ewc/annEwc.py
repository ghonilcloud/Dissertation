"""ANN with diagonal-Fisher Elastic Weight Consolidation."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Sequence

import utilities.windowsTorchRuntime  # noqa: F401
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset

from baseline.baselineAnn import (
    DEFAULT_SEEDS,
    BaselineANN,
    TaskIncrementalANN,
    evaluate,
    set_seed,
)
from utilities.dataset import get_cifar_datasets
from utilities.preprocessing import get_cifar_transforms
from utilities.taskList import CIFAR10_CLASSES, CIFAR10_TASKS, Task, build_task_loaders


class EWC:
    """Stores diagonal Fisher estimates and parameter snapshots."""

    def __init__(
        self,
        fisher_samples: int = 1024,
        fisher_method: str = "empirical",
    ) -> None:
        if fisher_samples <= 0:
            raise ValueError("fisher_samples must be positive")
        if fisher_method not in {"empirical", "sampled"}:
            raise ValueError("fisher_method must be 'empirical' or 'sampled'")
        self.fisher_samples = fisher_samples
        self.fisher_method = fisher_method
        self.snapshots: list[tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]] = []
        self.sample_counts: list[int] = []

    def penalty(self, model: nn.Module) -> torch.Tensor:
        value = next(model.parameters()).new_zeros(())
        for parameters, fisher in self.snapshots:
            for name, parameter in model.named_parameters():
                value = value + (fisher[name] * (parameter - parameters[name]).square()).sum()
        return value

    def consolidate(self, model: nn.Module, loader: DataLoader, device: torch.device) -> None:
        """Estimate diagonal empirical Fisher from individual example gradients.

        A bounded, shuffled sample keeps consolidation practical while avoiding
        squaring gradients after they have been averaged across a whole batch.
        """
        model.eval()
        fisher = {
            name: torch.zeros_like(parameter, device=device)
            for name, parameter in model.named_parameters()
        }
        criterion = nn.CrossEntropyLoss()
        examples = 0
        for inputs, labels in loader:
            for input_, label in zip(inputs, labels):
                model.zero_grad(set_to_none=True)
                input_ = input_.unsqueeze(0).to(device)
                label = label.unsqueeze(0).to(device)
                logits = model(input_)
                if self.fisher_method == "sampled":
                    with torch.no_grad():
                        label = torch.multinomial(
                            torch.softmax(logits, dim=1), num_samples=1
                        ).reshape(1)
                loss = criterion(logits, label)
                loss.backward()
                for name, parameter in model.named_parameters():
                    if parameter.grad is not None:
                        fisher[name] += parameter.grad.detach().square()
                examples += 1
                if examples >= self.fisher_samples:
                    break
            if examples >= self.fisher_samples:
                break
        if examples == 0:
            raise ValueError("cannot estimate Fisher information from an empty loader")
        self.snapshots.append(
            (
                {
                    name: parameter.detach().clone()
                    for name, parameter in model.named_parameters()
                },
                {name: value / examples for name, value in fisher.items()},
            )
        )
        self.sample_counts.append(examples)


def _forever(loader: DataLoader):
    while True:
        yield from loader


class Replay:
    """Stores a random subset of each finished task's training data.

    per_task=0 disables replay; per_task=None keeps every example, which is
    joint training (the upper bound). Examples are stored as dataset indices,
    so replayed images still get fresh augmentation.
    """

    def __init__(self, per_task: int | None, batch_size: int) -> None:
        if per_task is not None and per_task < 0:
            raise ValueError("per_task must be non-negative or None")
        self.per_task = per_task
        self.batch_size = batch_size
        self.streams: list[tuple[int, object]] = []

    def add(self, loader: DataLoader, task_index: int) -> None:
        if self.per_task == 0:
            return
        dataset = loader.dataset
        count = len(dataset) if self.per_task is None else min(self.per_task, len(dataset))
        indices = torch.randperm(len(dataset))[:count].tolist()
        memory = DataLoader(Subset(dataset, indices), batch_size=self.batch_size, shuffle=True)
        self.streams.append((task_index, _forever(memory)))

    def mix(
        self,
        new_loss: torch.Tensor,
        model: nn.Module,
        criterion: nn.Module,
        device: torch.device,
        current_task: int,
        scenario: str,
    ) -> torch.Tensor:
        """Average the current-task loss with one replayed batch per old task.

        Equal weighting per task (van de Ven et al., 2020) keeps old tasks from
        being drowned out as the number of tasks grows.
        """
        if not self.streams:
            return new_loss
        losses = [new_loss]
        for task_index, stream in self.streams:
            if scenario == "task":
                model.set_task(task_index)
            inputs, labels = next(stream)
            losses.append(criterion(model(inputs.to(device)), labels.to(device)))
        if scenario == "task":
            model.set_task(current_task)
        return torch.stack(losses).mean()


def train_task(
    model: nn.Module,
    loader: DataLoader,
    ewc: EWC,
    replay: Replay,
    scenario: str,
    *,
    device: torch.device,
    epochs: int,
    learning_rate: float,
    ewc_lambda: float,
    task_number: int,
    progress: bool,
) -> tuple[list[float], float]:
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    losses = []
    start = time.perf_counter()
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        batches = 0
        for inputs, labels in loader:
            optimizer.zero_grad()
            loss = criterion(model(inputs.to(device)), labels.to(device))
            loss = replay.mix(loss, model, criterion, device, task_number - 1, scenario)
            loss = loss + 0.5 * ewc_lambda * ewc.penalty(model)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            batches += 1
        losses.append(total_loss / batches)
        if progress:
            print(f"[{time.strftime('%H:%M:%S')}] Task {task_number}, epoch {epoch}/{epochs}, loss: {losses[-1]:.4f}")
    return losses, time.perf_counter() - start


def run_experiment(
    *,
    data_dir: str | Path = "data",
    batch_size: int = 128,
    epochs: int = 5,
    learning_rate: float = 1e-3,
    ewc_lambda: float = 100.0,
    seed: int = 42,
    num_workers: int = 0,
    download: bool = False,
    max_tasks: int | None = None,
    progress: bool = False,
    seeds: Sequence[int] = DEFAULT_SEEDS,
    fisher_samples: int = 1024,
    fisher_method: str = "empirical",
    scenario: str = "class",
    replay_per_task: int = 0,
    joint: bool = False,
) -> list[dict[str, object]]:
    if scenario not in {"class", "task"}:
        raise ValueError("scenario must be 'class' or 'task'")
    if max_tasks is not None and not 1 <= max_tasks <= len(CIFAR10_TASKS):
        raise ValueError(f"max_tasks must be between 1 and {len(CIFAR10_TASKS)}")
    all_results = []
    for run_seed in seeds if seed == 42 else (seed,):
        set_seed(run_seed)
        if progress:
            print(f"[{time.strftime('%H:%M:%S')}] ANN-EWC seed {run_seed} starting")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        train_transform, test_transform = get_cifar_transforms("cifar10")
        train_dataset, test_dataset = get_cifar_datasets(
            "cifar10", data_dir, download=download,
            train_transform=train_transform, test_transform=test_transform,
        )
        tasks = [Task(i + 1, classes) for i, classes in enumerate(CIFAR10_TASKS[:max_tasks])]
        train_loaders = build_task_loaders(
            train_dataset, tasks, batch_size=batch_size, num_workers=num_workers,
            remap_labels=scenario == "task",
        )
        test_loaders = build_task_loaders(
            test_dataset, tasks, batch_size=batch_size, num_workers=num_workers,
            remap_labels=scenario == "task",
        )
        full_test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        model = (
            BaselineANN()
            if scenario == "class"
            else TaskIncrementalANN(num_tasks=len(tasks))
        ).to(device)
        ewc = EWC(fisher_samples=fisher_samples, fisher_method=fisher_method)
        replay = Replay(None if joint else replay_per_task, batch_size)
        task_results, seen_classes, best, total_time = [], [], [], 0.0
        for task, train_loader in zip(tasks, train_loaders):
            if scenario == "task":
                model.set_task(task.number - 1)
            losses, elapsed = train_task(
                model, train_loader, ewc, replay, scenario, device=device, epochs=epochs,
                learning_rate=learning_rate, ewc_lambda=ewc_lambda,
                task_number=task.number, progress=progress,
            )
            total_time += elapsed
            if ewc_lambda > 0:
                ewc.consolidate(model, train_loader, device)
            replay.add(train_loader, task.number - 1)
            if progress:
                print(f"[{time.strftime('%H:%M:%S')}] Task {task.number} done ({elapsed:.1f}s training)")
            seen_classes.extend(task.classes)
            accuracies = []
            for i in range(task.number):
                if scenario == "task":
                    model.set_task(i)
                    accuracies.append(evaluate(model, test_loaders[i], device))
                else:
                    accuracies.append(
                        evaluate(model, test_loaders[i], device, tasks[i].classes)
                    )
            if scenario == "task":
                model.set_task(task.number - 1)
            forgetting = {f"task_{i + 1}": best[i] - accuracies[i] for i in range(task.number - 1)}
            best = [max(old, current) for old, current in zip(best + [0.0], accuracies)]
            task_results.append({
                "task": task.number,
                "classes": [CIFAR10_CLASSES[i] for i in task.classes],
                "current_task_accuracy": accuracies[-1],
                "seen_accuracy": (
                    sum(accuracies) / len(accuracies)
                    if scenario == "task"
                    else evaluate(model, full_test_loader, device, seen_classes)
                ),
                "previous_task_accuracy": {f"task_{i + 1}": accuracies[i] for i in range(task.number - 1)},
                "average_accuracy": sum(accuracies) / len(accuracies),
                "forgetting": forgetting,
                "training_loss": losses,
                "training_time_seconds": elapsed,
                "task_accuracies": accuracies,
            })
        all_results.append({
            "seed": run_seed, "scenario": scenario, "ewc_lambda": ewc_lambda,
            "replay_per_task": "all" if joint else replay_per_task,
            "tasks": task_results,
            "final_accuracy": task_results[-1]["average_accuracy"],
            "final_seen_accuracy": task_results[-1]["seen_accuracy"],
            "total_training_time_seconds": total_time,
        })
    return all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--ewc-lambda", type=float, default=100.0)
    parser.add_argument("--max-tasks", type=int, default=None)
    parser.add_argument("--fisher-samples", type=int, default=1024)
    parser.add_argument("--fisher-method", choices=("empirical", "sampled"), default="empirical")
    parser.add_argument("--scenario", choices=("class", "task"), default="class")
    parser.add_argument("--replay", type=int, default=0, help="stored examples per old task (0 = off)")
    parser.add_argument("--joint", action="store_true", help="replay all old data (upper bound)")
    parser.add_argument("--progress", action="store_true")
    args = parser.parse_args()
    print(json.dumps(run_experiment(
        epochs=args.epochs, batch_size=args.batch_size,
        ewc_lambda=args.ewc_lambda, max_tasks=args.max_tasks,
        progress=args.progress, fisher_samples=args.fisher_samples,
        fisher_method=args.fisher_method, scenario=args.scenario,
        replay_per_task=args.replay, joint=args.joint,
    ), indent=2))
