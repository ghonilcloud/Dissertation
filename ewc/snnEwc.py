"""SpikingJelly SNN with diagonal-Fisher Elastic Weight Consolidation."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Sequence

import utilities.windowsTorchRuntime  # noqa: F401
import torch
from torch import nn
from torch.utils.data import DataLoader

from ewc.annEwc import EWC, Replay
from baseline.baselineAnn import DEFAULT_SEEDS, evaluate, set_seed
from baseline.baselineSnn import BaselineSNN, TaskIncrementalSNN
from utilities.dataset import get_cifar_datasets
from utilities.preprocessing import get_cifar_transforms
from utilities.taskList import CIFAR10_CLASSES, CIFAR10_TASKS, Task, build_task_loaders


def run_experiment(
    *,
    data_dir: str | Path = "data",
    batch_size: int = 128,
    epochs: int = 5,
    learning_rate: float = 1e-3,
    ewc_lambda: float = 100.0,
    time_steps: int = 10,
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
    """Train a SpikingJelly SNN sequentially with EWC."""
    if scenario not in {"class", "task"}:
        raise ValueError("scenario must be 'class' or 'task'")
    if max_tasks is not None and not 1 <= max_tasks <= len(CIFAR10_TASKS):
        raise ValueError(f"max_tasks must be between 1 and {len(CIFAR10_TASKS)}")
    run_seeds = seeds if seed == 42 else (seed,)
    results = []
    for run_seed in run_seeds:
        set_seed(run_seed)
        if progress:
            print(f"[{time.strftime('%H:%M:%S')}] SNN-EWC seed {run_seed} starting")
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
            train_dataset, tasks, batch_size=batch_size, num_workers=num_workers,
            remap_labels=scenario == "task",
        )
        test_loaders = build_task_loaders(
            test_dataset, tasks, batch_size=batch_size, num_workers=num_workers,
            remap_labels=scenario == "task",
        )
        full_test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
        )
        model = (
            BaselineSNN(time_steps=time_steps)
            if scenario == "class"
            else TaskIncrementalSNN(num_tasks=len(tasks), time_steps=time_steps)
        ).to(device)
        criterion = nn.CrossEntropyLoss()
        ewc = EWC(fisher_samples=fisher_samples, fisher_method=fisher_method)
        replay = Replay(None if joint else replay_per_task, batch_size)
        task_results = []
        seen_classes: list[int] = []
        best_accuracies: list[float] = []
        total_training_time = 0.0

        for task, train_loader in zip(tasks, train_loaders):
            if scenario == "task":
                model.set_task(task.number - 1)
            # Fresh optimizer per task, matching the baselines and ANN-EWC.
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
            losses = []
            start = time.perf_counter()
            for epoch in range(1, epochs + 1):
                model.train()
                total_loss = 0.0
                batches = 0
                for inputs, labels in train_loader:
                    optimizer.zero_grad()
                    loss = criterion(model(inputs.to(device)), labels.to(device))
                    loss = replay.mix(
                        loss, model, criterion, device, task.number - 1, scenario
                    )
                    loss = loss + 0.5 * ewc_lambda * ewc.penalty(model)
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()
                    batches += 1
                losses.append(total_loss / batches)
                if progress:
                    print(
                        f"[{time.strftime('%H:%M:%S')}] Task {task.number}, epoch {epoch}/{epochs}, "
                        f"loss: {losses[-1]:.4f}"
                    )
            elapsed = time.perf_counter() - start
            total_training_time += elapsed
            if ewc_lambda > 0:
                ewc.consolidate(model, train_loader, device)
            replay.add(train_loader, task.number - 1)
            if progress:
                print(f"[{time.strftime('%H:%M:%S')}] Task {task.number} done ({elapsed:.1f}s training)")
            seen_classes.extend(task.classes)
            accuracies = []
            for index in range(task.number):
                if scenario == "task":
                    model.set_task(index)
                    accuracies.append(evaluate(model, test_loaders[index], device))
                else:
                    accuracies.append(
                        evaluate(
                            model, test_loaders[index], device, tasks[index].classes
                        )
                    )
            if scenario == "task":
                model.set_task(task.number - 1)
            forgetting = {
                f"task_{index + 1}": best_accuracies[index] - accuracies[index]
                for index in range(task.number - 1)
            }
            best_accuracies = [
                max(previous, current)
                for previous, current in zip(
                    best_accuracies + [0.0], accuracies
                )
            ]
            task_results.append(
                {
                    "task": task.number,
                    "classes": [
                        CIFAR10_CLASSES[index] for index in task.classes
                    ],
                    "current_task_accuracy": accuracies[-1],
                    "seen_accuracy": (
                        sum(accuracies) / len(accuracies)
                        if scenario == "task"
                        else evaluate(model, full_test_loader, device, seen_classes)
                    ),
                    "previous_task_accuracy": {
                        f"task_{index + 1}": accuracies[index]
                        for index in range(task.number - 1)
                    },
                    "average_accuracy": sum(accuracies) / len(accuracies),
                    "forgetting": forgetting,
                    "training_loss": losses,
                    "training_time_seconds": elapsed,
                    "task_accuracies": accuracies,
                }
            )
        results.append(
            {
                "seed": run_seed,
                "scenario": scenario,
                "ewc_lambda": ewc_lambda,
                "replay_per_task": "all" if joint else replay_per_task,
                "tasks": task_results,
                "final_accuracy": task_results[-1]["average_accuracy"],
                "final_seen_accuracy": task_results[-1]["seen_accuracy"],
                "total_training_time_seconds": total_training_time,
            }
        )
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--time-steps", type=int, default=10)
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
    print(
        json.dumps(
            run_experiment(
                epochs=args.epochs,
                time_steps=args.time_steps,
                batch_size=args.batch_size,
                ewc_lambda=args.ewc_lambda,
                max_tasks=args.max_tasks,
                progress=args.progress,
                fisher_samples=args.fisher_samples,
                fisher_method=args.fisher_method,
                scenario=args.scenario,
                replay_per_task=args.replay,
                joint=args.joint,
            ),
            indent=2,
        )
    )
