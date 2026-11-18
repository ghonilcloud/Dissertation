"""Compare EWC lambdas from one saved Task 1 checkpoint with learning curves."""

from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path
from typing import Any

import utilities.windowsTorchRuntime  # noqa: F401
import torch
from torch import nn

from ewc.annEwc import EWC
from baseline.baselineAnn import BaselineANN, TaskIncrementalANN, evaluate, set_seed
from baseline.baselineSnn import BaselineSNN, TaskIncrementalSNN
from utilities.dataset import get_cifar_datasets
from utilities.preprocessing import get_cifar_transforms
from utilities.taskList import CIFAR10_TASKS, Task, build_task_loaders


def _make_model(model_name: str, scenario: str, time_steps: int) -> nn.Module:
    if model_name == "ann":
        return BaselineANN() if scenario == "class" else TaskIncrementalANN(2)
    return BaselineSNN(time_steps=time_steps) if scenario == "class" else TaskIncrementalSNN(2, time_steps)


def _set_task(model: nn.Module, task_index: int, scenario: str) -> None:
    if scenario == "task":
        model.set_task(task_index)


def _evaluate_tasks(model, test_loaders, scenario: str, device: torch.device) -> list[float]:
    accuracies = []
    for task_index, loader in enumerate(test_loaders):
        _set_task(model, task_index, scenario)
        if scenario == "class":
            accuracies.append(
                evaluate(model, loader, device, CIFAR10_TASKS[task_index])
            )
        else:
            accuracies.append(evaluate(model, loader, device))
    return accuracies


def _train_task(
    model: nn.Module,
    loader,
    optimizer,
    ewc: EWC,
    *,
    task_index: int,
    scenario: str,
    ewc_lambda: float,
    epochs: int,
    device: torch.device,
    eval_loaders,
    eval_every: int,
    global_step: int,
) -> tuple[list[dict[str, Any]], int, float]:
    criterion = nn.CrossEntropyLoss()
    curve: list[dict[str, Any]] = []
    started = time.perf_counter()
    model.train()
    _set_task(model, task_index, scenario)
    last_ce = last_penalty = 0.0
    total_steps = epochs * len(loader)

    for _epoch in range(epochs):
        for batch_index, (inputs, labels) in enumerate(loader):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            ce_loss = criterion(model(inputs), labels)
            penalty = ewc.penalty(model)
            weighted_penalty = 0.5 * ewc_lambda * penalty
            loss = ce_loss + weighted_penalty
            loss.backward()
            optimizer.step()
            global_step += 1
            last_ce = float(ce_loss.detach())
            last_penalty = float(weighted_penalty.detach())

            should_record = global_step % eval_every == 0 or (batch_index == len(loader) - 1 and _epoch == epochs - 1)
            if should_record:
                model.eval()
                accuracies = _evaluate_tasks(model, eval_loaders, scenario, device)
                _set_task(model, task_index, scenario)
                model.train()
                curve.append({
                    "global_step": global_step,
                    "cross_entropy": last_ce,
                    "weighted_ewc_penalty": last_penalty,
                    "task_accuracies": accuracies,
                    "task_1_accuracy": accuracies[0],
                    "task_2_accuracy": accuracies[1],
                })

    return curve, global_step, time.perf_counter() - started


def run_sweep(
    *,
    data_dir: str | Path = "data",
    output_path: str | Path = "results/ewc_lambda_sweep.json",
    checkpoint_dir: str | Path = "results/ewc_checkpoints",
    model_name: str = "ann",
    scenario: str = "class",
    epochs: int = 10,
    lambdas: tuple[float, ...] = (10.0, 100.0, 1000.0),
    fisher_method: str = "sampled",
    fisher_samples: int = 512,
    seed: int = 42,
    batch_size: int = 128,
    time_steps: int = 2,
    eval_every: int = 50,
) -> dict[str, Any]:
    if model_name not in {"ann", "snn"}:
        raise ValueError("model_name must be 'ann' or 'snn'")
    if scenario not in {"class", "task"}:
        raise ValueError("scenario must be 'class' or 'task'")
    if epochs <= 0 or eval_every <= 0:
        raise ValueError("epochs and eval_every must be positive")

    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_transform, test_transform = get_cifar_transforms("cifar10")
    train_dataset, test_dataset = get_cifar_datasets(
        "cifar10", data_dir, download=False,
        train_transform=train_transform, test_transform=test_transform,
    )
    tasks = [Task(i + 1, classes) for i, classes in enumerate(CIFAR10_TASKS[:2])]
    train_loaders = build_task_loaders(
        train_dataset, tasks, batch_size=batch_size,
        remap_labels=scenario == "task",
    )
    test_loaders = build_task_loaders(
        test_dataset, tasks, batch_size=batch_size,
        remap_labels=scenario == "task",
    )

    model = _make_model(model_name, scenario, time_steps).to(device)
    ewc = EWC(fisher_samples=fisher_samples, fisher_method=fisher_method)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Train Task 1 once; each lambda then starts from these exact parameters,
    # Fisher values, and (for the SNN) optimizer state.
    task1_curve, global_step, task1_seconds = _train_task(
        model, train_loaders[0], optimizer, ewc,
        task_index=0, scenario=scenario, ewc_lambda=0.0, epochs=epochs,
        device=device, eval_loaders=test_loaders, eval_every=eval_every,
        global_step=0,
    )
    model.eval()
    task1_accuracy = _evaluate_tasks(model, test_loaders[:1], scenario, device)[0]
    ewc.consolidate(model, train_loaders[0], device)
    model.train()

    output = Path(output_path)
    checkpoint_folder = Path(checkpoint_dir)
    checkpoint_folder.mkdir(parents=True, exist_ok=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_folder / (
        f"{model_name}_{scenario}_epochs{epochs}_seed{seed}_{fisher_method}.pt"
    )
    torch.save({
        "seed": seed,
        "scenario": scenario,
        "model": model_name,
        "epochs_task_1": epochs,
        "fisher_method": fisher_method,
        "fisher_samples": ewc.sample_counts[0],
        "model_state_dict": model.state_dict(),
        "ewc_snapshots": ewc.snapshots,
        "optimizer_state_dict": optimizer.state_dict(),
    }, checkpoint_path)

    anchor_model = copy.deepcopy(model)
    anchor_ewc = copy.deepcopy(ewc)
    anchor_optimizer_state = copy.deepcopy(optimizer.state_dict())
    lambda_results = []

    for ewc_lambda in lambdas:
        candidate = copy.deepcopy(anchor_model).to(device)
        candidate_ewc = copy.deepcopy(anchor_ewc)
        candidate_optimizer = torch.optim.Adam(candidate.parameters(), lr=1e-3)
        if model_name == "snn":
            candidate_optimizer.load_state_dict(copy.deepcopy(anchor_optimizer_state))

        # Reset the RNG so every lambda sees the same shuffled Task 2 batches.
        set_seed(seed + 10_000)
        curve, final_step, task2_seconds = _train_task(
            candidate, train_loaders[1], candidate_optimizer, candidate_ewc,
            task_index=1, scenario=scenario, ewc_lambda=ewc_lambda, epochs=epochs,
            device=device, eval_loaders=test_loaders, eval_every=eval_every,
            global_step=global_step,
        )
        candidate.eval()
        final_accuracies = _evaluate_tasks(candidate, test_loaders, scenario, device)
        lambda_results.append({
            "lambda": ewc_lambda,
            "task_1_accuracy_after_task_1": task1_accuracy,
            "final_task_accuracies": final_accuracies,
            "final_average_accuracy": sum(final_accuracies) / len(final_accuracies),
            "task_2_training_seconds": task2_seconds,
            "task_2_curve": curve,
            "final_global_step": final_step,
        })

    result = {
        "seed": seed,
        "model": model_name,
        "scenario": scenario,
        "epochs_per_task": epochs,
        "optimizer_updates_per_task": global_step,
        "batch_size": batch_size,
        "ewc_lambdas": list(lambdas),
        "fisher_method": fisher_method,
        "fisher_samples": ewc.sample_counts[0],
        "task_1_training_seconds": task1_seconds,
        "task_1_curve": task1_curve,
        "task_1_accuracy": task1_accuracy,
        "task_1_checkpoint": str(checkpoint_path),
        "lambda_results": lambda_results,
    }
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--output", default="results/ewc_lambda_sweep.json")
    parser.add_argument("--checkpoint-dir", default="results/ewc_checkpoints")
    parser.add_argument("--model", choices=("ann", "snn"), default="ann")
    parser.add_argument("--scenario", choices=("class", "task"), default="class")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lambdas", type=float, nargs="+", default=[10, 100, 1000])
    parser.add_argument("--fisher-method", choices=("empirical", "sampled"), default="sampled")
    parser.add_argument("--fisher-samples", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--time-steps", type=int, default=2)
    parser.add_argument("--eval-every", type=int, default=50)
    args = parser.parse_args()
    result = run_sweep(
        data_dir=args.data_dir, output_path=args.output,
        checkpoint_dir=args.checkpoint_dir, model_name=args.model,
        scenario=args.scenario, epochs=args.epochs, lambdas=tuple(args.lambdas),
        fisher_method=args.fisher_method, fisher_samples=args.fisher_samples,
        seed=args.seed, batch_size=args.batch_size, time_steps=args.time_steps,
        eval_every=args.eval_every,
    )
    print(json.dumps(result, indent=2))
