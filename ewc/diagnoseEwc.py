"""Run a short two-task EWC diagnostic and save loss/gradient measurements."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import utilities.windowsTorchRuntime  # noqa: F401
import torch
from torch import nn

from ewc.annEwc import EWC
from baseline.baselineAnn import BaselineANN, evaluate, set_seed
from baseline.baselineSnn import BaselineSNN
from utilities.dataset import get_cifar_datasets
from utilities.preprocessing import get_cifar_transforms
from utilities.taskList import CIFAR10_TASKS, Task, build_task_loaders


def norm(tensors: list[torch.Tensor]) -> float:
    return float(torch.sqrt(sum(t.detach().square().sum() for t in tensors))) if tensors else 0.0


def fisher_summary(ewc: EWC) -> dict[str, object]:
    result = []
    for index, (_, fisher) in enumerate(ewc.snapshots, 1):
        layers = {}
        for name, tensor in fisher.items():
            layers[name] = {
                "mean": float(tensor.mean()),
                "max": float(tensor.max()),
                "sum": float(tensor.sum()),
            }
        all_values = torch.cat([x.detach().reshape(-1).cpu() for x in fisher.values()])
        result.append({
            "task_snapshot": index,
            "sample_count": ewc.sample_counts[index - 1],
            "overall_mean": float(all_values.mean()),
            "overall_max": float(all_values.max()),
            "overall_sum": float(all_values.sum()),
            "layers": layers,
        })
    return {"snapshots": result}


def run_model(kind: str, train_loader, test_loader, seed: int, ewc_lambda: float, device: torch.device) -> dict[str, object]:
    set_seed(seed)
    model = BaselineANN() if kind == "ann" else BaselineSNN(time_steps=2)
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    fisher_samples = 128
    ewc = EWC(fisher_samples=fisher_samples)
    learning_rate = 1e-3
    losses_by_task = []
    task1_accuracy = None
    task2_diagnostic = {}

    for task_number, loader in enumerate((train_loader[0], train_loader[1]), 1):
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        epoch_loss = 0.0
        batches = 0
        if task_number == 2:
            anchor = {n: p.detach().clone() for n, p in model.named_parameters()}
            task2_diagnostic["fisher_before_task_2"] = fisher_summary(ewc)
            task2_diagnostic["fisher_sample_limit"] = fisher_samples
            task2_diagnostic["ewc_checks"] = []
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(inputs)
            ce_loss = criterion(logits, labels)
            penalty = ewc.penalty(model)
            if task_number == 2 and batches in (0, 1, 2, 9, 49, 99):
                ce_grads = torch.autograd.grad(ce_loss, tuple(model.parameters()), retain_graph=True, allow_unused=True)
                penalty_grads = torch.autograd.grad(0.5 * ewc_lambda * penalty, tuple(model.parameters()), retain_graph=True, allow_unused=True)
                ce_gradient_norm = norm([g for g in ce_grads if g is not None])
                ewc_gradient_norm = norm([g for g in penalty_grads if g is not None])
                task2_diagnostic["ewc_checks"].append({
                    "optimizer_steps_completed": batches,
                    "cross_entropy": float(ce_loss.detach()),
                    "ewc_penalty_unweighted": float(penalty.detach()),
                    "ewc_penalty_weighted": float((0.5 * ewc_lambda * penalty).detach()),
                    "cross_entropy_gradient_norm": ce_gradient_norm,
                    "weighted_ewc_gradient_norm": ewc_gradient_norm,
                    "weighted_ewc_to_ce_gradient_norm_ratio": ewc_gradient_norm / max(ce_gradient_norm, 1e-30),
                })
            loss = ce_loss + 0.5 * ewc_lambda * penalty
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.detach())
            batches += 1
        losses_by_task.append(epoch_loss / max(batches, 1))
        if task_number == 1:
            task1_accuracy = evaluate(model, test_loader[0], device, CIFAR10_TASKS[0])
            ewc.consolidate(model, loader, device)
        else:
            task2_diagnostic["parameter_drift_after_task_2"] = {
                "l2_norm": norm([p.detach() - anchor[n] for n, p in model.named_parameters()]),
                "relative_l2_norm": norm([p.detach() - anchor[n] for n, p in model.named_parameters()]) / max(norm([anchor[n] for n, _ in model.named_parameters()]), 1e-30),
            }
    task2_accuracy = evaluate(model, test_loader[0], device, CIFAR10_TASKS[0])
    task2_diagnostic["task_1_accuracy_after_task_1"] = task1_accuracy
    task2_diagnostic["task_1_accuracy_after_task_2"] = task2_accuracy
    task2_diagnostic["task_1_accuracy_drop"] = task1_accuracy - task2_accuracy
    task2_diagnostic["task_2_training_loss"] = losses_by_task[1]
    return task2_diagnostic


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ewc-lambda", type=float, default=100.0)
    parser.add_argument(
        "--output", default="results/ewc_diagnostics_per_example.json"
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_transform, test_transform = get_cifar_transforms("cifar10")
    train_dataset, test_dataset = get_cifar_datasets(
        "cifar10", args.data_dir, download=False,
        train_transform=train_transform, test_transform=test_transform,
    )
    tasks = [Task(i + 1, classes) for i, classes in enumerate(CIFAR10_TASKS[:2])]
    train_loaders = build_task_loaders(train_dataset, tasks, batch_size=128)
    test_loaders = build_task_loaders(test_dataset, tasks, batch_size=128)
    results = {
        "seed": args.seed,
        "epochs_per_task": 1,
        "tasks": [[int(x) for x in t.classes] for t in tasks],
        "ewc_lambda": args.ewc_lambda,
        "device": str(device),
        "ann": run_model("ann", train_loaders, test_loaders, args.seed, args.ewc_lambda, device),
        "snn": run_model("snn", train_loaders, test_loaders, args.seed, args.ewc_lambda, device),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))
    print(f"Saved diagnostics to {output}")


if __name__ == "__main__":
    main()
