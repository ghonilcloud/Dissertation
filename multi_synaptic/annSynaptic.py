"""Multi-Synaptic Cooperation Network (MSCN, Li et al., ICLR 2026), ANN version.

Stage (a), task-specific sub-networks (paper Sec. 3.3, Eq. 10-11, following WSN,
Kang et al. 2022):
- every weight has a learnable relevance score;
- each task uses only the top-c% of weights in each layer, ranked by |score|;
- after a task, its binary mask is stored, and every weight it used is frozen
  (accumulated mask M). Later tasks may reuse frozen weights but not change them.

Stage (b), parallel synapses: P weights per connection, each with its own
transfer function (see MaskedLinear). P = 1 gives stage (a) alone.

Stage (c), plasticity modulation by eligibility traces (Sec. 3.2, Eq. 6-8; ANN
App. A.1, Eq. 15-16): a trace of recent activity sets how big each synapse's
update is, Delta w = -lr * |f_mod(e)| * dL/dw. Settings in MODULATION (below).

Task mode only: the task ID picks both the output head and the stored mask.
Old tasks cannot be forgotten by construction (forgetting / BWT = 0).
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Callable, Sequence

import utilities.windowsTorchRuntime  # noqa: F401
import torch
import torch.nn.functional as F
from torch import nn

from baseline.baselineAnn import DEFAULT_SEEDS, evaluate, set_seed
from utilities.dataset import get_cifar_datasets
from utilities.preprocessing import get_cifar_transforms
from utilities.taskList import CIFAR10_CLASSES, CIFAR10_TASKS, Task, build_task_loaders


# Stage (c) settings. MSCN gives no values:
# - theta_max (maximum boost) and e_inv (activity where learning stops) are the
#   values of NACA (Zhang et al., 2023, Science Advances), the paper MSCN takes
#   f_mod from; e1 = 0.5 * e_inv as in MSCN Eq. 7;
# - tau (trace decay, in training batches) follows SA-SNN (Shen et al., 2024),
#   which uses the same trace equation.
MODULATION = {"theta_max": 1.2, "e_inv": 0.5, "tau": 10.0}


def f_mod(e: torch.Tensor, theta_max: float = MODULATION["theta_max"], e_inv: float = MODULATION["e_inv"]) -> torch.Tensor:
    """MSCN Eq. 7 on a trace normalised to [-1, 1]: rises from 1 to theta_max at e1,
    falls to 0 at e_inv, mirrors negatively up to 2 * e_inv, and is 0 beyond."""
    e1 = 0.5 * e_inv
    a = e.abs()
    rise = 1 + (theta_max - 1) * (a / e1) ** 2
    fall = lambda x: theta_max * (1 - ((x - e1) / (e_inv - e1)) ** 2)  # noqa: E731
    low = torch.where(a <= e1, rise, fall(a))
    mirror = 2 * e_inv - a
    high = -torch.where(mirror <= e1, 1 + (theta_max - 1) * (mirror / e1) ** 2, fall(mirror))
    return torch.where(a <= e_inv, low, torch.where(a <= 2 * e_inv, high, torch.zeros_like(a)))


class _TopK(torch.autograd.Function):
    """Binary mask of the top-k |scores|; gradient passes straight through."""

    @staticmethod
    def forward(ctx, scores: torch.Tensor, density: float) -> torch.Tensor:
        k = max(1, int(density * scores.numel()))
        mask = torch.zeros_like(scores)
        mask.view(-1)[scores.abs().flatten().topk(k).indices] = 1.0
        return mask

    @staticmethod
    def backward(ctx, grad: torch.Tensor):
        return grad, None


class MaskedLinear(nn.Module):
    """Linear layer (no bias) with P parallel synapses per connection, each synapse
    gated by a per-task binary mask.

    Stage (b), ANN version (paper App. A.1, Eq. 12-14): z_j = sum_i sum_p sigma_p(w^p_ij x_i).
    The paper does not name sigma_p. We use identity, ReLU and |.|, because each
    one splits into ordinary matrix products, so no batch x in x out tensor is built:
        ReLU(w x) = ReLU(w) ReLU(x) + ReLU(-w) ReLU(-x),   |w x| = |w| |x|.
    Different functions per synapse are needed: purely linear parallel synapses
    collapse into one weight (Song & Benna, 2025). P = 1 is a standard layer.
    """

    def __init__(self, in_features: int, out_features: int, density: float, synapses: int = 1) -> None:
        super().__init__()
        if not 0.0 < density <= 1.0:
            raise ValueError("mask density must be in (0, 1]")
        if not 1 <= synapses <= 3:
            raise ValueError("synapses per connection must be 1, 2 or 3")
        self.density = density
        self.synapses = synapses
        # ponytail: no bias, as in WSN; a shared bias would change on every task
        # and break the zero-forgetting guarantee.
        self.weight = nn.Parameter(torch.empty(synapses, out_features, in_features))
        self.scores = nn.Parameter(torch.empty_like(self.weight))
        for weight, score in zip(self.weight, self.scores):
            nn.init.kaiming_uniform_(weight, a=math.sqrt(5))
            nn.init.kaiming_uniform_(score, a=math.sqrt(5))
        with torch.no_grad():
            self.weight /= math.sqrt(synapses)  # keep the summed input at the P = 1 scale
        # ReLU and |.| synapses only add positive values, so summed over all inputs
        # they drift ~sqrt(in) times more than the identity synapse. Scaling them in
        # the forward pass (not at init) keeps Adam's fixed-size steps proportionate.
        self.nonlinear_gain = 1.0 / math.sqrt(in_features)
        self.register_buffer("frozen", torch.zeros_like(self.weight))  # accumulated mask M
        self.task_masks: list[torch.Tensor] = []
        self.task_id = 0
        self.mask: torch.Tensor | None = None
        # Stage (c): ANN trace follows each neuron's input z_j (Eq. 15), shared by
        # all of that neuron's incoming synapses. Kept across tasks.
        self.register_buffer("trace", torch.zeros(out_features))
        self.activity: torch.Tensor | None = None
        self.activity_count = 0

    def prepare(self) -> None:
        """Pick this forward pass's mask: stored for learned tasks, top-k while learning."""
        if self.task_id < len(self.task_masks):
            self.mask = self.task_masks[self.task_id]
        else:
            self.mask = _TopK.apply(self.scores, self.density)
        self.activity, self.activity_count = None, 0

    def record(self, signal: torch.Tensor) -> None:
        """Add this forward pass's batch-mean activity (training only)."""
        if self.training:
            mean = signal.detach().mean(0)
            self.activity = mean if self.activity is None else self.activity + mean
            self.activity_count += 1

    def update_trace(self) -> None:
        """e[t+1] = e[t] - e[t] / tau + activity, once per training batch (Eq. 6 / 15)."""
        self.trace += -self.trace / MODULATION["tau"] + self.activity / self.activity_count

    def modulation(self, scale: torch.Tensor) -> torch.Tensor:
        """|f_mod| of the normalised trace, shaped to broadcast over (P, out, in)."""
        return f_mod(self.trace / scale).abs()[None, :, None]

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        weight = self.weight * self.mask
        out = F.linear(inputs, weight[0])  # synapse 1: identity
        if self.synapses > 1:  # synapse 2: ReLU(w x)
            w = weight[1] * self.nonlinear_gain
            out = out + F.linear(F.relu(inputs), F.relu(w)) + F.linear(F.relu(-inputs), F.relu(-w))
        if self.synapses > 2:  # synapse 3: |w x|
            out = out + F.linear(inputs.abs(), (weight[2] * self.nonlinear_gain).abs())
        self.record(out)
        return out

    def consolidate(self) -> None:
        """Store the current task's mask and freeze the weights it used (Eq. 11)."""
        mask = _TopK.apply(self.scores.detach(), self.density)
        self.task_masks.append(mask)
        self.frozen = torch.maximum(self.frozen, mask)

    def protect_old_weights(self) -> None:
        """Zero gradients of frozen weights before the optimiser step."""
        if self.weight.grad is not None:
            self.weight.grad.mul_(1.0 - self.frozen)


class SynapticANN(nn.Module):
    """3072-512-256 ReLU MLP with masked layers and one 2-class head per task."""

    layer_type = MaskedLinear

    def __init__(
        self, num_tasks: int = 5, mask_density: float = 0.5, synapses: int = 1, modulation: bool = False,
    ) -> None:
        super().__init__()
        self.modulate = modulation
        self.layers = nn.ModuleList([
            self.layer_type(3 * 32 * 32, 512, mask_density, synapses),
            self.layer_type(512, 256, mask_density, synapses),
        ])
        self.heads = nn.ModuleList([nn.Linear(256, 2) for _ in range(num_tasks)])
        self.task_id = 0

    def set_task(self, task_id: int) -> None:
        if not 0 <= task_id < len(self.heads):
            raise ValueError("task_id is outside the configured task heads")
        self.task_id = task_id
        for layer in self.layers:
            layer.task_id = task_id

    def features(self, inputs: torch.Tensor) -> torch.Tensor:
        hidden = inputs.flatten(1)
        for layer in self.layers:
            hidden = F.relu(layer(hidden))
        return hidden

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            layer.prepare()  # once per forward, not once per SNN time step
        return self.heads[self.task_id](self.features(inputs))

    def consolidate(self) -> None:
        for layer in self.layers:
            layer.consolidate()

    def protect_old_weights(self) -> None:
        for layer in self.layers:
            layer.protect_old_weights()

    def step(self, optimizer: torch.optim.Optimizer) -> None:
        """Optimiser step; with stage (c), each weight's update is scaled by |f_mod|.

        Adam divides each gradient by its own running size, so scaling the gradient
        would be almost cancelled out. Scaling the update Adam makes gives the
        paper's Delta w = -lr * |f_mod(e)| * dL/dw (Eq. 8 / 16).
        """
        if not self.modulate:
            optimizer.step()
            return
        before = [layer.weight.detach().clone() for layer in self.layers]
        optimizer.step()
        for layer in self.layers:
            layer.update_trace()
        # Normalise the trace over all model synapses to [-1, 1] (Sec. 3.2).
        scale = torch.stack([layer.trace.abs().max() for layer in self.layers]).max().clamp_min(1e-12)
        with torch.no_grad():
            for layer, old in zip(self.layers, before):
                layer.weight.copy_(old + layer.modulation(scale) * (layer.weight - old))

    def capacity_used(self) -> list[float]:
        """Fraction of each layer's weights frozen by learned tasks."""
        return [float(layer.frozen.mean()) for layer in self.layers]


def train_task(
    model: SynapticANN,
    loader,
    *,
    device: torch.device,
    epochs: int,
    learning_rate: float,
    task_number: int,
    progress: bool,
) -> tuple[list[float], float]:
    """Train weights and relevance scores on one task; frozen weights stay fixed."""
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    losses = []
    start = time.perf_counter()
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss, batches = 0.0, 0
        for inputs, labels in loader:
            optimizer.zero_grad()
            loss = criterion(model(inputs.to(device)), labels.to(device))
            loss.backward()
            model.protect_old_weights()
            model.step(optimizer)
            total_loss += loss.item()
            batches += 1
        losses.append(total_loss / batches)
        if progress:
            print(f"[{time.strftime('%H:%M:%S')}] Task {task_number}, epoch {epoch}/{epochs}, loss: {losses[-1]:.4f}")
    return losses, time.perf_counter() - start


def run_tasks(
    build_model: Callable[[int], SynapticANN],
    *,
    name: str,
    data_dir: str | Path,
    batch_size: int,
    epochs: int,
    learning_rate: float,
    num_workers: int,
    download: bool,
    max_tasks: int | None,
    progress: bool,
    seeds: Sequence[int],
    extra: dict[str, object],
) -> list[dict[str, object]]:
    """Task-incremental Split CIFAR-10 loop shared by the ANN and SNN versions."""
    if max_tasks is not None and not 1 <= max_tasks <= len(CIFAR10_TASKS):
        raise ValueError(f"max_tasks must be between 1 and {len(CIFAR10_TASKS)}")
    all_results = []
    for run_seed in seeds:
        set_seed(run_seed)
        if progress:
            print(f"[{time.strftime('%H:%M:%S')}] {name} seed {run_seed} starting")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        train_transform, test_transform = get_cifar_transforms("cifar10")
        train_dataset, test_dataset = get_cifar_datasets(
            "cifar10", data_dir, download=download,
            train_transform=train_transform, test_transform=test_transform,
        )
        tasks = [Task(i + 1, classes) for i, classes in enumerate(CIFAR10_TASKS[:max_tasks])]
        train_loaders = build_task_loaders(
            train_dataset, tasks, batch_size=batch_size, num_workers=num_workers, remap_labels=True,
        )
        test_loaders = build_task_loaders(
            test_dataset, tasks, batch_size=batch_size, num_workers=num_workers, remap_labels=True,
        )
        model = build_model(len(tasks)).to(device)
        task_results, best, total_time = [], [], 0.0
        for task, train_loader in zip(tasks, train_loaders):
            model.set_task(task.number - 1)
            losses, elapsed = train_task(
                model, train_loader, device=device, epochs=epochs,
                learning_rate=learning_rate, task_number=task.number, progress=progress,
            )
            total_time += elapsed
            model.consolidate()
            if progress:
                print(f"[{time.strftime('%H:%M:%S')}] Task {task.number} done ({elapsed:.1f}s training)")
            accuracies = []
            for i in range(task.number):
                model.set_task(i)
                accuracies.append(evaluate(model, test_loaders[i], device))
            forgetting = {f"task_{i + 1}": best[i] - accuracies[i] for i in range(task.number - 1)}
            best = [max(old, current) for old, current in zip(best + [0.0], accuracies)]
            task_results.append({
                "task": task.number,
                "classes": [CIFAR10_CLASSES[i] for i in task.classes],
                "current_task_accuracy": accuracies[-1],
                "seen_accuracy": sum(accuracies) / len(accuracies),
                "previous_task_accuracy": {f"task_{i + 1}": accuracies[i] for i in range(task.number - 1)},
                "average_accuracy": sum(accuracies) / len(accuracies),
                "forgetting": forgetting,
                "capacity_used": model.capacity_used(),
                "training_loss": losses,
                "training_time_seconds": elapsed,
                "task_accuracies": accuracies,
            })
        all_results.append({
            "seed": run_seed, "scenario": "task", **extra,
            "tasks": task_results,
            "final_accuracy": task_results[-1]["average_accuracy"],
            "final_seen_accuracy": task_results[-1]["seen_accuracy"],
            "total_training_time_seconds": total_time,
        })
    return all_results


def run_experiment(
    *,
    data_dir: str | Path = "data",
    batch_size: int = 128,
    epochs: int = 5,
    learning_rate: float = 1e-3,
    mask_density: float = 0.5,
    synapses: int = 1,
    modulation: bool = False,
    num_workers: int = 0,
    download: bool = False,
    max_tasks: int | None = None,
    progress: bool = False,
    seeds: Sequence[int] = DEFAULT_SEEDS,
) -> list[dict[str, object]]:
    return run_tasks(
        lambda num_tasks: SynapticANN(num_tasks, mask_density, synapses, modulation),
        name="ANN-MSCN", data_dir=data_dir, batch_size=batch_size, epochs=epochs,
        learning_rate=learning_rate, num_workers=num_workers, download=download,
        max_tasks=max_tasks, progress=progress, seeds=seeds,
        extra={
            "mask_density": mask_density, "synapses": synapses,
            "modulation": MODULATION if modulation else None,
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--mask-density", type=float, default=0.5)
    parser.add_argument("--synapses", type=int, default=1)
    parser.add_argument("--modulation", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-tasks", type=int, default=None)
    args = parser.parse_args()
    print(json.dumps(run_experiment(
        epochs=args.epochs, mask_density=args.mask_density, synapses=args.synapses,
        modulation=args.modulation, seeds=(args.seed,),
        max_tasks=args.max_tasks, progress=True,
    ), indent=2))
