"""Multi-Synaptic Cooperation Network (MSCN, Li et al., ICLR 2026), SNN version.

Same masks and task loop as annSynaptic.py; only the neurons and synapses differ:
LIF neurons (SpikingJelly, sigmoid surrogate gradient), direct input encoding,
hidden spikes averaged over the time steps, as in the baseline SNN.

Stage (b), SNN version (paper Sec. 3.1, Eq. 2-4): each of the P parallel synapses
from input i filters the incoming spikes with its own decay time constant tau_ip
(fixed, non-trainable), and the filtered signals are weighted and summed:
    PSP_ip[t] = r_ip * PSP_ip[t-1] + S_i[t],   r_ip = exp(-1 / tau_ip)
    I_j[t]    = sum_i sum_p w^p_ij * PSP_ip[t]
The paper gives no tau values. Following HetSyn (Deng et al., NeurIPS 2025), which
the paper cites for this, tau_ip is drawn from a uniform range, expressed here in
time steps instead of milliseconds (TAU_RANGE).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import utilities.windowsTorchRuntime  # noqa: F401
import torch
from spikingjelly.activation_based import functional, neuron, surrogate
from torch import nn

from baseline.baselineAnn import DEFAULT_SEEDS
from multi_synaptic.annSynaptic import MODULATION, MaskedLinear, SynapticANN, f_mod, run_tasks

# Synaptic decay time constants, in time steps, sampled per (synapse, input neuron).
TAU_RANGE = (1.0, 10.0)


class SpikingMaskedLinear(MaskedLinear):
    """Masked layer whose parallel synapses each filter spikes with their own decay."""

    def __init__(self, in_features: int, out_features: int, density: float, synapses: int = 1) -> None:
        super().__init__(in_features, out_features, density, synapses)
        tau = torch.empty(synapses, in_features).uniform_(*TAU_RANGE)
        self.register_buffer("decay", torch.exp(-1.0 / tau))  # non-trainable, as in the paper
        self.psp: torch.Tensor | None = None
        # Stage (c): SNN eligibility trace follows the spikes arriving at each
        # connection (Eq. 5-6), so one trace per input neuron, shared by its P synapses.
        self.trace = torch.zeros(in_features)

    def prepare(self) -> None:
        super().prepare()
        self.psp = None  # new input sample: clear the synaptic traces

    def modulation(self, scale: torch.Tensor) -> torch.Tensor:
        return f_mod(self.trace / scale).abs()[None, None, :]

    def forward(self, spikes: torch.Tensor) -> torch.Tensor:
        self.record(spikes)  # averaged over time steps and batch in update_trace
        weight = self.weight * self.mask
        if self.synapses == 1:  # P = 1: a standard synapse, identical to stage (a)
            return nn.functional.linear(spikes, weight[0])
        if self.psp is None:
            self.psp = spikes.new_zeros(spikes.shape[0], self.synapses, spikes.shape[1])
        # Paper's kernel (Eq. 3-4): each spike adds 1, then decays by r per step.
        self.psp = self.decay * self.psp + spikes[:, None, :]
        return torch.einsum("bpi,poi->bo", self.psp, weight)


class SynapticSNN(SynapticANN):
    """Masked 3072-512-256 LIF network with one 2-class head per task."""

    layer_type = SpikingMaskedLinear

    def __init__(
        self, num_tasks: int = 5, mask_density: float = 0.5, synapses: int = 1,
        time_steps: int = 7, modulation: bool = False,
    ) -> None:
        super().__init__(num_tasks, mask_density, synapses, modulation)
        if time_steps <= 0:
            raise ValueError("time_steps must be greater than zero")
        self.time_steps = time_steps
        self.lifs = nn.ModuleList([
            neuron.LIFNode(surrogate_function=surrogate.Sigmoid()) for _ in self.layers
        ])

    def features(self, inputs: torch.Tensor) -> torch.Tensor:
        functional.reset_net(self)
        inputs = inputs.flatten(1)
        total = 0.0
        for _ in range(self.time_steps):
            spikes = inputs
            for layer, lif in zip(self.layers, self.lifs):
                spikes = lif(layer(spikes))
            total = total + spikes
        return total / self.time_steps


def run_experiment(
    *,
    data_dir: str | Path = "data",
    batch_size: int = 128,
    epochs: int = 5,
    time_steps: int = 7,
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
        lambda num_tasks: SynapticSNN(num_tasks, mask_density, synapses, time_steps, modulation),
        name="SNN-MSCN", data_dir=data_dir, batch_size=batch_size, epochs=epochs,
        learning_rate=learning_rate, num_workers=num_workers, download=download,
        max_tasks=max_tasks, progress=progress, seeds=seeds,
        extra={
            "mask_density": mask_density, "synapses": synapses, "time_steps": time_steps,
            "tau_range_steps": list(TAU_RANGE),
            "modulation": MODULATION if modulation else None,
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--time-steps", type=int, default=7)
    parser.add_argument("--mask-density", type=float, default=0.5)
    parser.add_argument("--synapses", type=int, default=1)
    parser.add_argument("--modulation", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-tasks", type=int, default=None)
    args = parser.parse_args()
    print(json.dumps(run_experiment(
        epochs=args.epochs, time_steps=args.time_steps, mask_density=args.mask_density,
        synapses=args.synapses, modulation=args.modulation, seeds=(args.seed,),
        max_tasks=args.max_tasks, progress=True,
    ), indent=2))
