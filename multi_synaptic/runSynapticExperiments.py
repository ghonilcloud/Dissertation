"""Run ANN-MSCN and SNN-MSCN over the EWC condition grid and a mask-density sweep."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from baseline.runBaselineExperiments import summarize_model_results
from multi_synaptic.annSynaptic import run_experiment as run_ann_mscn
from multi_synaptic.snnSynaptic import run_experiment as run_snn_mscn

# Random seeds; keep the same as the EWC/baseline seeds you compare against.
SEEDS = (
    42,
    123,
    456,
)

# Fraction of each layer's weights a task's mask selects (paper's c).
# Every condition below runs once per density.
MASK_DENSITIES = (
    # 0.1, 
    # 0.3, 
    0.5)

# Which MSCN parts are switched on (ablation). Every condition runs once per variant.
# synapses = parallel synapses per connection (paper's P); modulation = stage (c).
MSCN_VARIANTS = (
    {"synapses": 1, "modulation": True},  # stage (a): task masks only
    {"synapses": 3, "modulation": True},  # stage (a) + (b): parallel synapses
    {"synapses": 3, "modulation": True},   # stage (a) + (b) + (c): full MSCN
)

# Same training conditions as ewc/runEwcExperiments.py. Task mode only.
MSCN_EXPERIMENT_CONDITIONS = (
    {"epochs": 1, "time_steps": 2},
    {"epochs": 2, "time_steps": 4},
    {"epochs": 5, "time_steps": 7},
)


def run_grid(
    *,
    data_dir: str | Path = "data",
    batch_size: int = 128,
    learning_rate: float = 1e-3,
    num_workers: int = 0,
    max_tasks: int | None = None,
    progress: bool = True,
    output_path: str | Path = "results/mscn_experiments.json",
) -> dict[str, Any]:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    results: dict[str, Any] = {"seeds": list(SEEDS), "conditions": []}
    runs = [
        (c, d, v) for c in MSCN_EXPERIMENT_CONDITIONS for d in MASK_DENSITIES for v in MSCN_VARIANTS
    ]
    for number, (condition, density, variant) in enumerate(runs, 1):
        print(
            f"Condition {number}/{len(runs)}: {condition['epochs']} epochs, "
            f"{condition['time_steps']} SNN time steps, scenario=task, "
            f"mask_density={density}, synapses={variant['synapses']}, "
            f"modulation={variant['modulation']}"
        )
        common = dict(
            data_dir=data_dir, batch_size=batch_size, epochs=condition["epochs"],
            learning_rate=learning_rate, mask_density=density, **variant,
            num_workers=num_workers, max_tasks=max_tasks, progress=progress, seeds=SEEDS,
        )
        ann = run_ann_mscn(**common)
        snn = run_snn_mscn(time_steps=condition["time_steps"], **common)
        results["conditions"].append({
            "epochs": condition["epochs"],
            "snn_time_steps": condition["time_steps"],
            "scenario": "task",
            "mask_density": density,
            **variant,
            "ann": summarize_model_results(ann),
            "snn": summarize_model_results(snn),
        })
        output.write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"Saved completed condition to {output}")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-tasks", type=int, default=None)
    parser.add_argument("--output", default="results/mscn_experiments.json")
    parser.add_argument("--no-progress", action="store_true")
    args = parser.parse_args()
    run_grid(
        data_dir=args.data_dir, batch_size=args.batch_size, learning_rate=args.learning_rate,
        num_workers=args.num_workers, max_tasks=args.max_tasks,
        progress=not args.no_progress, output_path=args.output,
    )
