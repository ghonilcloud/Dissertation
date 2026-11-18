"""Run ANN-EWC and SNN-EWC across the baseline condition grid."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from baseline.baselineAnn import DEFAULT_SEEDS
from ewc.annEwc import run_experiment as run_ann_ewc
from ewc.snnEwc import run_experiment as run_snn_ewc
from baseline.runBaselineExperiments import summarize_model_results

# 1 epoch is the existing reference. 5 gives an intermediate budget; 10 epochs
# is about 790 optimizer updates per 10,000-example CIFAR pair at batch size 128,
# close to the supplied TensorFlow notebook's 800 updates per task.
EWC_EXPERIMENT_CONDITIONS = tuple(
    {"epochs": epochs, "time_steps": 2} for epochs in (1, 5, 10)
)


def run_grid(
    *,
    data_dir: str | Path = "data",
    batch_size: int = 128,
    learning_rate: float = 1e-3,
    ewc_lambda: float = 100.0,
    num_workers: int = 0,
    max_tasks: int | None = None,
    progress: bool = True,
    output_path: str | Path = "results/ewc_experiments.json",
    fisher_samples: int = 1024,
    fisher_method: str = "empirical",
    scenario: str = "class",
    replay_per_task: int = 0,
    joint: bool = False,
) -> dict[str, Any]:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    results: dict[str, Any] = {"seeds": list(DEFAULT_SEEDS), "conditions": []}
    for number, condition in enumerate(EWC_EXPERIMENT_CONDITIONS, 1):
        print(
            f"Condition {number}/{len(EWC_EXPERIMENT_CONDITIONS)}: "
            f"{condition['epochs']} epochs/task, scenario={scenario}"
        )
        ann = run_ann_ewc(
            data_dir=data_dir, batch_size=batch_size, epochs=condition["epochs"],
            learning_rate=learning_rate, ewc_lambda=ewc_lambda,
            num_workers=num_workers, max_tasks=max_tasks, progress=progress,
            seeds=DEFAULT_SEEDS,
            fisher_samples=fisher_samples, fisher_method=fisher_method,
            scenario=scenario, replay_per_task=replay_per_task, joint=joint,
        )
        snn = run_snn_ewc(
            data_dir=data_dir, batch_size=batch_size, epochs=condition["epochs"],
            time_steps=condition["time_steps"], learning_rate=learning_rate,
            ewc_lambda=ewc_lambda, num_workers=num_workers, max_tasks=max_tasks,
            progress=progress, seeds=DEFAULT_SEEDS,
            fisher_samples=fisher_samples, fisher_method=fisher_method,
            scenario=scenario, replay_per_task=replay_per_task, joint=joint,
        )
        results["conditions"].append({
            "epochs": condition["epochs"],
            "snn_time_steps": condition["time_steps"],
            "ewc_lambda": ewc_lambda,
            "fisher_samples": fisher_samples,
            "fisher_method": fisher_method,
            "scenario": scenario,
            "replay_per_task": "all" if joint else replay_per_task,
            "ann_ewc": summarize_model_results(ann),
            "snn_ewc": summarize_model_results(snn),
        })
        output.write_text(json.dumps(results, indent=2), encoding="utf-8")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--ewc-lambda", type=float, default=100.0)
    parser.add_argument("--fisher-samples", type=int, default=1024)
    parser.add_argument("--fisher-method", choices=("empirical", "sampled"), default="empirical")
    parser.add_argument("--scenario", choices=("class", "task"), default="class")
    parser.add_argument("--replay", type=int, default=0)
    parser.add_argument("--joint", action="store_true")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-tasks", type=int, default=None)
    parser.add_argument("--output", default="results/ewc_experiments.json")
    parser.add_argument("--no-progress", action="store_true")
    args = parser.parse_args()
    run_grid(
        data_dir=args.data_dir, batch_size=args.batch_size,
        learning_rate=args.learning_rate, ewc_lambda=args.ewc_lambda,
        num_workers=args.num_workers, max_tasks=args.max_tasks,
        progress=not args.no_progress, output_path=args.output,
        fisher_samples=args.fisher_samples, fisher_method=args.fisher_method,
        scenario=args.scenario, replay_per_task=args.replay, joint=args.joint,
    )
