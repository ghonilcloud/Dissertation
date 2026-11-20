"""Run ANN-EWC and SNN-EWC across the baseline condition grid."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from ewc.annEwc import run_experiment as run_ann_ewc
from ewc.snnEwc import run_experiment as run_snn_ewc
from baseline.runBaselineExperiments import summarize_model_results

# Random seeds; every condition runs ANN and SNN once per seed.
# Keep these the same as the baseline seeds you compare against.
SEEDS = (
    # 42,
    # 123, 
    # 456,
    789, 
    1011)

# Images stored per finished task when a condition has "replay": True.
REPLAY_PER_TASK = 200

EWC_EXPERIMENT_CONDITIONS = (
    {"epochs": 1, "time_steps": 2, "ewc_lambda": 100.0, "replay": False},
    {"epochs": 2, "time_steps": 4, "ewc_lambda": 100.0, "replay": False},
    # {"epochs": 5, "time_steps": 7, "ewc_lambda": 100.0, "replay": False},
)


def run_grid(
    *,
    data_dir: str | Path = "data",
    batch_size: int = 128,
    learning_rate: float = 1e-3,
    ewc_lambda: float | None = None,
    num_workers: int = 0,
    max_tasks: int | None = None,
    progress: bool = True,
    output_path: str | Path = "results/ewc_experiments.json",
    fisher_samples: int = 1024,
    fisher_method: str = "empirical",
    scenario: str = "class", # 'task' or set to "class" for 10-way
    replay_per_task: int | None = None,
    joint: bool = False,
) -> dict[str, Any]:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    results: dict[str, Any] = {"seeds": list(SEEDS), "conditions": []}
    for number, condition in enumerate(EWC_EXPERIMENT_CONDITIONS, 1):
        # --ewc-lambda on the command line overrides every condition (e.g. 0 for replay-only runs).
        lam = condition["ewc_lambda"] if ewc_lambda is None else ewc_lambda
        # Same for --replay N; otherwise the condition's replay switch decides.
        per_task = (
            replay_per_task if replay_per_task is not None
            else REPLAY_PER_TASK if condition["replay"] else 0
        )
        print(
            f"Condition {number}/{len(EWC_EXPERIMENT_CONDITIONS)}: "
            f"{condition['epochs']} epochs, {condition['time_steps']} SNN time steps, "
            f"scenario={scenario}, ewc_lambda={lam}, "
            f"replay={'all' if joint else per_task}"
        )
        ann = run_ann_ewc(
            data_dir=data_dir, batch_size=batch_size, epochs=condition["epochs"],
            learning_rate=learning_rate, ewc_lambda=lam,
            num_workers=num_workers, max_tasks=max_tasks, progress=progress,
            seeds=SEEDS,
            fisher_samples=fisher_samples, fisher_method=fisher_method,
            scenario=scenario, replay_per_task=per_task, joint=joint,
        )
        snn = run_snn_ewc(
            data_dir=data_dir, batch_size=batch_size, epochs=condition["epochs"],
            time_steps=condition["time_steps"], learning_rate=learning_rate,
            ewc_lambda=lam, num_workers=num_workers, max_tasks=max_tasks,
            progress=progress, seeds=SEEDS,
            fisher_samples=fisher_samples, fisher_method=fisher_method,
            scenario=scenario, replay_per_task=per_task, joint=joint,
        )
        results["conditions"].append({
            "epochs": condition["epochs"],
            "snn_time_steps": condition["time_steps"],
            "ewc_lambda": lam,
            "fisher_samples": fisher_samples,
            "fisher_method": fisher_method,
            "scenario": scenario,
            "replay_per_task": "all" if joint else per_task,
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
    parser.add_argument("--ewc-lambda", type=float, default=None, help="override every condition's lambda")
    parser.add_argument("--fisher-samples", type=int, default=1024)
    parser.add_argument("--fisher-method", choices=("empirical", "sampled"), default="empirical")
    parser.add_argument("--scenario", choices=("class", "task"), default="class")
    parser.add_argument("--replay", type=int, default=None, help="override every condition: images per old task (0 = off)")
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
