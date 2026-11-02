"""Run ANN and SpikingJelly SNN baseline conditions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from baselineAnn import DEFAULT_SEEDS, run_experiment as run_ann
from baselineSnn import run_experiment as run_snn


EXPERIMENT_CONDITIONS = (
    # {"epochs": 1, "time_steps": 2},
    # {"epochs": 2, "time_steps": 4},
    {"epochs": 5, "time_steps": 7},
    # {"epochs": 10, "time_steps": 10},
    # {"epochs": 20, "time_steps": 10},
    # {"epochs": 25, "time_steps": 10},
    # {"epochs": 30, "time_steps": 10},
)


def summarize_model_results(runs: list[dict[str, Any]]) -> dict[str, Any]:
    """Add across-seed final-accuracy and training-time summaries."""
    final_accuracies = [float(run["final_accuracy"]) for run in runs]
    final_seen_accuracies = [float(run["final_seen_accuracy"]) for run in runs]
    training_times = [float(run["total_training_time_seconds"]) for run in runs]
    return {
        "runs": runs,
        "mean_final_accuracy": sum(final_accuracies) / len(final_accuracies),
        "mean_final_seen_accuracy": (
            sum(final_seen_accuracies) / len(final_seen_accuracies)
        ),
        "training_time_total_seconds": sum(training_times),
        "training_time_mean_seconds_per_seed": (
            sum(training_times) / len(training_times)
        ),
    }


def run_baseline_grid(
    *,
    data_dir: str | Path = "data",
    batch_size: int = 128,
    learning_rate: float = 1e-3,
    num_workers: int = 0,
    download: bool = False,
    max_tasks: int | None = None,
    progress: bool = True,
    output_path: str | Path = "results/baseline_experiments.json",
) -> dict[str, Any]:
    """Run every ANN/SNN condition for all configured random seeds.

    Results are written after each condition so completed experiments remain
    available if a later condition is interrupted.
    """
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    results: dict[str, Any] = {
        "seeds": list(DEFAULT_SEEDS),
        "conditions": [],
    }

    for condition_number, condition in enumerate(EXPERIMENT_CONDITIONS, start=1):
        epochs = condition["epochs"]
        time_steps = condition["time_steps"]
        print(
            f"Condition {condition_number}/{len(EXPERIMENT_CONDITIONS)}: "
            f"{epochs} epochs, {time_steps} SNN time steps"
        )
        ann_runs = run_ann(
            data_dir=data_dir,
            batch_size=batch_size,
            epochs=epochs,
            learning_rate=learning_rate,
            num_workers=num_workers,
            download=download,
            max_tasks=max_tasks,
            progress=progress,
            seeds=DEFAULT_SEEDS,
        )
        snn_runs = run_snn(
            data_dir=data_dir,
            batch_size=batch_size,
            epochs=epochs,
            learning_rate=learning_rate,
            time_steps=time_steps,
            num_workers=num_workers,
            download=download,
            max_tasks=max_tasks,
            progress=progress,
            seeds=DEFAULT_SEEDS,
        )
        condition_result = {
            "epochs": epochs,
            "snn_time_steps": time_steps,
            "ann": summarize_model_results(ann_runs),
            "snn": summarize_model_results(snn_runs),
        }
        results["conditions"].append(condition_result)
        output_file.write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"Saved completed condition to {output_file}")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-tasks", type=int, default=None)
    parser.add_argument("--output", default="results/baseline_experiments.json")
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    args = parser.parse_args()
    run_baseline_grid(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        num_workers=args.num_workers,
        download=args.download,
        max_tasks=args.max_tasks,
        progress=not args.no_progress,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
