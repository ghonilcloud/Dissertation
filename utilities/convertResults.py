"""Convert baseline experiment JSON results into Excel-compatible CSV files."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def _format_value(value: Any) -> Any:
    """Format values for locales using commas as decimal separators."""
    if isinstance(value, float):
        return format(value, ".15g").replace(".", ",")
    if isinstance(value, (dict, list)):
        return json.dumps(value, separators=(",", ":"))
    return value


def _rows_from_results(data: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for condition in data["conditions"]:
        for model_name in ("ann", "snn"):
            model = condition[model_name]
            for run in model["runs"]:
                for task in run["tasks"]:
                    row = {
                        "model": model_name.upper(),
                        "epochs": condition["epochs"],
                        "snn_time_steps": condition["snn_time_steps"],
                        "seed": run["seed"],
                        "task": task["task"],
                        "classes": ", ".join(task["classes"]),
                        "current_task_accuracy": task["current_task_accuracy"],
                        "seen_accuracy": task["seen_accuracy"],
                        "previous_task_accuracy": task["previous_task_accuracy"],
                        "average_accuracy": task["average_accuracy"],
                        "forgetting": task["forgetting"],
                        "training_loss": task["training_loss"],
                        "training_time_seconds": task["training_time_seconds"],
                        "final_accuracy": run["final_accuracy"],
                        "final_seen_accuracy": run["final_seen_accuracy"],
                        "total_training_time_seconds": (
                            run["total_training_time_seconds"]
                        ),
                        "mean_final_accuracy": model["mean_final_accuracy"],
                        "mean_final_seen_accuracy": model["mean_final_seen_accuracy"],
                        "training_time_total_seconds": (
                            model["training_time_total_seconds"]
                        ),
                        "training_time_mean_seconds_per_seed": (
                            model["training_time_mean_seconds_per_seed"]
                        ),
                    }
                    rows.append(
                        {key: _format_value(value) for key, value in row.items()}
                    )
    return rows


def convert_file(input_path: str | Path, output_path: str | Path | None = None) -> Path:
    """Convert one JSON result file into an Excel-compatible CSV file."""
    source = Path(input_path)
    target = (
        Path(output_path)
        if output_path is not None
        else source.with_name(f"{source.stem}_excel.csv")
    )
    data = json.loads(source.read_text(encoding="utf-8"))
    rows = _rows_from_results(data)
    if not rows:
        raise ValueError(f"No experiment rows found in {source}")

    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(rows[0]),
            delimiter=";",
        )
        writer.writeheader()
        writer.writerows(rows)
    return target


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("files", nargs="+", type=Path)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for generated CSV files; defaults to each input directory.",
    )
    args = parser.parse_args()

    for input_path in args.files:
        output_path = None
        if args.output_dir is not None:
            output_path = args.output_dir / f"{input_path.stem}_excel.csv"
        print(convert_file(input_path, output_path))


if __name__ == "__main__":
    main()
