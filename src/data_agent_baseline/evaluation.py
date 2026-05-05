from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class TaskEvalResult:
    task_id: str
    has_prediction: bool
    has_gold: bool
    columns_match: bool
    row_count_prediction: int
    row_count_gold: int
    exact_match: bool
    unordered_row_match: bool
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "has_prediction": self.has_prediction,
            "has_gold": self.has_gold,
            "columns_match": self.columns_match,
            "row_count_prediction": self.row_count_prediction,
            "row_count_gold": self.row_count_gold,
            "exact_match": self.exact_match,
            "unordered_row_match": self.unordered_row_match,
            "error": self.error,
        }


def _read_csv(path: Path) -> tuple[list[str], list[list[str]]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        rows = list(reader)
    if not rows:
        return [], []
    return list(rows[0]), [list(row) for row in rows[1:]]


def _unordered_rows(rows: list[list[str]]) -> list[tuple[str, ...]]:
    normalized = [tuple(item.strip() for item in row) for row in rows]
    normalized.sort()
    return normalized


def evaluate_run_outputs(run_output_dir: Path, gold_output_root: Path) -> dict[str, Any]:
    task_dirs = sorted([path for path in run_output_dir.iterdir() if path.is_dir() and path.name.startswith("task_")])
    results: list[TaskEvalResult] = []

    for task_dir in task_dirs:
        task_id = task_dir.name
        prediction_path = task_dir / "prediction.csv"
        gold_path = gold_output_root / task_id / "gold.csv"

        has_prediction = prediction_path.exists()
        has_gold = gold_path.exists()

        if not has_prediction or not has_gold:
            results.append(
                TaskEvalResult(
                    task_id=task_id,
                    has_prediction=has_prediction,
                    has_gold=has_gold,
                    columns_match=False,
                    row_count_prediction=0,
                    row_count_gold=0,
                    exact_match=False,
                    unordered_row_match=False,
                    error="missing prediction.csv" if not has_prediction else "missing gold.csv",
                )
            )
            continue

        try:
            pred_columns, pred_rows = _read_csv(prediction_path)
            gold_columns, gold_rows = _read_csv(gold_path)
            columns_match = pred_columns == gold_columns
            exact_match = columns_match and pred_rows == gold_rows
            unordered_row_match = columns_match and _unordered_rows(pred_rows) == _unordered_rows(gold_rows)

            results.append(
                TaskEvalResult(
                    task_id=task_id,
                    has_prediction=True,
                    has_gold=True,
                    columns_match=columns_match,
                    row_count_prediction=len(pred_rows),
                    row_count_gold=len(gold_rows),
                    exact_match=exact_match,
                    unordered_row_match=unordered_row_match,
                )
            )
        except Exception as exc:  # noqa: BLE001
            results.append(
                TaskEvalResult(
                    task_id=task_id,
                    has_prediction=True,
                    has_gold=True,
                    columns_match=False,
                    row_count_prediction=0,
                    row_count_gold=0,
                    exact_match=False,
                    unordered_row_match=False,
                    error=str(exc),
                )
            )

    evaluated_count = len(results)
    exact_match_count = sum(1 for item in results if item.exact_match)
    unordered_row_match_count = sum(1 for item in results if item.unordered_row_match)
    missing_prediction_count = sum(1 for item in results if not item.has_prediction)
    missing_gold_count = sum(1 for item in results if not item.has_gold)
    error_count = sum(1 for item in results if item.error is not None)

    summary = {
        "run_output_dir": str(run_output_dir),
        "gold_output_root": str(gold_output_root),
        "task_count_evaluated": evaluated_count,
        "exact_match_count": exact_match_count,
        "exact_match_rate": (exact_match_count / evaluated_count) if evaluated_count else 0.0,
        "unordered_row_match_count": unordered_row_match_count,
        "unordered_row_match_rate": (unordered_row_match_count / evaluated_count) if evaluated_count else 0.0,
        "missing_prediction_count": missing_prediction_count,
        "missing_gold_count": missing_gold_count,
        "error_count": error_count,
        "tasks": [item.to_dict() for item in results],
    }
    return summary


def write_evaluation_artifacts(summary: dict[str, Any], destination_dir: Path) -> tuple[Path, Path]:
    destination_dir.mkdir(parents=True, exist_ok=True)
    summary_json_path = destination_dir / "evaluation_summary.json"
    details_csv_path = destination_dir / "evaluation_details.csv"

    summary_json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    fields = [
        "task_id",
        "has_prediction",
        "has_gold",
        "columns_match",
        "row_count_prediction",
        "row_count_gold",
        "exact_match",
        "unordered_row_match",
        "error",
    ]
    with details_csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for item in summary.get("tasks", []):
            writer.writerow({field: item.get(field) for field in fields})

    return summary_json_path, details_csv_path
