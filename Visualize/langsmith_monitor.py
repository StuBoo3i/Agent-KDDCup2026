from __future__ import annotations

import csv
import json
import shutil
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from langsmith import Client

from data_agent_baseline.benchmark.dataset import DABenchPublicDataset


FAILED_RUNS_DIRNAME = "_failed_runs"
SYNC_STATE_FILENAME = ".langsmith_sync_state.json"
SCRIPT_VERSION = "1.0.0"


# -----------------------------------------------------------------------------
# Runtime parameters (edit here before running)
# -----------------------------------------------------------------------------
MONITOR_CONFIG: dict[str, Any] = {
    # Root directory that contains local run folders (for example 20260502T134143Z).
    "runs_root": Path("D:\\Project\\Hackthon\\KDDCUP\\baseline\\Agent-KDDCup2026\\artifacts") / "runs",
    # Optional explicit run directories to process.
    # If non-empty, only these runs are processed; if empty, script scans runs_root automatically.
    # Example:
    # "runs_paths": [
    #     Path("artifacts") / "runs" / "20260502T134143Z",
    #     Path("artifacts") / "runs" / "20260504T073346Z",
    # ],
    "runs_paths": [],
    # Dataset root used to map each task to difficulty labels (easy/medium/hard/extreme).
    "dataset_root": Path("data") / "public" / "input",
    # LangSmith project name where monitoring runs will be uploaded.
    "project": "Trial",
    # Whether to clean failed local runs after sync.
    "cleanup_failed_runs": True,
    # Cleanup policy:
    # - failed-only: clean failed/invalid runs only
    # - failed-or-partial: clean failed/invalid/partial runs
    "cleanup_policy": "failed-only",
    # Cleanup mode:
    # - move: move cleaned runs into artifacts/runs/_failed_runs
    # - delete: permanently delete cleaned runs
    "cleanup_mode": "move",
    # Dry run prints planned actions and report, without writing to LangSmith
    # and without filesystem cleanup side effects.
    "dry_run": False,
    # Force re-sync even if local .langsmith_sync_state.json fingerprint is unchanged.
    "force": False,
}


def _parse_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    normalized = str(value).strip().lower()
    return normalized in {"1", "true", "yes", "y", "t"}


def _sanitize_tag_fragment(value: object) -> str:
    text = str(value).strip().lower()
    if not text:
        return "none"
    sanitized = []
    for char in text:
        if char.isalnum() or char in {"-", "_", ".", ":"}:
            sanitized.append(char)
        else:
            sanitized.append("_")
    return "".join(sanitized)


def _safe_task_sort_key(task_id: str) -> tuple[int, str]:
    if task_id.startswith("task_"):
        try:
            return int(task_id.split("_", 1)[1]), task_id
        except ValueError:
            pass
    return 10**9, task_id


def _difficulty_group(difficulty: str) -> str:
    normalized = difficulty.strip().lower()
    if normalized == "easy":
        return "easy"
    if normalized in {"medium", "hard", "extreme", "difficult"}:
        return "difficult"
    return "unknown"


def _iso_now() -> str:
    return datetime.now(UTC).isoformat()


def _contains_path(parent: Path, child: Path) -> bool:
    parent_resolved = parent.resolve()
    child_resolved = child.resolve()
    return parent_resolved == child_resolved or parent_resolved in child_resolved.parents


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_eval_details(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        result: dict[str, dict[str, Any]] = {}
        for row in reader:
            task_id = str(row.get("task_id", "")).strip()
            if not task_id:
                continue
            result[task_id] = dict(row)
        return result


def _build_difficulty_map(dataset_root: Path) -> dict[str, str]:
    dataset = DABenchPublicDataset(dataset_root)
    mapping: dict[str, str] = {}
    if not dataset.exists:
        return mapping
    for task in dataset.iter_tasks():
        mapping[task.task_id] = task.difficulty
    return mapping


@dataclass(slots=True)
class TaskMonitorRecord:
    task_id: str
    difficulty: str
    succeeded: bool
    completed: bool
    exact_match: bool
    unordered_row_match: bool
    columns_match: bool
    has_prediction: bool
    has_gold: bool
    row_count_prediction: int | None
    row_count_gold: int | None
    failure_reason: str | None
    trace_path: str | None
    prediction_csv_path: str | None
    eval_error: str | None
    eval_tags: dict[str, Any]


@dataclass(slots=True)
class RunMonitorRecord:
    run_id: str
    run_dir: Path
    status: str  # success|partial|failed|invalid
    task_count: int
    succeeded_task_count: int
    exact_match_count: int
    exact_match_rate: float
    missing_prediction_count: int
    error_count: int
    tasks: list[TaskMonitorRecord]
    has_evaluation: bool


def _to_int(value: object) -> int | None:
    try:
        if value is None or value == "":
            return None
        return int(str(value))
    except ValueError:
        return None


def _collect_run_record(run_dir: Path, difficulty_map: dict[str, str]) -> RunMonitorRecord | None:
    summary_path = run_dir / "summary.json"
    summary: dict[str, Any]
    if summary_path.exists():
        summary = _read_json(summary_path)
    else:
        task_dirs = sorted(
            [item for item in run_dir.iterdir() if item.is_dir() and item.name.startswith("task_")],
            key=lambda item: _safe_task_sort_key(item.name),
        )
        synthesized_tasks: list[dict[str, Any]] = []
        for task_dir in task_dirs:
            trace_path = task_dir / "trace.json"
            prediction_path = task_dir / "prediction.csv"
            succeeded = False
            failure_reason: str | None = None
            if trace_path.exists():
                try:
                    trace_payload = _read_json(trace_path)
                    succeeded = bool(trace_payload.get("succeeded", False))
                    reason = trace_payload.get("failure_reason")
                    failure_reason = str(reason).strip() if reason else None
                except Exception:  # noqa: BLE001
                    failure_reason = "trace.json parse error"
            else:
                failure_reason = "missing trace.json"
            synthesized_tasks.append(
                {
                    "task_id": task_dir.name,
                    "prediction_csv_path": str(prediction_path) if prediction_path.exists() else None,
                    "trace_path": str(trace_path) if trace_path.exists() else None,
                    "succeeded": succeeded,
                    "failure_reason": failure_reason,
                }
            )
        summary = {
            "run_id": run_dir.name,
            "task_count": len(synthesized_tasks),
            "succeeded_task_count": sum(1 for item in synthesized_tasks if item["succeeded"]),
            "tasks": synthesized_tasks,
        }
    eval_summary_path = run_dir / "evaluation_summary.json"
    eval_details_path = run_dir / "evaluation_details.csv"
    eval_summary = _read_json(eval_summary_path) if eval_summary_path.exists() else {}
    eval_details_by_task = _read_eval_details(eval_details_path)

    task_count = int(summary.get("task_count", 0))
    succeeded_task_count = int(summary.get("succeeded_task_count", 0))
    if task_count <= 0:
        status = "invalid"
    elif succeeded_task_count == task_count:
        status = "success"
    elif succeeded_task_count > 0:
        status = "partial"
    else:
        status = "failed"

    summary_task_map: dict[str, dict[str, Any]] = {}
    for task_item in summary.get("tasks", []):
        task_id = str(task_item.get("task_id", "")).strip()
        if task_id:
            summary_task_map[task_id] = task_item

    all_task_ids = set(summary_task_map) | set(eval_details_by_task)
    tasks: list[TaskMonitorRecord] = []
    for task_id in sorted(all_task_ids, key=_safe_task_sort_key):
        task_summary = summary_task_map.get(task_id, {})
        eval_row = eval_details_by_task.get(task_id, {})
        difficulty = difficulty_map.get(task_id, "unknown")
        succeeded = bool(task_summary.get("succeeded", False))
        failure_reason = task_summary.get("failure_reason")

        has_prediction = _parse_bool(eval_row.get("has_prediction")) if eval_row else bool(
            task_summary.get("prediction_csv_path")
        )
        has_gold = _parse_bool(eval_row.get("has_gold")) if eval_row else False
        columns_match = _parse_bool(eval_row.get("columns_match")) if eval_row else False
        exact_match = _parse_bool(eval_row.get("exact_match")) if eval_row else False
        unordered_row_match = _parse_bool(eval_row.get("unordered_row_match")) if eval_row else False
        eval_error = str(eval_row.get("error", "")).strip() or None
        completed = succeeded and has_prediction

        eval_tags = {k: v for k, v in eval_row.items() if k and k != "task_id"} if eval_row else {}
        tasks.append(
            TaskMonitorRecord(
                task_id=task_id,
                difficulty=difficulty,
                succeeded=succeeded,
                completed=completed,
                exact_match=exact_match,
                unordered_row_match=unordered_row_match,
                columns_match=columns_match,
                has_prediction=has_prediction,
                has_gold=has_gold,
                row_count_prediction=_to_int(eval_row.get("row_count_prediction")) if eval_row else None,
                row_count_gold=_to_int(eval_row.get("row_count_gold")) if eval_row else None,
                failure_reason=str(failure_reason).strip() if failure_reason else None,
                trace_path=str(task_summary.get("trace_path")) if task_summary.get("trace_path") else None,
                prediction_csv_path=(
                    str(task_summary.get("prediction_csv_path"))
                    if task_summary.get("prediction_csv_path")
                    else None
                ),
                eval_error=eval_error,
                eval_tags=eval_tags,
            )
        )

    return RunMonitorRecord(
        run_id=run_dir.name,
        run_dir=run_dir,
        status=status,
        task_count=task_count,
        succeeded_task_count=succeeded_task_count,
        exact_match_count=int(eval_summary.get("exact_match_count", 0)),
        exact_match_rate=float(eval_summary.get("exact_match_rate", 0.0)),
        missing_prediction_count=int(eval_summary.get("missing_prediction_count", 0)),
        error_count=int(eval_summary.get("error_count", 0)),
        tasks=tasks,
        has_evaluation=eval_summary_path.exists() and eval_details_path.exists(),
    )


def _read_sync_state(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return {}


def _write_sync_state(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _make_eval_tags(record: TaskMonitorRecord) -> list[str]:
    tags = [
        f"difficulty:{_sanitize_tag_fragment(record.difficulty)}",
        f"difficulty_group:{_difficulty_group(record.difficulty)}",
        f"task_completed:{str(record.completed).lower()}",
        f"task_succeeded:{str(record.succeeded).lower()}",
        f"exact_match:{str(record.exact_match).lower()}",
        f"unordered_row_match:{str(record.unordered_row_match).lower()}",
        f"columns_match:{str(record.columns_match).lower()}",
        f"has_prediction:{str(record.has_prediction).lower()}",
        f"has_gold:{str(record.has_gold).lower()}",
    ]
    for key, value in record.eval_tags.items():
        if key == "error":
            continue
        tags.append(f"eval_{_sanitize_tag_fragment(key)}:{_sanitize_tag_fragment(value)}")
    if record.eval_error:
        tags.append("eval_has_error:true")
    return tags


def _sync_run_to_langsmith(client: Client, project_name: str, run_record: RunMonitorRecord) -> dict[str, Any]:
    sync_time = datetime.now(UTC)
    run_id_uuid = uuid.uuid4()
    run_tags = [
        f"run_id:{run_record.run_id}",
        f"run_status:{run_record.status}",
        f"exact_match_rate:{run_record.exact_match_rate:.4f}",
        f"task_count:{run_record.task_count}",
        f"succeeded_task_count:{run_record.succeeded_task_count}",
        f"has_evaluation:{str(run_record.has_evaluation).lower()}",
    ]
    difficulty_set = sorted({task.difficulty for task in run_record.tasks})
    difficulty_group_set = sorted({_difficulty_group(task.difficulty) for task in run_record.tasks})
    run_tags.extend([f"difficulty:{_sanitize_tag_fragment(item)}" for item in difficulty_set])
    run_tags.extend([f"difficulty_group:{item}" for item in difficulty_group_set])

    run_inputs = {
        "run_id": run_record.run_id,
        "run_dir": str(run_record.run_dir),
    }
    run_outputs = {
        "status": run_record.status,
        "task_count": run_record.task_count,
        "succeeded_task_count": run_record.succeeded_task_count,
        "exact_match_count": run_record.exact_match_count,
        "exact_match_rate": run_record.exact_match_rate,
        "missing_prediction_count": run_record.missing_prediction_count,
        "error_count": run_record.error_count,
    }
    client.create_run(
        id=run_id_uuid,
        name=f"kddcup_run_{run_record.run_id}",
        run_type="chain",
        project_name=project_name,
        inputs=run_inputs,
        outputs=run_outputs,
        tags=run_tags,
        start_time=sync_time,
        end_time=sync_time,
        extra={
            "metadata": {
                "source": "artifacts/runs",
                "script_version": SCRIPT_VERSION,
                "difficulty_set": difficulty_set,
                "difficulty_group_set": difficulty_group_set,
            }
        },
    )

    child_count = 0
    for task in run_record.tasks:
        task_run_id = uuid.uuid4()
        task_inputs = {
            "run_id": run_record.run_id,
            "task_id": task.task_id,
            "difficulty": task.difficulty,
        }
        task_outputs = {
            "completed": task.completed,
            "succeeded": task.succeeded,
            "exact_match": task.exact_match,
            "unordered_row_match": task.unordered_row_match,
            "columns_match": task.columns_match,
            "has_prediction": task.has_prediction,
            "has_gold": task.has_gold,
            "row_count_prediction": task.row_count_prediction,
            "row_count_gold": task.row_count_gold,
            "failure_reason": task.failure_reason,
            "evaluation_error": task.eval_error,
            "trace_path": task.trace_path,
            "prediction_csv_path": task.prediction_csv_path,
        }
        task_tags = [f"run_id:{run_record.run_id}", f"task_id:{task.task_id}", *_make_eval_tags(task)]
        client.create_run(
            id=task_run_id,
            name=f"{run_record.run_id}:{task.task_id}",
            run_type="tool",
            project_name=project_name,
            inputs=task_inputs,
            outputs=task_outputs,
            tags=task_tags,
            parent_run_id=run_id_uuid,
            start_time=sync_time,
            end_time=sync_time,
            extra={
                "metadata": {
                    "eval_details": task.eval_tags,
                }
            },
        )
        child_count += 1
    return {"root_run_id": str(run_id_uuid), "child_run_count": child_count}


def _cleanup_run(run_record: RunMonitorRecord, runs_root: Path, cleanup_mode: str, dry_run: bool) -> str:
    run_dir = run_record.run_dir.resolve()
    if not _contains_path(runs_root.resolve(), run_dir):
        raise RuntimeError(f"Refusing cleanup outside runs root: {run_dir}")

    if cleanup_mode == "delete":
        if dry_run:
            return f"dry-run delete {run_dir}"
        shutil.rmtree(run_dir)
        return f"deleted {run_dir}"

    failed_root = runs_root / FAILED_RUNS_DIRNAME
    target = failed_root / run_record.run_id
    suffix = 1
    while target.exists():
        target = failed_root / f"{run_record.run_id}_{suffix}"
        suffix += 1
    if dry_run:
        return f"dry-run move {run_dir} -> {target}"
    failed_root.mkdir(parents=True, exist_ok=True)
    shutil.move(str(run_dir), str(target))
    return f"moved {run_dir} -> {target}"


def _build_report(run_records: list[RunMonitorRecord], synced_count: int, cleaned_count: int) -> dict[str, Any]:
    status_counts: dict[str, int] = {}
    difficulty_counts: dict[str, int] = {}
    for run in run_records:
        status_counts[run.status] = status_counts.get(run.status, 0) + 1
        for task in run.tasks:
            difficulty_counts[task.difficulty] = difficulty_counts.get(task.difficulty, 0) + 1
    return {
        "generated_at": _iso_now(),
        "script_version": SCRIPT_VERSION,
        "run_count": len(run_records),
        "synced_run_count": synced_count,
        "cleaned_run_count": cleaned_count,
        "status_counts": status_counts,
        "task_difficulty_counts": dict(sorted(difficulty_counts.items(), key=lambda item: item[0])),
    }


def _resolve_monitor_config(config: dict[str, Any]) -> dict[str, Any]:
    resolved = dict(config)
    resolved["runs_root"] = Path(resolved.get("runs_root", Path("artifacts") / "runs")).resolve()
    raw_runs_paths = resolved.get("runs_paths", [])
    if raw_runs_paths is None:
        raw_runs_paths = []
    if not isinstance(raw_runs_paths, list):
        raise ValueError("runs_paths must be a list of paths.")
    resolved["runs_paths"] = [Path(item).resolve() for item in raw_runs_paths]
    resolved["dataset_root"] = Path(resolved.get("dataset_root", Path("data") / "public" / "input")).resolve()
    resolved["project"] = str(resolved.get("project", "kddcup-runs-monitor")).strip()
    resolved["cleanup_failed_runs"] = bool(resolved.get("cleanup_failed_runs", True))
    resolved["cleanup_policy"] = str(resolved.get("cleanup_policy", "failed-only")).strip()
    resolved["cleanup_mode"] = str(resolved.get("cleanup_mode", "move")).strip()
    resolved["dry_run"] = bool(resolved.get("dry_run", False))
    resolved["force"] = bool(resolved.get("force", False))

    if resolved["cleanup_policy"] not in {"failed-only", "failed-or-partial"}:
        raise ValueError("cleanup_policy must be 'failed-only' or 'failed-or-partial'.")
    if resolved["cleanup_mode"] not in {"move", "delete"}:
        raise ValueError("cleanup_mode must be 'move' or 'delete'.")
    if not resolved["project"]:
        raise ValueError("project must not be empty.")
    return resolved


def _resolve_run_dirs(runs_root: Path, runs_paths: list[Path]) -> list[Path]:
    if runs_paths:
        run_dirs: list[Path] = []
        for run_path in runs_paths:
            if not run_path.exists():
                print(f"[warn] runs_paths item not found: {run_path}")
                continue
            if not run_path.is_dir():
                print(f"[warn] runs_paths item is not a directory: {run_path}")
                continue
            run_dirs.append(run_path)
        return sorted(run_dirs, key=lambda path: path.name)

    return sorted(
        [
            path
            for path in runs_root.iterdir()
            if path.is_dir() and path.name.startswith("20") and path.name != FAILED_RUNS_DIRNAME
        ],
        key=lambda path: path.name,
    )


def main(config: dict[str, Any] | None = None) -> None:
    effective_config = _resolve_monitor_config(config or MONITOR_CONFIG)

    runs_root = effective_config["runs_root"]
    runs_paths = effective_config["runs_paths"]
    dataset_root = effective_config["dataset_root"]
    project = effective_config["project"]
    cleanup_failed_runs = effective_config["cleanup_failed_runs"]
    cleanup_policy = effective_config["cleanup_policy"]
    cleanup_mode = effective_config["cleanup_mode"]
    dry_run = effective_config["dry_run"]
    force = effective_config["force"]

    if not runs_root.exists():
        raise SystemExit(f"Runs root not found: {runs_root}")

    difficulty_map = _build_difficulty_map(dataset_root)
    run_dirs = _resolve_run_dirs(runs_root, runs_paths)

    run_records: list[RunMonitorRecord] = []
    for run_dir in run_dirs:
        record = _collect_run_record(run_dir, difficulty_map)
        if record is not None:
            run_records.append(record)

    synced_count = 0
    cleaned_count = 0
    client = None if dry_run else Client()

    for run_record in run_records:
        sync_state_path = run_record.run_dir / SYNC_STATE_FILENAME
        sync_state = _read_sync_state(sync_state_path)
        fingerprint = {
            "script_version": SCRIPT_VERSION,
            "status": run_record.status,
            "task_count": run_record.task_count,
            "succeeded_task_count": run_record.succeeded_task_count,
            "exact_match_count": run_record.exact_match_count,
            "exact_match_rate": run_record.exact_match_rate,
            "missing_prediction_count": run_record.missing_prediction_count,
            "error_count": run_record.error_count,
            "task_ids": [task.task_id for task in run_record.tasks],
        }
        if not force and sync_state.get("project") == project and sync_state.get("fingerprint") == fingerprint:
            print(f"[skip] {run_record.run_id}: unchanged and already synced")
        else:
            if dry_run:
                print(f"[dry-run] would sync run {run_record.run_id} to project={project}")
            else:
                sync_result = _sync_run_to_langsmith(client, project, run_record)
                _write_sync_state(
                    sync_state_path,
                    {
                        "project": project,
                        "synced_at": _iso_now(),
                        "fingerprint": fingerprint,
                        "langsmith_root_run_id": sync_result["root_run_id"],
                        "child_run_count": sync_result["child_run_count"],
                    },
                )
                print(
                    f"[synced] {run_record.run_id}: root_run_id={sync_result['root_run_id']} "
                    f"tasks={sync_result['child_run_count']}"
                )
            synced_count += 1

        should_cleanup = False
        if cleanup_failed_runs:
            if cleanup_policy == "failed-only":
                should_cleanup = run_record.status in {"failed", "invalid"}
            elif cleanup_policy == "failed-or-partial":
                should_cleanup = run_record.status in {"failed", "partial", "invalid"}
        if should_cleanup:
            message = _cleanup_run(run_record, runs_root, cleanup_mode, dry_run)
            print(f"[cleanup] {message}")
            cleaned_count += 1

    report = _build_report(run_records, synced_count=synced_count, cleaned_count=cleaned_count)
    report_path = runs_root / "langsmith_sync_report.json"
    if dry_run:
        print("[dry-run] report")
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"[done] report saved to {report_path}")
        print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
