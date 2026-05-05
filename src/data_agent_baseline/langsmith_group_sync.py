from __future__ import annotations

import json
import os
import re
import time
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from langsmith import Client

from data_agent_baseline.benchmark.dataset import DABenchPublicDataset
from data_agent_baseline.config import AppConfig
from data_agent_baseline.run.runner import TaskRunArtifacts


def _sanitize_tag_fragment(value: object) -> str:
    text = str(value).strip().lower()
    if not text:
        return "none"
    chars: list[str] = []
    for char in text:
        if char.isalnum() or char in {"-", "_", ".", ":"}:
            chars.append(char)
        else:
            chars.append("_")
    return "".join(chars)


def _difficulty_group(difficulty: str) -> str:
    normalized = difficulty.strip().lower()
    if normalized == "easy":
        return "easy"
    if normalized in {"medium", "hard", "extreme", "difficult"}:
        return "difficult"
    return "unknown"


def _safe_task_sort_key(task_id: str) -> tuple[int, str]:
    if task_id.startswith("task_"):
        try:
            return int(task_id.split("_", 1)[1]), task_id
        except ValueError:
            pass
    return 10**9, task_id


def _format_group_time_prefix(run_group_id: str) -> str:
    # Run ids are usually UTC timestamps like 20260504T073346Z.
    if re.fullmatch(r"\d{8}T\d{6}Z", run_group_id):
        dt_utc = datetime.strptime(run_group_id, "%Y%m%dT%H%M%SZ").replace(tzinfo=UTC)
        dt_local = dt_utc.astimezone()
        return dt_local.strftime("%Y_%m_%d_%H_%M")
    return datetime.now().strftime("%Y_%m_%d_%H_%M")


def _task_label(task_id: str) -> str:
    if task_id.startswith("task_"):
        suffix = task_id.split("_", 1)[1]
        if suffix.isdigit():
            return f"Task{int(suffix)}"
    return f"Task{_sanitize_tag_fragment(task_id)}"


def _task_trace_name(run_group_id: str, task_id: str) -> str:
    return f"{_format_group_time_prefix(run_group_id)}_{_task_label(task_id)}"


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _bounded_text(value: object, *, max_chars: int = 8000) -> str:
    text = str(value)
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n...[truncated]"


def _to_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    normalized = str(value).strip().lower()
    return normalized in {"1", "true", "yes", "y", "t"}


def _non_empty_unique(values: list[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        normalized = value.strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        ordered.append(normalized)
    return ordered


def _resolve_runtime_project_candidates(langsmith_meta: dict[str, Any]) -> list[str]:
    return _non_empty_unique(
        [
            str(langsmith_meta.get("project", "")),
            os.getenv("LANGSMITH_PROJECT", ""),
            os.getenv("LANGCHAIN_PROJECT", ""),
        ]
    )


def _find_runtime_run_id_fallback(
    *,
    client: Client,
    project_candidates: list[str],
    task_id: str,
    task_trace_name: str,
) -> str | None:
    best_run_id: str | None = None
    best_start_time: datetime | None = None
    task_tag = f"task_id:{task_id}" if task_id else ""
    for project_name in project_candidates:
        try:
            runs = client.list_runs(project_name=project_name, is_root=True, limit=200)
        except Exception:  # noqa: BLE001
            continue
        for run in runs:
            run_tags = list(run.tags or [])
            if task_tag and task_tag not in run_tags:
                continue
            if task_trace_name and str(run.name or "") == task_trace_name:
                return str(run.id)
            if "source:langgraph_runtime" not in run_tags:
                continue
            started_at = getattr(run, "start_time", None)
            if best_start_time is None or (isinstance(started_at, datetime) and started_at > best_start_time):
                best_run_id = str(run.id)
                best_start_time = started_at if isinstance(started_at, datetime) else best_start_time
    return best_run_id


def _run_has_eval_metadata(run: Any) -> bool:
    if not isinstance(getattr(run, "extra", None), dict):
        return False
    metadata = run.extra.get("metadata")
    if not isinstance(metadata, dict):
        return False
    return "eval_exact_match" in metadata or "evaluation" in metadata


def _sync_eval_feedback(
    *,
    client: Client,
    run: Any,
    eval_payload: dict[str, Any],
) -> dict[str, Any]:
    run_id = str(getattr(run, "id"))
    trace_id = getattr(run, "trace_id", None) or getattr(run, "id")
    feedback_count = 0
    feedback_error: str | None = None
    entries: list[tuple[str, Any, Any, str | None]] = [
        ("eval_exact_match", eval_payload.get("exact_match"), None, None),
        ("eval_columns_match", eval_payload.get("columns_match"), None, None),
        ("eval_unordered_row_match", eval_payload.get("unordered_row_match"), None, None),
        ("eval_has_prediction", eval_payload.get("has_prediction"), None, None),
        ("eval_has_gold", eval_payload.get("has_gold"), None, None),
        (
            "eval_row_count_prediction",
            float(eval_payload.get("row_count_prediction") or 0),
            eval_payload.get("row_count_prediction"),
            None,
        ),
        (
            "eval_row_count_gold",
            float(eval_payload.get("row_count_gold") or 0),
            eval_payload.get("row_count_gold"),
            None,
        ),
        ("eval_error", None, eval_payload.get("error") or "", "Empty means no evaluation error."),
    ]
    for key, score, value, comment in entries:
        try:
            client.create_feedback(
                run_id=run_id,
                trace_id=str(trace_id),
                key=key,
                score=score if isinstance(score, (bool, int, float)) else None,
                value=value,
                comment=comment,
                source_info={"source": "run-task:auto-evaluation"},
            )
            feedback_count += 1
        except Exception as exc:  # noqa: BLE001
            feedback_error = str(exc)
    return {"feedback_count": feedback_count, "feedback_error": feedback_error}


@dataclass(frozen=True, slots=True)
class TaskSyncContext:
    task_id: str
    difficulty: str
    question: str
    trace_path: Path
    prediction_csv_path: Path | None
    succeeded: bool
    failure_reason: str | None
    eval_row: dict[str, Any]


def _build_eval_task_map(evaluation_summary: dict[str, Any]) -> dict[str, dict[str, Any]]:
    task_rows = evaluation_summary.get("tasks", [])
    if not isinstance(task_rows, list):
        return {}
    result: dict[str, dict[str, Any]] = {}
    for row in task_rows:
        if not isinstance(row, dict):
            continue
        task_id = str(row.get("task_id", "")).strip()
        if not task_id:
            continue
        result[task_id] = row
    return result


def _build_task_contexts(
    artifacts: list[TaskRunArtifacts],
    app_config: AppConfig,
    eval_map: dict[str, dict[str, Any]],
) -> list[TaskSyncContext]:
    dataset = DABenchPublicDataset(app_config.dataset.root_path)
    contexts: list[TaskSyncContext] = []
    for artifact in sorted(artifacts, key=lambda item: _safe_task_sort_key(item.task_id)):
        task_id = artifact.task_id
        difficulty = "unknown"
        question = ""
        try:
            task = dataset.get_task(task_id)
            difficulty = task.difficulty
            question = task.question
        except Exception:  # noqa: BLE001
            pass

        contexts.append(
            TaskSyncContext(
                task_id=task_id,
                difficulty=difficulty,
                question=question,
                trace_path=artifact.trace_path,
                prediction_csv_path=artifact.prediction_csv_path,
                succeeded=artifact.succeeded,
                failure_reason=artifact.failure_reason,
                eval_row=eval_map.get(task_id, {}),
            )
        )
    return contexts


def _task_eval_tags(eval_row: dict[str, Any]) -> list[str]:
    tags: list[str] = []
    for key, value in sorted(eval_row.items(), key=lambda item: item[0]):
        if key == "task_id":
            continue
        if key == "error" and value:
            tags.append("eval_has_error:true")
            continue
        tags.append(f"eval_{_sanitize_tag_fragment(key)}:{_sanitize_tag_fragment(value)}")
    return tags


def sync_run_and_eval_to_langsmith(
    *,
    app_config: AppConfig,
    run_output_dir: Path,
    artifacts: list[TaskRunArtifacts],
    evaluation_summary: dict[str, Any],
    difficulty_filters: list[str] | None = None,
) -> dict[str, Any]:
    client = Client()
    sync_time = datetime.now(UTC)
    run_group_id = run_output_dir.name
    project_name = app_config.agent.langsmith_project
    eval_map = _build_eval_task_map(evaluation_summary)
    task_contexts = _build_task_contexts(artifacts, app_config, eval_map)

    root_run_uuid = uuid.uuid4()
    group_tags = [
        "source:dabench_run_and_eval",
        f"run_group_id:{run_group_id}",
        f"agent_framework:{_sanitize_tag_fragment(app_config.agent.framework)}",
        f"model:{_sanitize_tag_fragment(app_config.agent.model)}",
        f"task_count:{len(task_contexts)}",
    ]
    if difficulty_filters:
        normalized_filters = sorted({_sanitize_tag_fragment(item) for item in difficulty_filters})
        group_tags.extend([f"difficulty_filter:{item}" for item in normalized_filters])

    root_inputs = {
        "run_group_id": run_group_id,
        "run_output_dir": str(run_output_dir),
        "dataset_root": str(app_config.dataset.root_path),
        "difficulty_filters": difficulty_filters or [],
    }
    root_outputs = {
        "task_count_evaluated": evaluation_summary.get("task_count_evaluated"),
        "exact_match_count": evaluation_summary.get("exact_match_count"),
        "exact_match_rate": evaluation_summary.get("exact_match_rate"),
        "unordered_row_match_count": evaluation_summary.get("unordered_row_match_count"),
        "unordered_row_match_rate": evaluation_summary.get("unordered_row_match_rate"),
        "missing_prediction_count": evaluation_summary.get("missing_prediction_count"),
        "missing_gold_count": evaluation_summary.get("missing_gold_count"),
        "error_count": evaluation_summary.get("error_count"),
    }
    client.create_run(
        id=root_run_uuid,
        name=f"kddcup_run_group_{run_group_id}",
        run_type="chain",
        project_name=project_name,
        inputs=root_inputs,
        outputs=root_outputs,
        tags=group_tags,
        start_time=sync_time,
        end_time=sync_time,
        extra={
            "metadata": {
                "run_group_id": run_group_id,
                "framework": app_config.agent.framework,
                "model": app_config.agent.model,
            }
        },
    )

    task_run_count = 0
    step_run_count = 0
    for task_ctx in task_contexts:
        trace_payload = _read_json(task_ctx.trace_path)
        task_run_uuid = uuid.uuid4()

        completed = bool(task_ctx.prediction_csv_path) and task_ctx.succeeded
        exact_match = _to_bool(task_ctx.eval_row.get("exact_match"))
        unordered_row_match = _to_bool(task_ctx.eval_row.get("unordered_row_match"))
        columns_match = _to_bool(task_ctx.eval_row.get("columns_match"))
        has_prediction = _to_bool(task_ctx.eval_row.get("has_prediction")) or bool(task_ctx.prediction_csv_path)
        has_gold = _to_bool(task_ctx.eval_row.get("has_gold"))

        task_tags = [
            f"run_group_id:{run_group_id}",
            f"task_id:{task_ctx.task_id}",
            f"difficulty:{_sanitize_tag_fragment(task_ctx.difficulty)}",
            f"difficulty_group:{_difficulty_group(task_ctx.difficulty)}",
            f"task_completed:{str(completed).lower()}",
            f"task_succeeded:{str(task_ctx.succeeded).lower()}",
            f"task_correct:{str(exact_match).lower()}",
            f"columns_match:{str(columns_match).lower()}",
            f"unordered_row_match:{str(unordered_row_match).lower()}",
            f"has_prediction:{str(has_prediction).lower()}",
            f"has_gold:{str(has_gold).lower()}",
            *_task_eval_tags(task_ctx.eval_row),
        ]
        task_inputs = {
            "run_group_id": run_group_id,
            "task_id": task_ctx.task_id,
            "difficulty": task_ctx.difficulty,
            "question": task_ctx.question,
            "trace_path": str(task_ctx.trace_path),
        }
        task_outputs = {
            "completed": completed,
            "succeeded": task_ctx.succeeded,
            "failure_reason": task_ctx.failure_reason,
            "prediction_csv_path": str(task_ctx.prediction_csv_path) if task_ctx.prediction_csv_path else None,
            "exact_match": exact_match,
            "unordered_row_match": unordered_row_match,
            "columns_match": columns_match,
            "row_count_prediction": task_ctx.eval_row.get("row_count_prediction"),
            "row_count_gold": task_ctx.eval_row.get("row_count_gold"),
            "evaluation_error": task_ctx.eval_row.get("error"),
        }
        client.create_run(
            id=task_run_uuid,
            name=_task_trace_name(run_group_id, task_ctx.task_id),
            run_type="chain",
            project_name=project_name,
            parent_run_id=root_run_uuid,
            inputs=task_inputs,
            outputs=task_outputs,
            tags=task_tags,
            start_time=sync_time,
            end_time=sync_time,
            error=task_ctx.failure_reason,
            extra={
                "metadata": {
                    "run_group_id": run_group_id,
                    "task_id": task_ctx.task_id,
                    "difficulty": task_ctx.difficulty,
                    "difficulty_group": _difficulty_group(task_ctx.difficulty),
                    "eval_row": task_ctx.eval_row,
                }
            },
        )
        task_run_count += 1

        steps = trace_payload.get("steps", [])
        if not isinstance(steps, list):
            steps = []
        for raw_step in steps:
            if not isinstance(raw_step, dict):
                continue
            step_index = raw_step.get("step_index")
            action = str(raw_step.get("action", "unknown"))
            step_name = f"{task_ctx.task_id}:step_{int(step_index) if isinstance(step_index, int) else 'x'}_{action}"
            step_inputs = {
                "run_group_id": run_group_id,
                "task_id": task_ctx.task_id,
                "action": action,
                "action_input": raw_step.get("action_input"),
            }
            step_outputs = {
                "ok": raw_step.get("ok"),
                "thought": raw_step.get("thought"),
                "observation": raw_step.get("observation"),
                "raw_response": _bounded_text(raw_step.get("raw_response", "")),
            }
            step_error = None
            observation = raw_step.get("observation")
            if isinstance(observation, dict) and not observation.get("ok", True):
                if observation.get("error") is not None:
                    step_error = str(observation.get("error"))
            client.create_run(
                id=uuid.uuid4(),
                name=step_name,
                run_type="tool",
                project_name=project_name,
                parent_run_id=task_run_uuid,
                inputs=step_inputs,
                outputs=step_outputs,
                tags=[
                    f"run_group_id:{run_group_id}",
                    f"task_id:{task_ctx.task_id}",
                    f"action:{_sanitize_tag_fragment(action)}",
                    f"step_ok:{str(bool(raw_step.get('ok'))).lower()}",
                ],
                start_time=sync_time,
                end_time=sync_time,
                error=step_error,
                extra={
                    "metadata": {
                        "run_group_id": run_group_id,
                        "task_id": task_ctx.task_id,
                        "step_index": step_index,
                        "action": action,
                    }
                },
            )
            step_run_count += 1

        eval_span_outputs = {
            "exact_match": exact_match,
            "unordered_row_match": unordered_row_match,
            "columns_match": columns_match,
            "has_prediction": has_prediction,
            "has_gold": has_gold,
            "row_count_prediction": task_ctx.eval_row.get("row_count_prediction"),
            "row_count_gold": task_ctx.eval_row.get("row_count_gold"),
            "error": task_ctx.eval_row.get("error"),
        }
        client.create_run(
            id=uuid.uuid4(),
            name=f"{task_ctx.task_id}:evaluation",
            run_type="tool",
            project_name=project_name,
            parent_run_id=task_run_uuid,
            inputs={"run_group_id": run_group_id, "task_id": task_ctx.task_id},
            outputs=eval_span_outputs,
            tags=[
                f"run_group_id:{run_group_id}",
                f"task_id:{task_ctx.task_id}",
                f"task_correct:{str(exact_match).lower()}",
                f"task_completed:{str(completed).lower()}",
            ],
            start_time=sync_time,
            end_time=sync_time,
            error=str(task_ctx.eval_row.get("error")) if task_ctx.eval_row.get("error") else None,
            extra={
                "metadata": {
                    "run_group_id": run_group_id,
                    "task_id": task_ctx.task_id,
                }
            },
        )

    flush_fn = getattr(client, "flush", None)
    if callable(flush_fn):
        flush_fn()
    return {
        "project_name": project_name,
        "run_group_id": run_group_id,
        "root_run_id": str(root_run_uuid),
        "task_run_count": task_run_count,
        "step_run_count": step_run_count,
    }


def sync_task_eval_to_runtime_trace(
    *,
    trace_payload: dict[str, Any],
    task_eval_row: dict[str, Any],
) -> dict[str, Any]:
    metadata = trace_payload.get("metadata")
    if not isinstance(metadata, dict):
        return {"updated": False, "reason": "missing trace metadata"}
    langsmith_meta = metadata.get("langsmith")
    if not isinstance(langsmith_meta, dict):
        return {"updated": False, "reason": "missing langsmith metadata"}
    runtime_run_id = str(langsmith_meta.get("task_runtime_run_id", "")).strip()
    task_trace_name = str(langsmith_meta.get("task_trace_name", "")).strip()
    task_id = str(metadata.get("task_id", "")).strip()
    project_candidates = _resolve_runtime_project_candidates(langsmith_meta)
    if not runtime_run_id and not task_id:
        return {"updated": False, "reason": "missing task_runtime_run_id and task_id"}

    client = Client()
    run = None
    read_error_text: str | None = None
    if runtime_run_id:
        for attempt in range(1, 4):
            try:
                run = client.read_run(runtime_run_id)
                break
            except Exception as exc:  # noqa: BLE001
                read_error_text = str(exc)
                if attempt < 3:
                    time.sleep(0.8 * attempt)

    if run is None:
        fallback_run_id = _find_runtime_run_id_fallback(
            client=client,
            project_candidates=project_candidates,
            task_id=task_id,
            task_trace_name=task_trace_name,
        )
        if fallback_run_id:
            runtime_run_id = fallback_run_id
            try:
                run = client.read_run(runtime_run_id)
            except Exception as exc:  # noqa: BLE001
                read_error_text = str(exc)

    if run is None:
        return {
            "updated": False,
            "reason": "runtime trace not found",
            "runtime_run_id": runtime_run_id or None,
            "task_trace_name": task_trace_name or None,
            "error": read_error_text,
        }

    eval_payload = {
        "exact_match": _to_bool(task_eval_row.get("exact_match")),
        "unordered_row_match": _to_bool(task_eval_row.get("unordered_row_match")),
        "columns_match": _to_bool(task_eval_row.get("columns_match")),
        "has_prediction": _to_bool(task_eval_row.get("has_prediction")),
        "has_gold": _to_bool(task_eval_row.get("has_gold")),
        "row_count_prediction": task_eval_row.get("row_count_prediction"),
        "row_count_gold": task_eval_row.get("row_count_gold"),
        "error": task_eval_row.get("error"),
    }

    def _apply_eval_update(current_run: Any) -> None:
        existing_outputs = dict(current_run.outputs or {}) if isinstance(current_run.outputs, dict) else {}
        existing_extra = dict(current_run.extra or {}) if isinstance(current_run.extra, dict) else {}
        existing_meta = (
            dict(existing_extra.get("metadata", {}))
            if isinstance(existing_extra.get("metadata"), dict)
            else {}
        )
        merged_outputs = dict(existing_outputs)
        merged_outputs["evaluation"] = eval_payload

        merged_metadata = dict(existing_meta)
        merged_metadata["evaluation"] = eval_payload
        merged_metadata["evaluation_source"] = "run-task:auto-evaluation"
        merged_metadata["eval_exact_match"] = eval_payload["exact_match"]
        merged_metadata["eval_unordered_row_match"] = eval_payload["unordered_row_match"]
        merged_metadata["eval_columns_match"] = eval_payload["columns_match"]
        merged_metadata["eval_has_prediction"] = eval_payload["has_prediction"]
        merged_metadata["eval_has_gold"] = eval_payload["has_gold"]
        merged_metadata["eval_row_count_prediction"] = eval_payload["row_count_prediction"]
        merged_metadata["eval_row_count_gold"] = eval_payload["row_count_gold"]
        merged_metadata["eval_error"] = eval_payload["error"]

        merged_extra = dict(existing_extra)
        merged_extra["metadata"] = merged_metadata

        existing_tags = list(current_run.tags or [])
        new_tags = list(existing_tags)
        for tag in [
            f"task_correct:{str(eval_payload['exact_match']).lower()}",
            f"columns_match:{str(eval_payload['columns_match']).lower()}",
            f"has_prediction:{str(eval_payload['has_prediction']).lower()}",
        ]:
            if tag not in new_tags:
                new_tags.append(tag)

        client.update_run(
            runtime_run_id,
            name=task_trace_name or None,
            outputs=merged_outputs,
            extra=merged_extra,
            tags=new_tags,
        )
        flush_fn = getattr(client, "flush", None)
        if callable(flush_fn):
            flush_fn()

    _apply_eval_update(run)

    # LangGraph runtime may continue async ingest for a short window and overwrite metadata.
    # Re-check a few times and re-apply if eval metadata is missing.
    verified = False
    latest_run = run
    for check_index in range(1, 5):
        time.sleep(0.8 * check_index)
        latest_run = client.read_run(runtime_run_id)
        if _run_has_eval_metadata(latest_run):
            verified = True
            break
        _apply_eval_update(latest_run)

    feedback_status = _sync_eval_feedback(
        client=client,
        run=latest_run,
        eval_payload=eval_payload,
    )

    return {
        "updated": True,
        "verified": verified,
        "runtime_run_id": runtime_run_id,
        "task_correct": eval_payload["exact_match"],
        "task_trace_name": task_trace_name or None,
        "feedback_count": feedback_status["feedback_count"],
        "feedback_error": feedback_status["feedback_error"],
    }
