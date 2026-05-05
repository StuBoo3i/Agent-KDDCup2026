from __future__ import annotations

import csv
import json
import shutil
import urllib.parse
from datetime import datetime
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any


# Runtime configuration for the dashboard (no CLI args required).
CONFIG: dict[str, Any] = {
    # Root directory that contains run folders such as 20260505T003611Z.
    "runs_dir": Path(__file__).resolve().parents[1] / "artifacts" / "runs",
    # Web server host/port.
    "host": "127.0.0.1",
    "port": 8765,
    # Auto cleanup behavior:
    # - startup: remove failed runs once when server starts.
    # - page_load: remove failed runs when frontend opens.
    "auto_cleanup": {
        "startup": False,
        "page_load": False,
        # Rules: "no_successful_tasks" | "any_task_failed"
        "failed_rule": "no_successful_tasks",
    },
    # UI payload limits.
    "preview": {
        "prediction_rows": 30,
        "raw_response_chars": 1600,
        "observation_chars": 2200,
        "prompt_message_chars": 1200,
        "prompt_messages_per_step": 24,
    },
}


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WEB_ROOT = Path(__file__).resolve().parent / "web"
RUNS_DIR = Path(CONFIG["runs_dir"]).resolve()
RUN_REVIEWS_PATH = RUNS_DIR / ".run_reviews.json"


def _boolish(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if value is None:
        return None
    text = str(value).strip().lower()
    if text in {"true", "1", "yes", "y"}:
        return True
    if text in {"false", "0", "no", "n"}:
        return False
    return None


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_eval_details(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    result: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            task_id = str(row.get("task_id", "")).strip()
            if not task_id:
                continue
            row["has_prediction"] = _boolish(row.get("has_prediction"))
            row["has_gold"] = _boolish(row.get("has_gold"))
            row["columns_match"] = _boolish(row.get("columns_match"))
            row["exact_match"] = _boolish(row.get("exact_match"))
            row["unordered_row_match"] = _boolish(row.get("unordered_row_match"))
            result[task_id] = row
    return result


def _truncate(text: Any, max_chars: int) -> str:
    raw = str(text)
    if len(raw) <= max_chars:
        return raw
    return raw[:max_chars] + "\n...[truncated]"


def _parse_iso_datetime(value: Any) -> datetime | None:
    if not value:
        return None
    text = str(value).strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        return datetime.fromisoformat(text)
    except ValueError:
        return None


def _safe_run_sort_key(name: str) -> tuple[int, str]:
    if name.isdigit():
        return int(name), name
    if len(name) == 16 and "T" in name and name.endswith("Z"):
        return 10**14, name
    return 0, name


def _safe_task_sort_key(task_id: str) -> tuple[int, str]:
    if task_id.startswith("task_"):
        suffix = task_id.split("_", 1)[1]
        if suffix.isdigit():
            return int(suffix), task_id
    return 10**9, task_id


def _run_dirs() -> list[Path]:
    if not RUNS_DIR.exists():
        return []
    return sorted(
        [path for path in RUNS_DIR.iterdir() if path.is_dir()],
        key=lambda path: _safe_run_sort_key(path.name),
        reverse=True,
    )


def _load_run_reviews() -> dict[str, dict[str, Any]]:
    if not RUN_REVIEWS_PATH.exists():
        return {}
    try:
        payload = _read_json(RUN_REVIEWS_PATH)
    except Exception:  # noqa: BLE001
        return {}
    if not isinstance(payload, dict):
        return {}
    result: dict[str, dict[str, Any]] = {}
    for run_id, review in payload.items():
        if not isinstance(run_id, str) or not isinstance(review, dict):
            continue
        result[run_id] = review
    return result


def _save_run_reviews(reviews: dict[str, dict[str, Any]]) -> None:
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    RUN_REVIEWS_PATH.write_text(
        json.dumps(reviews, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _normalize_review_payload(payload: dict[str, Any]) -> dict[str, Any]:
    score_raw = payload.get("score")
    score: int | None = None
    if isinstance(score_raw, int) and not isinstance(score_raw, bool) and 1 <= score_raw <= 5:
        score = score_raw
    elif isinstance(score_raw, str) and score_raw.strip().isdigit():
        parsed = int(score_raw.strip())
        if 1 <= parsed <= 5:
            score = parsed
    label = str(payload.get("label", "")).strip()[:48]
    note = str(payload.get("note", "")).strip()[:4000]
    return {
        "score": score,
        "label": label,
        "note": note,
        "updated_at": datetime.now().isoformat(timespec="seconds"),
    }


def _review_summary(review: dict[str, Any], *, with_note: bool = True) -> dict[str, Any]:
    summary = {
        "score": review.get("score"),
        "label": review.get("label", ""),
        "updated_at": review.get("updated_at", ""),
    }
    if with_note:
        summary["note"] = review.get("note", "")
    return summary


def _scan_run(
    run_dir: Path,
    run_reviews: dict[str, dict[str, Any]] | None = None,
    *,
    include_review_note: bool = False,
) -> dict[str, Any]:
    eval_summary_path = run_dir / "evaluation_summary.json"
    eval_details_path = run_dir / "evaluation_details.csv"
    eval_summary = _read_json(eval_summary_path) if eval_summary_path.exists() else {}
    eval_by_task = _read_eval_details(eval_details_path)

    task_dirs = sorted(
        [path for path in run_dir.iterdir() if path.is_dir() and path.name.startswith("task_")],
        key=lambda path: _safe_task_sort_key(path.name),
    )
    tasks: list[dict[str, Any]] = []
    succeeded_count = 0

    for task_dir in task_dirs:
        task_id = task_dir.name
        trace_path = task_dir / "trace.json"
        trace_payload: dict[str, Any] = _read_json(trace_path) if trace_path.exists() else {}
        succeeded = bool(trace_payload.get("succeeded"))
        if succeeded:
            succeeded_count += 1
        failure_reason = trace_payload.get("failure_reason")
        steps = trace_payload.get("steps", [])
        eval_row = eval_by_task.get(task_id, {})
        tasks.append(
            {
                "task_id": task_id,
                "task_dir": str(task_dir),
                "succeeded": succeeded,
                "failure_reason": failure_reason,
                "step_count": len(steps) if isinstance(steps, list) else 0,
                "has_trace": trace_path.exists(),
                "has_prediction_csv": (task_dir / "prediction.csv").exists(),
                "has_graph_mmd": (task_dir / "graph.mmd").exists(),
                "exact_match": eval_row.get("exact_match"),
                "columns_match": eval_row.get("columns_match"),
                "unordered_row_match": eval_row.get("unordered_row_match"),
                "has_prediction_eval": eval_row.get("has_prediction"),
                "eval_error": eval_row.get("error"),
            }
        )

    task_count = len(tasks)
    failed_count = max(0, task_count - succeeded_count)
    if task_count == 0:
        run_status = "failed"
    elif succeeded_count == task_count:
        run_status = "success"
    elif succeeded_count == 0:
        run_status = "failed"
    else:
        run_status = "partial"

    reviews = run_reviews or {}
    review = reviews.get(run_dir.name, {})

    return {
        "run_id": run_dir.name,
        "run_dir": str(run_dir),
        "modified_at": datetime.fromtimestamp(run_dir.stat().st_mtime).isoformat(timespec="seconds"),
        "task_count": task_count,
        "task_succeeded_count": succeeded_count,
        "task_failed_count": failed_count,
        "run_status": run_status,
        "has_evaluation": bool(eval_summary),
        "eval_summary": {
            "task_count_evaluated": eval_summary.get("task_count_evaluated"),
            "exact_match_count": eval_summary.get("exact_match_count"),
            "exact_match_rate": eval_summary.get("exact_match_rate"),
            "unordered_row_match_count": eval_summary.get("unordered_row_match_count"),
            "unordered_row_match_rate": eval_summary.get("unordered_row_match_rate"),
            "missing_prediction_count": eval_summary.get("missing_prediction_count"),
            "error_count": eval_summary.get("error_count"),
        },
        "review": _review_summary(review, with_note=include_review_note)
        if isinstance(review, dict)
        else _review_summary({}, with_note=include_review_note),
        "tasks": tasks,
    }


def _scan_all_runs() -> list[dict[str, Any]]:
    reviews = _load_run_reviews()
    return [_scan_run(path, reviews) for path in _run_dirs()]


def _should_delete_failed_run(run_summary: dict[str, Any]) -> bool:
    rule = str(CONFIG["auto_cleanup"].get("failed_rule", "no_successful_tasks")).strip()
    task_count = int(run_summary.get("task_count", 0) or 0)
    success_count = int(run_summary.get("task_succeeded_count", 0) or 0)
    failed_count = int(run_summary.get("task_failed_count", 0) or 0)
    if task_count == 0:
        return True
    if rule == "any_task_failed":
        return failed_count > 0
    return success_count == 0


def _safe_delete_run(run_dir: Path) -> tuple[bool, str]:
    resolved = run_dir.resolve()
    if resolved.parent != RUNS_DIR:
        return False, "run path is outside runs_dir"
    if not resolved.exists() or not resolved.is_dir():
        return False, "run path not found"
    try:
        shutil.rmtree(resolved)
    except Exception as exc:  # noqa: BLE001
        return False, f"{type(exc).__name__}: {exc}"
    return True, ""


def _cleanup_failed_runs() -> dict[str, Any]:
    deleted: list[str] = []
    skipped: list[str] = []
    reviews = _load_run_reviews()
    for run_summary in _scan_all_runs():
        if not _should_delete_failed_run(run_summary):
            continue
        run_path = Path(run_summary["run_dir"])
        deleted_ok, _ = _safe_delete_run(run_path)
        if deleted_ok:
            deleted.append(run_summary["run_id"])
            reviews.pop(run_summary["run_id"], None)
        else:
            skipped.append(run_summary["run_id"])
    if deleted:
        _save_run_reviews(reviews)
    return {"deleted": deleted, "skipped": skipped, "deleted_count": len(deleted)}


def _delete_run_with_review(run_id: str) -> dict[str, Any]:
    run_path = RUNS_DIR / run_id
    if not run_path.exists() or not run_path.is_dir():
        return {"deleted": False, "reason": "run not found"}
    deleted_ok, reason = _safe_delete_run(run_path)
    if not deleted_ok:
        return {"deleted": False, "reason": reason or "run delete rejected by safety checks"}
    reviews = _load_run_reviews()
    reviews.pop(run_id, None)
    _save_run_reviews(reviews)
    return {"deleted": True, "run_id": run_id}


def _read_prediction_preview(path: Path, max_rows: int) -> dict[str, Any]:
    if not path.exists():
        return {"columns": [], "rows": []}
    rows: list[list[str]] = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        for index, row in enumerate(reader):
            rows.append(list(row))
            if index >= max_rows:
                break
    if not rows:
        return {"columns": [], "rows": []}
    return {"columns": rows[0], "rows": rows[1:]}


def _resolve_gold_csv_path(run_dir: Path, task_id: str) -> Path:
    candidate_roots: list[Path] = []
    eval_summary_path = run_dir / "evaluation_summary.json"
    if eval_summary_path.exists():
        try:
            summary = _read_json(eval_summary_path)
            raw_root = str(summary.get("gold_output_root", "")).strip()
            if raw_root:
                candidate_roots.append(Path(raw_root))
        except Exception:  # noqa: BLE001
            pass

    candidate_roots.append(PROJECT_ROOT / "data" / "public" / "output")
    for root in candidate_roots:
        candidate = root / task_id / "gold.csv"
        if candidate.exists():
            return candidate
    return candidate_roots[-1] / task_id / "gold.csv"


def _build_waterfall(steps: list[dict[str, Any]], total_seconds: float | None) -> list[dict[str, Any]]:
    if not steps:
        return []

    def _prompt_preview(step: dict[str, Any]) -> tuple[list[dict[str, str]], int]:
        raw_messages = step.get("prompt_messages", [])
        if not isinstance(raw_messages, list):
            return [], 0
        max_messages = int(CONFIG["preview"]["prompt_messages_per_step"])
        max_chars = int(CONFIG["preview"]["prompt_message_chars"])
        rendered: list[dict[str, str]] = []
        for message in raw_messages[:max_messages]:
            if not isinstance(message, dict):
                continue
            role = str(message.get("role", "")).strip() or "unknown"
            content = _truncate(message.get("content", ""), max_chars)
            rendered.append({"role": role, "content": content})
        return rendered, len(raw_messages)

    parsed_ranges: list[tuple[datetime | None, datetime | None, float | None]] = []
    has_real_timestamps = True
    for step in steps:
        started_at = _parse_iso_datetime(step.get("started_at_utc"))
        completed_at = _parse_iso_datetime(step.get("completed_at_utc"))
        elapsed_raw = step.get("elapsed_seconds")
        elapsed_seconds = float(elapsed_raw) if isinstance(elapsed_raw, (int, float)) else None
        parsed_ranges.append((started_at, completed_at, elapsed_seconds))
        if started_at is None or completed_at is None:
            has_real_timestamps = False

    if has_real_timestamps:
        base_start = min(item[0] for item in parsed_ranges if item[0] is not None)
        timeline: list[dict[str, Any]] = []
        for step, (started_at, completed_at, elapsed_seconds) in zip(steps, parsed_ranges, strict=False):
            assert started_at is not None and completed_at is not None
            start = max(0.0, (started_at - base_start).total_seconds())
            duration = max(0.0, (completed_at - started_at).total_seconds())
            if duration <= 0 and isinstance(elapsed_seconds, float):
                duration = max(0.0, elapsed_seconds)
            end = start + duration
            prompt_messages_preview, prompt_message_count = _prompt_preview(step)
            timeline.append(
                {
                    "step_index": step.get("step_index"),
                    "action": step.get("action"),
                    "ok": bool(step.get("ok")),
                    "start_s": round(start, 3),
                    "duration_s": round(duration, 3),
                    "end_s": round(end, 3),
                    "started_at_utc": step.get("started_at_utc"),
                    "completed_at_utc": step.get("completed_at_utc"),
                    "thought": step.get("thought", ""),
                    "action_input": step.get("action_input", {}),
                    "raw_response_preview": _truncate(
                        step.get("raw_response", ""),
                        CONFIG["preview"]["raw_response_chars"],
                    ),
                    "observation_preview": _truncate(
                        json.dumps(step.get("observation", {}), ensure_ascii=False, indent=2),
                        CONFIG["preview"]["observation_chars"],
                    ),
                    "prompt_messages_preview": prompt_messages_preview,
                    "prompt_message_count": prompt_message_count,
                }
            )
        return timeline

    weights: list[float] = []
    for step in steps:
        raw_len = len(str(step.get("raw_response", "")))
        obs_len = len(json.dumps(step.get("observation", {}), ensure_ascii=False))
        weight = 1.0 + min(6.0, raw_len / 1400.0 + obs_len / 2400.0)
        weights.append(weight)

    total_weight = sum(weights) or float(len(steps))
    total = float(total_seconds) if total_seconds and total_seconds > 0 else total_weight
    cursor = 0.0
    timeline: list[dict[str, Any]] = []
    for step, weight in zip(steps, weights, strict=False):
        duration = total * (weight / total_weight)
        start = cursor
        end = start + duration
        cursor = end
        prompt_messages_preview, prompt_message_count = _prompt_preview(step)
        timeline.append(
            {
                "step_index": step.get("step_index"),
                "action": step.get("action"),
                "ok": bool(step.get("ok")),
                "start_s": round(start, 3),
                "duration_s": round(duration, 3),
                "end_s": round(end, 3),
                "thought": step.get("thought", ""),
                "action_input": step.get("action_input", {}),
                "raw_response_preview": _truncate(step.get("raw_response", ""), CONFIG["preview"]["raw_response_chars"]),
                "observation_preview": _truncate(
                    json.dumps(step.get("observation", {}), ensure_ascii=False, indent=2),
                    CONFIG["preview"]["observation_chars"],
                ),
                "prompt_messages_preview": prompt_messages_preview,
                "prompt_message_count": prompt_message_count,
            }
        )
    return timeline


def _task_detail(run_id: str, task_id: str) -> dict[str, Any] | None:
    run_dir = RUNS_DIR / run_id
    task_dir = run_dir / task_id
    if not run_dir.exists() or not task_dir.exists():
        return None

    trace_path = task_dir / "trace.json"
    trace_payload = _read_json(trace_path) if trace_path.exists() else {}
    eval_by_task = _read_eval_details(run_dir / "evaluation_details.csv")
    eval_row = eval_by_task.get(task_id, {})

    steps = trace_payload.get("steps", [])
    if not isinstance(steps, list):
        steps = []
    timeline = _build_waterfall(steps, trace_payload.get("e2e_elapsed_seconds"))
    traced_step_seconds = round(sum(float(item.get("duration_s", 0) or 0) for item in timeline), 3)
    e2e_seconds = trace_payload.get("e2e_elapsed_seconds")
    non_step_overhead_seconds = None
    if isinstance(e2e_seconds, (int, float)):
        non_step_overhead_seconds = round(float(e2e_seconds) - traced_step_seconds, 3)

    graph_mmd = ""
    graph_path = task_dir / "graph.mmd"
    if graph_path.exists():
        graph_mmd = graph_path.read_text(encoding="utf-8")

    prediction_path = task_dir / "prediction.csv"
    gold_path = _resolve_gold_csv_path(run_dir, task_id)
    prediction_preview = _read_prediction_preview(prediction_path, CONFIG["preview"]["prediction_rows"])
    gold_preview = _read_prediction_preview(gold_path, CONFIG["preview"]["prediction_rows"])
    review = _load_run_reviews().get(run_id, {})
    return {
        "run_id": run_id,
        "task_id": task_id,
        "task_dir": str(task_dir),
        "succeeded": bool(trace_payload.get("succeeded")),
        "failure_reason": trace_payload.get("failure_reason"),
        "e2e_elapsed_seconds": e2e_seconds,
        "step_count": len(steps),
        "traced_step_seconds": traced_step_seconds,
        "non_step_overhead_seconds": non_step_overhead_seconds,
        "eval": eval_row,
        "timeline": timeline,
        "graph_mmd": graph_mmd,
        "prediction_csv_path": str(prediction_path) if prediction_path.exists() else None,
        "gold_csv_path": str(gold_path) if gold_path.exists() else None,
        "prediction_preview": prediction_preview,
        "gold_preview": gold_preview,
        "review": _review_summary(review, with_note=True) if isinstance(review, dict) else _review_summary({}, with_note=True),
    }


class DashboardHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, directory=str(WEB_ROOT), **kwargs)

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A003
        message = format % args
        print(f"[dashboard] {message}")

    def _json_response(self, payload: dict[str, Any], status: int = HTTPStatus.OK) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_json_body(self) -> dict[str, Any]:
        content_length = self.headers.get("Content-Length", "0")
        try:
            body_length = max(0, int(content_length))
        except ValueError as exc:
            raise ValueError("invalid Content-Length header") from exc

        if body_length == 0:
            return {}

        raw = self.rfile.read(body_length)
        if not raw:
            return {}
        try:
            payload = json.loads(raw.decode("utf-8"))
        except Exception as exc:  # noqa: BLE001
            raise ValueError("invalid JSON body") from exc
        if not isinstance(payload, dict):
            raise ValueError("request body must be a JSON object")
        return payload

    def do_OPTIONS(self) -> None:  # noqa: N802
        self.send_response(HTTPStatus.NO_CONTENT)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET,POST,OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self) -> None:  # noqa: N802
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path
        if path == "/api/config":
            self._json_response(
                {
                    "runs_dir": str(RUNS_DIR),
                    "auto_cleanup_on_page_load": bool(CONFIG["auto_cleanup"].get("page_load", False)),
                    "failed_rule": CONFIG["auto_cleanup"].get("failed_rule", "no_successful_tasks"),
                }
            )
            return
        if path == "/api/runs":
            runs = _scan_all_runs()
            self._json_response({"runs": runs, "count": len(runs)})
            return
        if path.startswith("/api/runs/"):
            parts = [segment for segment in path.split("/") if segment]
            # /api/runs/{run_id}
            if len(parts) == 3:
                run_id = urllib.parse.unquote(parts[2])
                run_dir = RUNS_DIR / run_id
                if not run_dir.exists():
                    self._json_response({"error": "run not found"}, status=HTTPStatus.NOT_FOUND)
                    return
                self._json_response({"run": _scan_run(run_dir, _load_run_reviews(), include_review_note=True)})
                return
            # /api/runs/{run_id}/tasks/{task_id}
            if len(parts) == 5 and parts[3] == "tasks":
                run_id = urllib.parse.unquote(parts[2])
                task_id = urllib.parse.unquote(parts[4])
                detail = _task_detail(run_id, task_id)
                if detail is None:
                    self._json_response({"error": "task not found"}, status=HTTPStatus.NOT_FOUND)
                    return
                self._json_response({"task": detail})
                return

        if path == "/":
            self.path = "/index.html"
        super().do_GET()

    def do_POST(self) -> None:  # noqa: N802
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path == "/api/cleanup-failed":
            result = _cleanup_failed_runs()
            self._json_response(result)
            return

        if parsed.path.startswith("/api/runs/"):
            parts = [segment for segment in parsed.path.split("/") if segment]
            # /api/runs/{run_id}/review
            if len(parts) == 4 and parts[3] == "review":
                run_id = urllib.parse.unquote(parts[2])
                run_dir = RUNS_DIR / run_id
                if not run_dir.exists() or not run_dir.is_dir():
                    self._json_response({"error": "run not found"}, status=HTTPStatus.NOT_FOUND)
                    return
                try:
                    payload = self._read_json_body()
                except ValueError as exc:
                    self._json_response({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
                    return
                review = _normalize_review_payload(payload)
                reviews = _load_run_reviews()
                reviews[run_id] = review
                _save_run_reviews(reviews)
                self._json_response({"ok": True, "run_id": run_id, "review": _review_summary(review, with_note=True)})
                return

            # /api/runs/{run_id}/delete
            if len(parts) == 4 and parts[3] == "delete":
                run_id = urllib.parse.unquote(parts[2])
                result = _delete_run_with_review(run_id)
                if not result.get("deleted"):
                    status = HTTPStatus.NOT_FOUND if result.get("reason") == "run not found" else HTTPStatus.BAD_REQUEST
                    self._json_response(result, status=status)
                    return
                self._json_response(result)
                return

        self._json_response({"error": "unknown endpoint"}, status=HTTPStatus.NOT_FOUND)


def main() -> None:
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    WEB_ROOT.mkdir(parents=True, exist_ok=True)

    if bool(CONFIG["auto_cleanup"].get("startup", False)):
        cleanup_result = _cleanup_failed_runs()
        print(
            f"[dashboard] startup cleanup finished: "
            f"deleted={cleanup_result['deleted_count']} skipped={len(cleanup_result['skipped'])}"
        )

    host = str(CONFIG["host"])
    port = int(CONFIG["port"])
    with ThreadingHTTPServer((host, port), DashboardHandler) as server:
        print(f"[dashboard] runs_dir: {RUNS_DIR}")
        print(f"[dashboard] open: http://{host}:{port}")
        server.serve_forever()


if __name__ == "__main__":
    main()
