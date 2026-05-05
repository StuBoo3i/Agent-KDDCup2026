import json
from pathlib import Path
from time import perf_counter

import typer
from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.table import Table

from data_agent_baseline.benchmark.dataset import DABenchPublicDataset
from data_agent_baseline.config import load_app_config
from data_agent_baseline.evaluation import evaluate_run_outputs, write_evaluation_artifacts
from data_agent_baseline.langsmith_group_sync import (
    sync_run_and_eval_to_langsmith,
    sync_task_eval_to_runtime_trace,
)
from data_agent_baseline.run.runner import TaskRunArtifacts, create_run_output_dir, run_benchmark, run_single_task
from data_agent_baseline.tools.filesystem import list_context_tree

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIGS_DIR = PROJECT_ROOT / "configs"
DATA_DIR = PROJECT_ROOT / "data"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
ARTIFACT_RUNS_DIR = ARTIFACTS_DIR / "runs"

app = typer.Typer(add_completion=False, no_args_is_help=False)
console = Console()


def _status_value(path: Path) -> str:
    return "present" if path.exists() else "missing"


def _format_compact_rate(completed_count: int, elapsed_seconds: float) -> str:
    if completed_count <= 0 or elapsed_seconds <= 0:
        return "rate=0.0 task/min"
    return f"rate={(completed_count / elapsed_seconds) * 60:.1f} task/min"


def _format_last_task(artifact: TaskRunArtifacts | None) -> str:
    if artifact is None:
        return "last=-"
    status = "ok" if artifact.succeeded else "fail"
    return f"last={artifact.task_id} ({status})"


def _build_compact_progress_fields(
    *,
    completed_count: int,
    succeeded_count: int,
    failed_count: int,
    task_total: int,
    max_workers: int,
    elapsed_seconds: float,
    last_artifact: TaskRunArtifacts | None,
) -> dict[str, str]:
    remaining_count = max(task_total - completed_count, 0)
    running_count = min(max_workers, remaining_count)
    queued_count = max(remaining_count - running_count, 0)
    return {
        "ok": str(succeeded_count),
        "fail": str(failed_count),
        "run": str(running_count),
        "queue": str(queued_count),
        "speed": _format_compact_rate(completed_count, elapsed_seconds),
        "last": _format_last_task(last_artifact),
    }


@app.callback()
def cli() -> None:
    """Utilities for working with the local DABench baseline project."""


@app.command()
def status(
    config: Path = typer.Option(..., exists=True, dir_okay=False, help="YAML config path."),
) -> None:
    """Show the local project layout and public dataset presence."""
    app_config = load_app_config(config)
    config_path = config.resolve()
    public_dataset = DABenchPublicDataset(app_config.dataset.root_path)

    table = Table(title="DABench Baseline Status")
    table.add_column("Item")
    table.add_column("Path")
    table.add_column("State")

    table.add_row("project_root", str(PROJECT_ROOT), "ready")
    table.add_row("data_dir", str(DATA_DIR), _status_value(DATA_DIR))
    table.add_row("configs_dir", str(CONFIGS_DIR), _status_value(CONFIGS_DIR))
    table.add_row("artifacts_dir", str(ARTIFACTS_DIR), _status_value(ARTIFACTS_DIR))
    table.add_row("runs_dir", str(ARTIFACT_RUNS_DIR), _status_value(ARTIFACT_RUNS_DIR))
    table.add_row("dataset_root", str(app_config.dataset.root_path), _status_value(app_config.dataset.root_path))
    table.add_row("config_path", str(config_path), _status_value(config_path))
    table.add_row("agent_framework", app_config.agent.framework, "ready")
    table.add_row(
        "langsmith",
        app_config.agent.langsmith_project if app_config.agent.enable_langsmith else "disabled",
        "enabled" if app_config.agent.enable_langsmith else "disabled",
    )

    console.print(table)

    if public_dataset.exists:
        console.print(f"Public tasks: {len(public_dataset.list_task_ids())}")
        counts = public_dataset.task_counts()
        if counts:
            rendered_counts = ", ".join(
                f"{difficulty}={count}" for difficulty, count in sorted(counts.items())
            )
            console.print(f"Public task counts: {rendered_counts}")


@app.command("inspect-task")
def inspect_task(
    task_id: str,
    config: Path = typer.Option(..., exists=True, dir_okay=False, help="YAML config path."),
) -> None:
    """Show task metadata and available context files."""
    app_config = load_app_config(config)
    dataset = DABenchPublicDataset(app_config.dataset.root_path)
    task = dataset.get_task(task_id)
    console.print(f"Task: {task.task_id}")
    console.print(f"Difficulty: {task.difficulty}")
    console.print(f"Question: {task.question}")
    context_listing = list_context_tree(task)
    table = Table(title=f"Context Files for {task.task_id}")
    table.add_column("Path")
    table.add_column("Kind")
    table.add_column("Size")
    for entry in context_listing["entries"]:
        table.add_row(str(entry["path"]), str(entry["kind"]), str(entry["size"] or ""))
    console.print(table)


@app.command("run-task")
def run_task_command(
    task_id: str,
    config: Path = typer.Option(..., exists=True, dir_okay=False, help="YAML config path."),
    gold_root: Path = typer.Option(
        PROJECT_ROOT / "data" / "public" / "output",
        exists=True,
        file_okay=False,
        dir_okay=True,
        help="Directory containing task_<id>/gold.csv files for post-run validation.",
    ),
) -> None:
    """Run the configured agent framework on one task."""
    app_config = load_app_config(config)
    try:
        _, run_output_dir = create_run_output_dir(app_config.run.output_dir, run_id=app_config.run.run_id)
    except (ValueError, FileExistsError) as exc:
        raise typer.BadParameter(str(exc), param_hint="run.run_id") from exc
    artifacts = run_single_task(task_id=task_id, config=app_config, run_output_dir=run_output_dir)

    console.print(f"Run output: {run_output_dir}")
    console.print(f"Task output: {artifacts.task_output_dir}")
    if artifacts.prediction_csv_path is not None:
        console.print(f"Prediction CSV: {artifacts.prediction_csv_path}")
    else:
        console.print("Prediction CSV: not generated")
    if artifacts.failure_reason is not None:
        console.print(f"Failure: {artifacts.failure_reason}")

    # Auto validation right after run-task to compare with public gold outputs.
    summary = evaluate_run_outputs(run_output_dir, gold_root.resolve())
    summary_json_path, details_csv_path = write_evaluation_artifacts(summary, run_output_dir)
    console.print(f"Evaluation summary JSON: {summary_json_path}")
    console.print(f"Evaluation details CSV: {details_csv_path}")
    task_eval = next((item for item in summary.get("tasks", []) if item.get("task_id") == task_id), None)
    if task_eval is not None:
        console.print(
            "Task eval: "
            f"exact_match={task_eval.get('exact_match')} "
            f"unordered_row_match={task_eval.get('unordered_row_match')} "
            f"columns_match={task_eval.get('columns_match')} "
            f"has_prediction={task_eval.get('has_prediction')} "
            f"error={task_eval.get('error') or '-'}"
        )

    if app_config.agent.enable_langsmith:
        try:
            trace_payload = json.loads(artifacts.trace_path.read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001
            console.print(f"[yellow]LangSmith trace read failed:[/yellow] {exc}")
            trace_payload = None

        if task_eval is not None and isinstance(trace_payload, dict):
            try:
                runtime_sync = sync_task_eval_to_runtime_trace(
                    trace_payload=trace_payload,
                    task_eval_row=task_eval,
                )
                if runtime_sync.get("updated"):
                    console.print(
                        "LangSmith runtime trace updated: "
                        f"run_id={runtime_sync.get('runtime_run_id')} "
                        f"task_correct={runtime_sync.get('task_correct')} "
                        f"verified={runtime_sync.get('verified')} "
                        f"feedback_count={runtime_sync.get('feedback_count')}"
                    )
                    if runtime_sync.get("feedback_error"):
                        console.print(
                            "[yellow]LangSmith feedback warning:[/yellow] "
                            f"{runtime_sync.get('feedback_error')}"
                        )
                else:
                    console.print(
                        "[yellow]LangSmith runtime update skipped:[/yellow] "
                        f"{runtime_sync.get('reason')} "
                        f"run_id={runtime_sync.get('runtime_run_id') or '-'} "
                        f"trace_name={runtime_sync.get('task_trace_name') or '-'} "
                        f"error={runtime_sync.get('error') or '-'}"
                    )
            except Exception as exc:  # noqa: BLE001
                console.print(f"[yellow]LangSmith runtime update failed:[/yellow] {exc}")

        try:
            sync_result = sync_run_and_eval_to_langsmith(
                app_config=app_config,
                run_output_dir=run_output_dir,
                artifacts=[artifacts],
                evaluation_summary=summary,
                difficulty_filters=None,
            )
            console.print(
                "LangSmith sync: "
                f"project={sync_result['project_name']} "
                f"run_group_id={sync_result['run_group_id']} "
                f"task_runs={sync_result['task_run_count']} "
                f"step_runs={sync_result['step_run_count']}"
            )
        except Exception as exc:  # noqa: BLE001
            console.print(f"[yellow]LangSmith grouped sync failed:[/yellow] {exc}")


@app.command("run-benchmark")
def run_benchmark_command(
    config: Path = typer.Option(..., exists=True, dir_okay=False, help="YAML config path."),
    limit: int | None = typer.Option(None, min=1, help="Maximum number of tasks to run."),
    difficulty: list[str] = typer.Option(
        [],
        "--difficulty",
        help="Filter by difficulty. Repeat the option for multiple values.",
    ),
) -> None:
    """Run the configured agent framework on multiple tasks from the config selection."""
    app_config = load_app_config(config)
    dataset = DABenchPublicDataset(app_config.dataset.root_path)
    tasks = dataset.iter_tasks(difficulties=list(difficulty) or None)
    task_total = len(tasks)
    if limit is not None:
        task_total = min(task_total, limit)
    effective_workers = app_config.run.max_workers

    progress_columns = [
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TextColumn("[dim]|[/dim]"),
        TextColumn("[green]ok={task.fields[ok]}[/green]"),
        TextColumn("[red]fail={task.fields[fail]}[/red]"),
        TextColumn("[cyan]run={task.fields[run]}[/cyan]"),
        TextColumn("[yellow]queue={task.fields[queue]}[/yellow]"),
        TextColumn("[dim]|[/dim]"),
        TextColumn("{task.fields[speed]}"),
        TextColumn("[dim]| elapsed[/dim]"),
        TimeElapsedColumn(),
        TextColumn("[dim]| eta[/dim]"),
        TimeRemainingColumn(),
        TextColumn("[dim]|[/dim]"),
        TextColumn("{task.fields[last]}"),
    ]
    with Progress(*progress_columns, console=console) as progress:
        progress_task_id = progress.add_task(
            "Benchmark",
            total=task_total,
            completed=0,
            **_build_compact_progress_fields(
                completed_count=0,
                succeeded_count=0,
                failed_count=0,
                task_total=task_total,
                max_workers=effective_workers,
                elapsed_seconds=0.0,
                last_artifact=None,
            ),
        )

        completion_count = 0
        succeeded_count = 0
        failed_count = 0
        start_time = perf_counter()

        def on_task_complete(artifact) -> None:
            nonlocal completion_count, succeeded_count, failed_count
            completion_count += 1
            if artifact.succeeded:
                succeeded_count += 1
            else:
                failed_count += 1
            progress.update(
                progress_task_id,
                completed=completion_count,
                description="Benchmark",
                refresh=True,
                **_build_compact_progress_fields(
                    completed_count=completion_count,
                    succeeded_count=succeeded_count,
                    failed_count=failed_count,
                    task_total=task_total,
                    max_workers=effective_workers,
                    elapsed_seconds=perf_counter() - start_time,
                    last_artifact=artifact,
                ),
            )

        try:
            run_output_dir, artifacts = run_benchmark(
                config=app_config,
                limit=limit,
                difficulties=list(difficulty) or None,
                progress_callback=on_task_complete,
            )
        except (ValueError, FileExistsError) as exc:
            raise typer.BadParameter(str(exc), param_hint="run.run_id") from exc
        progress.update(
            progress_task_id,
            completed=task_total,
            description="Benchmark",
            refresh=True,
            **_build_compact_progress_fields(
                completed_count=task_total,
                succeeded_count=succeeded_count,
                failed_count=failed_count,
                task_total=task_total,
                max_workers=effective_workers,
                elapsed_seconds=perf_counter() - start_time,
                last_artifact=artifacts[-1] if artifacts else None,
            ),
        )
    console.print(f"Run output: {run_output_dir}")
    console.print(f"Tasks attempted: {len(artifacts)}")
    console.print(f"Succeeded tasks: {sum(1 for item in artifacts if item.succeeded)}")


@app.command("run-and-eval")
def run_and_evaluate_command(
    config: Path = typer.Option(..., exists=True, dir_okay=False, help="YAML config path."),
    limit: int | None = typer.Option(None, min=1, help="Maximum number of tasks to run."),
    difficulty: list[str] = typer.Option(
        [],
        "--difficulty",
        help="Filter by difficulty. Repeat the option for multiple values.",
    ),
    gold_root: Path = typer.Option(
        PROJECT_ROOT / "data" / "public" / "output",
        exists=True,
        file_okay=False,
        dir_okay=True,
        help="Directory containing task_<id>/gold.csv files.",
    ),
) -> None:
    """Run benchmark and evaluate prediction.csv against public gold.csv."""
    app_config = load_app_config(config)
    run_output_dir, artifacts = run_benchmark(
        config=app_config,
        limit=limit,
        difficulties=list(difficulty) or None,
    )
    console.print(f"Run output: {run_output_dir}")
    console.print(f"Tasks attempted: {len(artifacts)}")
    console.print(f"Succeeded tasks: {sum(1 for item in artifacts if item.succeeded)}")

    summary = evaluate_run_outputs(run_output_dir, gold_root.resolve())
    summary_json_path, details_csv_path = write_evaluation_artifacts(summary, run_output_dir)

    console.print(f"Evaluation summary JSON: {summary_json_path}")
    console.print(f"Evaluation details CSV: {details_csv_path}")
    console.print(
        "Exact match: "
        f"{summary['exact_match_count']}/{summary['task_count_evaluated']} "
        f"({summary['exact_match_rate']:.2%})"
    )
    console.print(
        "Unordered-row match: "
        f"{summary['unordered_row_match_count']}/{summary['task_count_evaluated']} "
        f"({summary['unordered_row_match_rate']:.2%})"
    )
    console.print(
        f"Missing prediction: {summary['missing_prediction_count']}, "
        f"Missing gold: {summary['missing_gold_count']}, Errors: {summary['error_count']}"
    )

    if app_config.agent.enable_langsmith:
        try:
            sync_result = sync_run_and_eval_to_langsmith(
                app_config=app_config,
                run_output_dir=run_output_dir,
                artifacts=artifacts,
                evaluation_summary=summary,
                difficulty_filters=list(difficulty) or None,
            )
            console.print(
                "LangSmith sync: "
                f"project={sync_result['project_name']} "
                f"run_group_id={sync_result['run_group_id']} "
                f"task_runs={sync_result['task_run_count']} "
                f"step_runs={sync_result['step_run_count']}"
            )
        except Exception as exc:  # noqa: BLE001
            console.print(f"[yellow]LangSmith sync failed:[/yellow] {exc}")


def main() -> None:
    app()
