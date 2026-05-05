# Visualize Tools

This folder has two tools:

1. `langsmith_monitor.py`: sync local runs to LangSmith.
2. `run_dashboard.py`: local web dashboard for checking run/eval/trace/graph.mmd.

## 1) Local run dashboard (frontend)

### Features

- Scan `artifacts/runs/<run_id>` automatically.
- One-click cleanup of failed runs.
- Show task-level evaluation results (`evaluation_details.csv` / `evaluation_summary.json`).
- LangSmith-style trace waterfall view from `trace.json` steps.
- Render `graph.mmd` with Mermaid.

### Start

1. Edit config at top of `Visualize/run_dashboard.py` (dictionary style, no CLI args).
2. Run:

```powershell
uv run python Visualize\run_dashboard.py
```

3. Open browser:

```text
http://127.0.0.1:8765
```

### Important config keys

- `runs_dir`: path to local run artifacts.
- `auto_cleanup.startup`: cleanup failed runs when server starts.
- `auto_cleanup.page_load`: cleanup failed runs when page opens.
- `auto_cleanup.failed_rule`:
  - `no_successful_tasks` (safer)
  - `any_task_failed` (more aggressive)

## 2) LangSmith monitor

### Start

1. Open `Visualize/langsmith_monitor.py` and edit `MONITOR_CONFIG`.
2. Set env vars:

```powershell
$env:LANGSMITH_API_KEY="your_langsmith_api_key"
$env:LANGSMITH_TRACING="true"
```

3. Run:

```powershell
uv run python Visualize\langsmith_monitor.py
```
