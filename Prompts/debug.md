# LangGraph Refactor Debug Log

## 2026-05-02 20:30 (Asia/Shanghai)

1. Added LangGraph state-machine agent implementation.
- Files: `src/data_agent_baseline/agents/langgraph_agent.py`, `src/data_agent_baseline/agents/runtime.py`
- Result: Replaced loop-based orchestration with graph nodes `profile_context -> plan_action -> execute_action` and conditional routing.

2. Added task-aware skill recommendation module.
- Files: `src/data_agent_baseline/agents/skills.py`
- Result: Introduced reusable skills for tabular, JSON, SQLite, document evidence, and cross-source validation. Skills are injected into agent prompts based on context file profile + question keywords.

3. Integrated framework switch and LangSmith controls in config.
- Files: `src/data_agent_baseline/config.py`, `configs/react_baseline.qwen.yaml`, `configs/react_baseline.local.yaml`
- Result: Added `agent.framework`, `agent.enable_langsmith`, `agent.langsmith_project`. Default framework set to `langgraph` in local configs.

4. Integrated LangGraph execution in runner and graph visualization artifact output.
- Files: `src/data_agent_baseline/run/runner.py`
- Result: Runner now dispatches by framework (`react`/`langgraph`), and writes Mermaid workflow file `graph.mmd` when available.

5. Added dependency support.
- Files: `pyproject.toml`, `uv.lock`
- Result: Added `langgraph` and `langsmith` dependencies and synced environment.

6. Verification.
- Command: `uv run dabench status --config configs/react_baseline.qwen.yaml`
- Command: `uv run dabench run-task task_11 --config configs/react_baseline.qwen.yaml`
- Result: task_11 ran successfully with prediction output and graph artifact generated at:
  - `artifacts/runs/20260502T122647Z/task_11/prediction.csv`
  - `artifacts/runs/20260502T122647Z/task_11/graph.mmd`

7. CLI text update for framework neutrality.
- Files: `src/data_agent_baseline/cli.py`
- Result: Help text now states "configured agent framework" instead of hardcoded ReAct wording.

8. Connection diagnostics hardening.
- Files: `src/data_agent_baseline/agents/model.py`
- Result: Reused a persistent OpenAI client to reduce repeated TCP handshakes; enhanced connection error messages with root cause and active proxy env hints.

9. Added mandatory global format-guardrail prompt.
- Files: `src/data_agent_baseline/Prompts/format_guardrails.md`, `src/data_agent_baseline/agents/prompt.py`
- Result: Every task now injects a shared baseline constraint prompt that enforces JSON action schema, strict `action_input` object format, and strict output column-name rules for scoring.

10. Added LangSmith run-monitor and failed-run cleanup workflow.
- Files: `Visualize/langsmith_monitor.py`, `Visualize/README.md`
- Result: Supports syncing local `artifacts/runs` to LangSmith with per-run/per-task traces, adds filterable tags from `evaluation_details.csv`, includes difficulty grouping tags, and auto-cleans failed runs (move/delete policy with dry-run and force sync support).

11. Added `difficulty_group` tags for LangSmith filtering.
- Files: `Visualize/langsmith_monitor.py`, `Visualize/README.md`
- Result: Each task/run now includes `difficulty_group:easy|difficult|unknown` tags, enabling direct easy-vs-difficult grouping in LangSmith.

12. Refactored monitor script to top-level dict config style.
- Files: `Visualize/langsmith_monitor.py`, `Visualize/README.md`
- Result: Removed CLI argument dependency and switched to editable `MONITOR_CONFIG` with inline comments; default execution now reads parameters from code header only.

13. Added automatic LangSmith grouped tracing inside `run-and-eval`.
- Files: `src/data_agent_baseline/langsmith_group_sync.py`, `src/data_agent_baseline/cli.py`
- Result: Each `run-and-eval` now creates one run-group trace (by local run_id), one child trace per task, one span per agent step from `trace.json`, and one evaluation span per task including correctness fields (`exact_match`, `columns_match`, etc.) and evaluation-derived tags for filtering/grouping.

14. Added metadata-level run-group fields for LangSmith filtering compatibility.
- Files: `src/data_agent_baseline/langsmith_group_sync.py`
- Result: `run_group_id`/`task_id` are now written into metadata and inputs for task/step/evaluation spans, so users can filter grouped runs even when UI does not expose tag filters.

15. Updated LangSmith task trace naming format.
- Files: `src/data_agent_baseline/langsmith_group_sync.py`
- Result: Task trace names now use `YYYY_MM_DD_HH_MM_TaskN` (derived from run_id UTC converted to local timezone), e.g. `2026_05_04_22_56_Task11`.

16. Renamed runtime LangGraph traces to timestamp+task format.
- Files: `src/data_agent_baseline/agents/langgraph_agent.py`
- Result: Live LangSmith traces emitted by LangGraph runtime now use `YYYY_MM_DD_HH_MM_TaskN` via invoke `run_name`, avoiding default "LangGraph" names.

17. Added post-run validation and LangSmith metadata sync to `run-task`.
- Files: `src/data_agent_baseline/cli.py`
- Result: After each `run-task`, the command now auto-runs evaluation against gold outputs, writes `evaluation_summary.json` and `evaluation_details.csv`, prints task-level correctness fields, and (when LangSmith is enabled) uploads the task run + step spans + evaluation metadata automatically.

18. Added runtime-trace evaluation backfill for `run-task`.
- Files: `src/data_agent_baseline/agents/langgraph_agent.py`, `src/data_agent_baseline/langsmith_group_sync.py`, `src/data_agent_baseline/cli.py`
- Result: LangGraph runtime trace now carries a deterministic runtime run_id in metadata. After `run-task` evaluation, the same runtime trace is updated via LangSmith API with evaluation payload (`exact_match`, `columns_match`, row counts, error) in both outputs and metadata.
