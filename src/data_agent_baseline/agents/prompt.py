from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

from data_agent_baseline.benchmark.schema import PublicTask


REACT_SYSTEM_PROMPT = """
You are a ReAct-style data agent.

You are solving a task from a public dataset. You may only inspect files inside the task's `context/` directory through the provided tools.

Tips:
1. For each task, read knowledge.md first to find relevant information for the task.


Rules:
1. Use tools to inspect the available context before answering.
2. Base your answer only on information you can observe through the provided tools.
3. The task is complete only when you call the `answer` tool.
4. The `answer` tool must receive a table with `columns` and `rows`.
5. Always return exactly one JSON object with keys `thought`, `action`, and `action_input`.
6. Always wrap that JSON object in exactly one fenced code block that starts with ```json and ends with ```.
7. Do not output any text before or after the fenced JSON block.


Keep reasoning concise and grounded in the observed data.
""".strip()

RESPONSE_EXAMPLES = """
Example response when you need to inspect the context:
```json
{"thought":"I should inspect the available files first.","action":"list_context","action_input":{"max_depth":4}}
```

Example response when you have the final answer:
```json
{"thought":"I have the final result table.","action":"answer","action_input":{"columns":["average_long_shots"],"rows":[["63.5"]]}}
```
""".strip()


PROMPTS_DIR = Path(__file__).resolve().parents[1] / "Prompts"
FORMAT_GUARDRAILS_PATH = PROMPTS_DIR / "format_guardrails.md"


@lru_cache(maxsize=1)
def _load_format_guardrails() -> str:
    if FORMAT_GUARDRAILS_PATH.exists():
        return FORMAT_GUARDRAILS_PATH.read_text(encoding="utf-8").strip()
    # Fallback keeps runtime stable if the file is missing.
    return (
        "Global Format Guardrails:\n"
        "1. Always return exactly one JSON object in one ```json fenced block.\n"
        "2. JSON keys must be exactly: thought, action, action_input.\n"
        "3. action_input must always be a JSON object (never a string/list/null).\n"
        "4. For action=answer, include action_input.columns and action_input.rows.\n"
        "5. Do not output any text before or after the fenced JSON block."
    )


def build_system_prompt(tool_descriptions: str, system_prompt: str | None = None) -> str:
    base_prompt = system_prompt or REACT_SYSTEM_PROMPT
    format_guardrails = _load_format_guardrails()
    return (
        f"{base_prompt}\n\n"
        "Global Format Guardrails:\n"
        f"{format_guardrails}\n\n"
        "Available tools:\n"
        f"{tool_descriptions}\n\n"
        f"{RESPONSE_EXAMPLES}\n\n"
        "You must always return a single ```json fenced block containing one JSON object "
        "with keys `thought`, `action`, and `action_input`, and no extra text."
    )


def build_task_prompt(task: PublicTask) -> str:
    return (
        f"Question: {task.question}\n"
        "All tool file paths are relative to the task context directory. "
        "When you have the final table, call the `answer` tool."
    )


def build_observation_prompt(observation: dict[str, object]) -> str:
    rendered = json.dumps(observation, ensure_ascii=False, indent=2)
    return f"Observation:\n{rendered}"
