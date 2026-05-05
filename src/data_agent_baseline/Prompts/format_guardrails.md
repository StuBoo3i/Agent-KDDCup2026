# Baseline Format Guardrails (Mandatory)

These rules are mandatory for every step and every task.

1. Output format
- You must output exactly one fenced JSON block starting with ```json and ending with ```.
- Do not output any extra text before or after the fenced block.
- The JSON object must contain exactly these top-level keys:
  - `thought` (string)
  - `action` (string)
  - `action_input` (object)

2. Action protocol
- `action_input` must always be a JSON object.
- Never put raw code or raw text directly into `action_input` as a string.
- For `execute_python`, use:
  - `action_input = {"code": "...python code..."}`
- For `execute_context_sql`, use:
  - `action_input = {"path": "...", "sql": "...", "limit": 200}`
- For file-reading tools, always pass required fields exactly as documented.

3. Answer protocol
- Task only ends when `action = "answer"`.
- For `answer`, `action_input` must include:
  - `columns`: list[string]
  - `rows`: list[list[Any]]
- Every row length must equal `len(columns)`.
- Do not include extra columns that were not asked.

4. Column-name strictness for scoring
- Output column names are scored strictly.
- Keep exact spelling, case, punctuation, and token form.
- If question/output expectation implies SQL-expression headers (for example `AVG(T1.x)`), preserve that exact header.
- If expected output uses original field names, do not rename with synonyms.

5. Determinism and validation before submit
- Before calling `answer`, verify:
  - column names are exact,
  - row count matches question constraints,
  - values are complete and not truncated.
- Prefer deterministic tie-breaking and stable ordering when relevant.
