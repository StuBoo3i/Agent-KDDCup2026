from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class SkillDefinition:
    name: str
    description: str
    trigger_extensions: tuple[str, ...]
    trigger_keywords: tuple[str, ...]
    playbook: str


SKILL_LIBRARY: tuple[SkillDefinition, ...] = (
    SkillDefinition(
        name="tabular_aggregation",
        description="Reliable tabular aggregation over CSV/Excel style files.",
        trigger_extensions=(".csv", ".tsv", ".xlsx", ".xls"),
        trigger_keywords=("average", "sum", "count", "top", "group", "排名", "平均", "统计"),
        playbook=(
            "Use list_context -> read_csv for schema preview; switch to execute_python for full-table "
            "aggregation when row count is large; submit only final columns requested by the question."
        ),
    ),
    SkillDefinition(
        name="json_nested_extraction",
        description="Nested-field extraction and normalization from JSON documents.",
        trigger_extensions=(".json",),
        trigger_keywords=("json", "field", "nested", "属性", "字段", "键"),
        playbook=(
            "Inspect representative JSON samples with read_json, then flatten nested keys in execute_python. "
            "Normalize missing values explicitly before computing aggregates."
        ),
    ),
    SkillDefinition(
        name="sqlite_analysis",
        description="Schema-first SQL analysis over SQLite/DB files.",
        trigger_extensions=(".sqlite", ".db"),
        trigger_keywords=("sql", "database", "join", "table", "数据库", "表", "关联"),
        playbook=(
            "Run inspect_sqlite_schema first, then execute_context_sql with explicit SELECT projections and "
            "limits before final aggregation."
        ),
    ),
    SkillDefinition(
        name="document_evidence",
        description="Fact extraction from text/markdown documents.",
        trigger_extensions=(".txt", ".md"),
        trigger_keywords=("document", "text", "policy", "report", "文档", "文本", "说明"),
        playbook=(
            "Use read_doc for evidence snippets; extract exact values and keep units/date context intact when "
            "constructing answers."
        ),
    ),
    SkillDefinition(
        name="cross_source_validation",
        description="Cross-check values across multiple file types and detect inconsistencies.",
        trigger_extensions=(),
        trigger_keywords=("compare", "difference", "match", "verify", "对比", "一致", "校验"),
        playbook=(
            "Collect candidate values from each source separately, then compare with execute_python before "
            "submitting. Prefer deterministic tie-breaking rules."
        ),
    ),
)


def build_context_profile(entries: list[dict[str, Any]]) -> dict[str, Any]:
    extension_counts: dict[str, int] = {}
    file_count = 0
    directory_count = 0
    for entry in entries:
        kind = str(entry.get("kind", ""))
        if kind == "dir":
            directory_count += 1
            continue
        file_count += 1
        ext = Path(str(entry.get("path", ""))).suffix.lower()
        extension_counts[ext or "<none>"] = extension_counts.get(ext or "<none>", 0) + 1
    return {
        "file_count": file_count,
        "directory_count": directory_count,
        "extension_counts": dict(sorted(extension_counts.items(), key=lambda item: item[0])),
    }


def recommend_skills(task_question: str, context_profile: dict[str, Any]) -> list[SkillDefinition]:
    lowered_question = task_question.lower()
    extension_counts = context_profile.get("extension_counts", {})
    selected: list[SkillDefinition] = []

    for skill in SKILL_LIBRARY:
        has_extension_trigger = any(extension_counts.get(ext, 0) > 0 for ext in skill.trigger_extensions)
        has_keyword_trigger = any(keyword.lower() in lowered_question for keyword in skill.trigger_keywords)
        if has_extension_trigger or has_keyword_trigger:
            selected.append(skill)

    # Always provide one generic guardrail if no specific skill matched.
    if not selected:
        selected.append(SKILL_LIBRARY[0])

    # Cross-source validation is valuable when multiple file types are present.
    if len(extension_counts) >= 2 and SKILL_LIBRARY[-1] not in selected:
        selected.append(SKILL_LIBRARY[-1])
    return selected


def render_skills_for_prompt(skills: list[SkillDefinition]) -> str:
    lines: list[str] = []
    for skill in skills:
        lines.append(f"- {skill.name}: {skill.description}")
        lines.append(f"  playbook: {skill.playbook}")
    return "\n".join(lines)

