from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime
from time import perf_counter
import uuid
from typing import Any, Literal, TypedDict, cast

from langgraph.graph import END, START, StateGraph

from data_agent_baseline.agents.model import ModelAdapter, ModelMessage
from data_agent_baseline.agents.prompt import (
    REACT_SYSTEM_PROMPT,
    build_observation_prompt,
    build_system_prompt,
    build_task_prompt,
)
from data_agent_baseline.agents.react import parse_model_step
from data_agent_baseline.agents.runtime import AgentRunResult, StepRecord, utc_now_iso
from data_agent_baseline.agents.skills import (
    SkillDefinition,
    build_context_profile,
    recommend_skills,
    render_skills_for_prompt,
)
from data_agent_baseline.benchmark.schema import AnswerTable, PublicTask
from data_agent_baseline.tools.registry import ToolRegistry


@dataclass(frozen=True, slots=True)
class LangGraphAgentConfig:
    max_steps: int = 16
    enable_langsmith: bool = False
    langsmith_project: str = "kddcup2026-data-agent"


class AgentGraphState(TypedDict):
    task: PublicTask
    steps: list[StepRecord]
    step_index: int
    max_steps: int
    answer: AnswerTable | None
    failure_reason: str | None
    context_profile: dict[str, Any]
    skill_definitions: list[SkillDefinition]
    bootstrap_observations: list[dict[str, Any]]
    pending_raw_response: str
    pending_step_started_at_utc: str
    pending_step_started_perf: float
    pending_prompt_messages: list[dict[str, str]]


class LangGraphReActAgent:
    def __init__(
        self,
        *,
        model: ModelAdapter,
        tools: ToolRegistry,
        config: LangGraphAgentConfig | None = None,
        system_prompt: str | None = None,
    ) -> None:
        self.model = model
        self.tools = tools
        self.config = config or LangGraphAgentConfig()
        self.system_prompt = system_prompt or REACT_SYSTEM_PROMPT
        self._graph = self._build_graph()

    def _build_graph(self):
        graph = StateGraph(AgentGraphState)
        graph.add_node("profile_context", self._node_profile_context)
        graph.add_node("plan_action", self._node_plan_action)
        graph.add_node("execute_action", self._node_execute_action)

        graph.add_edge(START, "profile_context")
        graph.add_edge("profile_context", "plan_action")
        graph.add_edge("plan_action", "execute_action")
        graph.add_conditional_edges(
            "execute_action",
            self._route_after_execute,
            {
                "continue": "plan_action",
                "end": END,
            },
        )
        return graph.compile()

    def _build_messages(self, state: AgentGraphState) -> list[ModelMessage]:
        task = state["task"]
        rendered_skills = render_skills_for_prompt(state["skill_definitions"])
        context_profile = json.dumps(state["context_profile"], ensure_ascii=False, indent=2)
        system_content = build_system_prompt(
            self.tools.describe_for_prompt(),
            system_prompt=self.system_prompt,
        )
        system_content = (
            f"{system_content}\n\n"
            "Specialized analysis skills you can apply based on this task:\n"
            f"{rendered_skills}\n\n"
            "Context profile inferred from context files:\n"
            f"{context_profile}\n\n"
            "Prefer deterministic workflows and explicit verification before calling answer."
        )

        messages = [ModelMessage(role="system", content=system_content)]
        messages.append(ModelMessage(role="user", content=build_task_prompt(task)))
        for observation in state["bootstrap_observations"]:
            messages.append(ModelMessage(role="user", content=build_observation_prompt(observation)))
        for step in state["steps"]:
            messages.append(ModelMessage(role="assistant", content=step.raw_response))
            messages.append(ModelMessage(role="user", content=build_observation_prompt(step.observation)))
        return messages

    def _node_profile_context(self, state: AgentGraphState) -> dict[str, Any]:
        task = state["task"]
        bootstrap_observations: list[dict[str, Any]] = []
        context_profile: dict[str, Any] = {
            "file_count": 0,
            "directory_count": 0,
            "extension_counts": {},
        }

        try:
            list_result = self.tools.execute(task, "list_context", {"max_depth": 4})
            context_observation = {
                "ok": list_result.ok,
                "tool": "list_context",
                "content": list_result.content,
            }
            bootstrap_observations.append(context_observation)
            if list_result.ok:
                entries = cast(list[dict[str, Any]], list_result.content.get("entries", []))
                context_profile = build_context_profile(entries)
        except Exception as exc:  # noqa: BLE001
            bootstrap_observations.append(
                {
                    "ok": False,
                    "tool": "list_context",
                    "error": str(exc),
                }
            )

        skills = recommend_skills(task.question, context_profile)
        bootstrap_observations.append(
            {
                "ok": True,
                "tool": "skill_recommender",
                "content": {
                    "recommended_skills": [skill.name for skill in skills],
                    "context_profile": context_profile,
                },
            }
        )
        return {
            "context_profile": context_profile,
            "skill_definitions": skills,
            "bootstrap_observations": bootstrap_observations,
        }

    def _node_plan_action(self, state: AgentGraphState) -> dict[str, Any]:
        if state["step_index"] >= state["max_steps"]:
            return {
                "failure_reason": "Agent did not submit an answer within max_steps.",
                "pending_raw_response": "",
                "pending_step_started_at_utc": "",
                "pending_step_started_perf": 0.0,
                "pending_prompt_messages": [],
            }
        step_started_perf = perf_counter()
        step_started_at = utc_now_iso()
        messages = self._build_messages(state)
        prompt_messages = [{"role": message.role, "content": message.content} for message in messages]
        raw_response = self.model.complete(messages)
        return {
            "pending_raw_response": raw_response,
            "pending_step_started_at_utc": step_started_at,
            "pending_step_started_perf": step_started_perf,
            "pending_prompt_messages": prompt_messages,
        }

    def _node_execute_action(self, state: AgentGraphState) -> dict[str, Any]:
        if state.get("failure_reason"):
            return {}

        raw_response = state.get("pending_raw_response", "")
        prompt_messages_raw = state.get("pending_prompt_messages", [])
        prompt_messages: list[dict[str, str]] = []
        if isinstance(prompt_messages_raw, list):
            for message in prompt_messages_raw:
                if not isinstance(message, dict):
                    continue
                role = str(message.get("role", "")).strip()
                content = str(message.get("content", ""))
                if role:
                    prompt_messages.append({"role": role, "content": content})
        step_started_at = str(state.get("pending_step_started_at_utc", "")).strip() or utc_now_iso()
        step_started_perf_raw = state.get("pending_step_started_perf", 0.0)
        if isinstance(step_started_perf_raw, (int, float)) and step_started_perf_raw > 0:
            step_started_perf = float(step_started_perf_raw)
        else:
            step_started_perf = perf_counter()
        step_index = state["step_index"] + 1
        steps = list(state["steps"])
        answer = state["answer"]
        failure_reason = state["failure_reason"]

        try:
            model_step = parse_model_step(raw_response)
            tool_result = self.tools.execute(state["task"], model_step.action, model_step.action_input)
            observation = {
                "ok": tool_result.ok,
                "tool": model_step.action,
                "content": tool_result.content,
            }
            steps.append(
                StepRecord(
                    step_index=step_index,
                    thought=model_step.thought,
                    action=model_step.action,
                    action_input=model_step.action_input,
                    raw_response=raw_response,
                    observation=observation,
                    ok=tool_result.ok,
                    started_at_utc=step_started_at,
                    completed_at_utc=utc_now_iso(),
                    elapsed_seconds=round(perf_counter() - step_started_perf, 3),
                    prompt_messages=prompt_messages,
                )
            )
            if tool_result.is_terminal:
                answer = tool_result.answer
        except Exception as exc:  # noqa: BLE001
            observation = {
                "ok": False,
                "error": str(exc),
            }
            steps.append(
                StepRecord(
                    step_index=step_index,
                    thought="",
                    action="__error__",
                    action_input={},
                    raw_response=raw_response,
                    observation=observation,
                    ok=False,
                    started_at_utc=step_started_at,
                    completed_at_utc=utc_now_iso(),
                    elapsed_seconds=round(perf_counter() - step_started_perf, 3),
                    prompt_messages=prompt_messages,
                )
            )
            if step_index >= state["max_steps"]:
                failure_reason = "Agent did not submit an answer within max_steps."

        return {
            "steps": steps,
            "step_index": step_index,
            "answer": answer,
            "failure_reason": failure_reason,
            "pending_raw_response": "",
            "pending_step_started_at_utc": "",
            "pending_step_started_perf": 0.0,
            "pending_prompt_messages": [],
        }

    def _route_after_execute(self, state: AgentGraphState) -> Literal["continue", "end"]:
        if state.get("answer") is not None:
            return "end"
        if state.get("failure_reason"):
            return "end"
        if state["step_index"] >= state["max_steps"]:
            return "end"
        return "continue"

    def _langgraph_mermaid(self) -> str | None:
        try:
            return self._graph.get_graph().draw_mermaid()
        except Exception:  # noqa: BLE001
            return None

    def _task_trace_name(self, task_id: str) -> str:
        timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M")
        if task_id.startswith("task_"):
            suffix = task_id.split("_", 1)[1]
            if suffix.isdigit():
                return f"{timestamp}_Task{int(suffix)}"
        sanitized = "".join(char if char.isalnum() else "_" for char in task_id)
        return f"{timestamp}_Task{sanitized}"

    def run(self, task: PublicTask) -> AgentRunResult:
        effective_langsmith_project = self.config.langsmith_project
        if self.config.enable_langsmith:
            os.environ.setdefault("LANGSMITH_TRACING", "true")
            env_project = os.getenv("LANGSMITH_PROJECT", "").strip()
            if env_project:
                effective_langsmith_project = env_project
            else:
                os.environ["LANGSMITH_PROJECT"] = effective_langsmith_project

        initial_state: AgentGraphState = {
            "task": task,
            "steps": [],
            "step_index": 0,
            "max_steps": self.config.max_steps,
            "answer": None,
            "failure_reason": None,
            "context_profile": {
                "file_count": 0,
                "directory_count": 0,
                "extension_counts": {},
            },
            "skill_definitions": [],
            "bootstrap_observations": [],
            "pending_raw_response": "",
            "pending_step_started_at_utc": "",
            "pending_step_started_perf": 0.0,
            "pending_prompt_messages": [],
        }
        task_trace_name = self._task_trace_name(task.task_id)
        runtime_run_id = uuid.uuid4()
        invoke_config = {
            "run_name": task_trace_name,
            "run_id": runtime_run_id,
            "tags": [f"task_id:{task.task_id}", "source:langgraph_runtime"],
            "metadata": {
                "task_id": task.task_id,
                "task_trace_name": task_trace_name,
            },
        }
        try:
            final_state = cast(AgentGraphState, self._graph.invoke(initial_state, invoke_config))
        except TypeError:
            final_state = cast(AgentGraphState, self._graph.invoke(initial_state))
        failure_reason = final_state.get("failure_reason")
        if final_state.get("answer") is None and failure_reason is None:
            failure_reason = "Agent did not submit an answer within max_steps."

        metadata: dict[str, Any] = {
            "agent_framework": "langgraph",
            "context_profile": final_state.get("context_profile", {}),
            "recommended_skills": [
                skill.name for skill in final_state.get("skill_definitions", [])
            ],
        }
        mermaid = self._langgraph_mermaid()
        if mermaid:
            metadata["langgraph_mermaid"] = mermaid
        if self.config.enable_langsmith:
            metadata["langsmith"] = {
                "enabled": True,
                "project": effective_langsmith_project,
                "task_trace_name": task_trace_name,
                "task_runtime_run_id": str(runtime_run_id),
            }

        return AgentRunResult(
            task_id=task.task_id,
            answer=final_state.get("answer"),
            steps=list(final_state.get("steps", [])),
            failure_reason=failure_reason,
            metadata=metadata,
        )
