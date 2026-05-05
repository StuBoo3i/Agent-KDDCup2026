from __future__ import annotations

from dataclasses import dataclass
import os
import time
from typing import Any, Protocol

from openai import APIConnectionError, APIError, OpenAI


@dataclass(frozen=True, slots=True)
class ModelMessage:
    role: str
    content: str


@dataclass(frozen=True, slots=True)
class ModelStep:
    thought: str
    action: str
    action_input: dict[str, Any]
    raw_response: str


class ModelAdapter(Protocol):
    def complete(self, messages: list[ModelMessage]) -> str:
        raise NotImplementedError


class OpenAIModelAdapter:
    def __init__(
        self,
        *,
        model: str,
        api_base: str,
        api_key: str,
        temperature: float,
    ) -> None:
        self.model = model
        self.api_base = api_base.rstrip("/")
        self.api_key = api_key
        self.temperature = temperature
        self._client = OpenAI(
            api_key=self.api_key,
            base_url=self.api_base,
        )

    def _format_connection_error(self, exc: APIConnectionError) -> str:
        cause = exc.__cause__
        details = str(exc) or "Connection error."
        if cause is not None:
            details = f"{details} (cause: {cause})"
        proxy_keys = ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy")
        active_proxy_env = [key for key in proxy_keys if os.getenv(key)]
        if active_proxy_env:
            details = f"{details} [active proxy env: {', '.join(active_proxy_env)}]"
        return details

    def complete(self, messages: list[ModelMessage]) -> str:
        if not self.api_key:
            raise RuntimeError(
                "Missing model API key. Set config.agent.api_key or environment variable OPENAI_API_KEY/DASHSCOPE_API_KEY."
            )

        max_attempts = 3
        last_exc: Exception | None = None
        for attempt in range(1, max_attempts + 1):
            try:
                response = self._client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": message.role, "content": message.content} for message in messages],
                    temperature=self.temperature,
                )
                break
            except APIConnectionError as exc:
                last_exc = exc
                if attempt >= max_attempts:
                    raise RuntimeError(
                        f"Model request failed after {max_attempts} attempts: {self._format_connection_error(exc)}"
                    ) from exc
                time.sleep(1.5 * attempt)
            except APIError as exc:
                raise RuntimeError(f"Model request failed: {exc}") from exc
        else:
            if last_exc is not None:
                raise RuntimeError(f"Model request failed: {last_exc}") from last_exc
            raise RuntimeError("Model request failed for unknown reasons.")

        choices = response.choices or []
        if not choices:
            raise RuntimeError("Model response missing choices.")
        content = choices[0].message.content
        if not isinstance(content, str):
            raise RuntimeError("Model response missing text content.")
        return content


class ScriptedModelAdapter:
    def __init__(self, responses: list[str]) -> None:
        self._responses = list(responses)

    def complete(self, messages: list[ModelMessage]) -> str:
        del messages
        if not self._responses:
            raise RuntimeError("No scripted model responses remaining.")
        return self._responses.pop(0)
