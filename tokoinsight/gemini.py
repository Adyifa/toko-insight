"""Minimal Google Gemini REST client with function calling (standard library only).

The free tier is rate limited, so 429 and 5xx responses are retried with the delay the API asks
for (RetryInfo.retryDelay) or exponential backoff. The transport is injectable for tests.
"""

from __future__ import annotations

import json
import re
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass, field

BASE_URL = "https://generativelanguage.googleapis.com/v1beta"
RETRY_STATUSES = {429, 500, 502, 503, 504}

# transport(url, headers, body) -> (status, response body)
Transport = Callable[[str, dict, bytes], tuple[int, bytes]]


class GeminiError(RuntimeError):
    def __init__(self, message: str, status: int | None = None) -> None:
        super().__init__(message)
        self.status = status


@dataclass
class Usage:
    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0

    def add(self, metadata: dict) -> None:
        self.calls += 1
        self.input_tokens += metadata.get("promptTokenCount", 0) or 0
        self.output_tokens += metadata.get("candidatesTokenCount", 0) or 0

    def to_dict(self) -> dict:
        return {"calls": self.calls, "input_tokens": self.input_tokens, "output_tokens": self.output_tokens}


def urllib_transport(timeout: float) -> Transport:
    def send(url: str, headers: dict, body: bytes) -> tuple[int, bytes]:
        request = urllib.request.Request(url, data=body, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return response.status, response.read()
        except urllib.error.HTTPError as error:
            return error.code, error.read()

    return send


def retry_delay(body: dict, attempt: int) -> float:
    """Seconds to wait: the API's RetryInfo.retryDelay ("31s") when present, else 2, 4, 8, ... s."""
    for detail in body.get("error", {}).get("details", []) or []:
        delay = detail.get("retryDelay")
        if isinstance(delay, str) and (match := re.fullmatch(r"(\d+(?:\.\d+)?)s", delay)):
            return min(float(match.group(1)) + 0.5, 90.0)
    return min(2.0 ** (attempt + 1), 30.0)


@dataclass
class GeminiClient:
    api_key: str
    model: str = "gemini-2.5-flash"
    thinking_budget: int | None = 0  # 0 turns off "thinking" on 2.5 Flash: faster and cheaper
    temperature: float = 0.1
    max_tokens: int = 4096
    max_retries: int = 5
    timeout: float = 90.0
    transport: Transport | None = None
    sleep: Callable[[float], None] = time.sleep
    usage: Usage = field(default_factory=Usage)

    def __post_init__(self) -> None:
        if not self.api_key:
            raise GeminiError("GEMINI_API_KEY is not set. Get a free key at https://aistudio.google.com/apikey")
        self.transport = self.transport or urllib_transport(self.timeout)

    @property
    def name(self) -> str:
        return self.model

    def _post(self, path: str, payload: dict) -> dict:
        url = f"{BASE_URL}/{path}"
        headers = {"Content-Type": "application/json", "x-goog-api-key": self.api_key}
        body = json.dumps(payload).encode()
        for attempt in range(self.max_retries + 1):
            status, raw = self.transport(url, headers, body)
            try:
                data = json.loads(raw or b"{}")
            except json.JSONDecodeError:
                data = {}
            if status == 200:
                return data
            message = data.get("error", {}).get("message") or raw[:200].decode(errors="replace")
            if status not in RETRY_STATUSES or attempt == self.max_retries:
                raise GeminiError(f"Gemini API error {status}: {message}", status)
            self.sleep(retry_delay(data, attempt))
        raise AssertionError("unreachable")

    def generate(self, contents: list[dict], *, system: str | None = None, tools: list[dict] | None = None,
                 tool_mode: str = "AUTO") -> dict:
        """One generateContent call. Returns the model's content: {"role": "model", "parts": [...]}.

        Parts are either {"text": ...} or {"functionCall": {"name": ..., "args": {...}}}. The content
        must be appended to the conversation unchanged (it can carry thought signatures).
        """
        config: dict = {"temperature": self.temperature, "maxOutputTokens": self.max_tokens}
        if self.thinking_budget is not None:
            config["thinkingConfig"] = {"thinkingBudget": self.thinking_budget}
        payload: dict = {"contents": contents, "generationConfig": config}
        if system:
            payload["systemInstruction"] = {"parts": [{"text": system}]}
        if tools:
            payload["tools"] = [{"functionDeclarations": tools}]
            payload["toolConfig"] = {"functionCallingConfig": {"mode": tool_mode}}
        data = self._post(f"models/{self.model}:generateContent", payload)
        self.usage.add(data.get("usageMetadata", {}))
        candidates = data.get("candidates") or []
        if not candidates:
            reason = data.get("promptFeedback", {}).get("blockReason", "no candidates")
            raise GeminiError(f"Gemini returned no answer ({reason})")
        content = candidates[0].get("content") or {}
        if not content.get("parts"):
            raise GeminiError(f"Gemini returned an empty answer (finishReason {candidates[0].get('finishReason')})")
        content.setdefault("role", "model")
        return content
