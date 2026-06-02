"""Test doubles: a fake OpenAI-style client.

Mimics openai>=1.x response shape (object attributes) so the adapter is exercised
exactly as it would be against the real SDK, with no network and no dependency.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class FakeUsage:
    prompt_tokens: int = 10
    completion_tokens: int = 5
    total_tokens: int = 15


@dataclass
class FakeMessage:
    role: str = "assistant"
    content: str = "hello world"


@dataclass
class FakeChoice:
    index: int = 0
    finish_reason: str = "stop"
    message: FakeMessage = field(default_factory=FakeMessage)


@dataclass
class FakeResponse:
    id: str = "resp_1"
    model: str = "gpt-4o-mini"
    choices: list = field(default_factory=lambda: [FakeChoice()])
    usage: Any = field(default_factory=FakeUsage)


class _Completions:
    def __init__(self, parent: "FakeOpenAIClient") -> None:
        self._parent = parent

    def create(self, **kwargs) -> FakeResponse:
        self._parent.last_call = kwargs
        self._parent.call_count += 1
        if self._parent.raise_on_call:
            raise RuntimeError("provider exploded")
        resp = self._parent.next_response or FakeResponse()
        # Echo the requested model so model_used tracking is meaningful.
        resp.model = kwargs.get("model", resp.model)
        return resp


class _Chat:
    def __init__(self, parent: "FakeOpenAIClient") -> None:
        self.completions = _Completions(parent)


class FakeOpenAIClient:
    def __init__(self) -> None:
        self.chat = _Chat(self)
        self.last_call: dict | None = None
        self.call_count = 0
        self.raise_on_call = False
        self.next_response: FakeResponse | None = None
