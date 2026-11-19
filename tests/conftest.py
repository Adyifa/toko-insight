"""Shared fixtures: one synthetic dataset per test session, and a scripted fake LLM.

No API key or network is needed: the LLM is replaced by FakeLLM, which replays scripted
Gemini-shaped responses (text parts or functionCall parts).
"""

from __future__ import annotations

from datetime import date

import pytest

from tokoinsight.db import open_db
from tokoinsight.generator import generate
from tokoinsight.tools import Toolbox


@pytest.fixture(scope="session")
def db_path(tmp_path_factory):
    path = tmp_path_factory.mktemp("data") / "toko.db"
    db = open_db(path)
    generate(db, start=date(2023, 1, 1), end=date(2026, 10, 31), seed=11, volume=0.35)
    db.close()
    return path


@pytest.fixture(scope="session")
def db(db_path):
    database = open_db(db_path)
    yield database
    database.close()


@pytest.fixture(scope="session")
def toolbox(db):
    return Toolbox(db)


def call(name: str, **args) -> dict:
    return {"role": "model", "parts": [{"functionCall": {"name": name, "args": args}}]}


def text(value: str) -> dict:
    return {"role": "model", "parts": [{"text": value}]}


class FakeLLM:
    """Replays a script of model contents; records what it was sent."""

    name = "fake-llm"

    def __init__(self, script):
        self.script = list(script)
        self.requests: list[dict] = []

    def generate(self, contents, *, system=None, tools=None, tool_mode="AUTO"):
        self.requests.append({"contents": [dict(c) for c in contents], "system": system, "tools": tools,
                              "tool_mode": tool_mode})
        if not self.script:
            return text("(script exhausted)")
        step = self.script.pop(0)
        return step(contents) if callable(step) else step
