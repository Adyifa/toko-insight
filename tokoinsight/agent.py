"""The analytics agent: an LLM that answers by calling tools, never by guessing numbers.

Loop: send the question + tool declarations -> the model either answers or asks for function
calls -> we run them (tools.Toolbox) and send the results back -> repeat until it answers or the
step limit is hit (then it must answer with what it has). Every tool call, its arguments and the
SQL it ran are returned with the answer, so the user can check the work.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Protocol

from .db import SCHEMA_DOC
from .tools import ToolCall, Toolbox

MAX_RESULT_CHARS = 14_000


class LLM(Protocol):
    name: str

    def generate(self, contents: list[dict], *, system: str | None = None, tools: list[dict] | None = None,
                 tool_mode: str = "AUTO") -> dict: ...


SYSTEM_PROMPT = """\
You are Toko Insight, a retail analytics assistant for the owner of a small minimarket chain in
Indonesia. You answer questions about their sales data by calling tools. The data runs from
{first} to {as_of}; "now", "this year" and "last 12 months" are relative to {as_of}.

{schema}

Rules:
1. Every number in your answer must come from a tool result in this conversation. Never estimate,
   round up or invent figures. If the tools cannot answer, say so.
2. Prefer the analytics tools (top_products, growth, abc_analysis, purchase_frequency,
   basket_pairs, seasonality, forecast, restock_plan). Use run_sql only for questions they do not
   cover (e.g. payment methods, hours, a specific store/date filter, promos). Use exact product
   names from list_products when a name is ambiguous.
3. Forecasts and stock advice: call forecast / restock_plan. Mention the model used and its
   backtest error (WAPE) next to the seasonal-naive baseline, and the 80% range.
4. Explain the reasoning in business terms (season, trend, promo, stock cover), briefly.
5. Reply in the language of the question (Bahasa Indonesia or English). Format money as
   Rupiah (e.g. Rp 12,5 juta / Rp 12.5 million). Say which period the numbers cover.
6. Structure: a direct answer first, then the key numbers (a short table or bullets), then
   reasoning and caveats. Keep it concise.
"""


def compact(value, max_chars: int = MAX_RESULT_CHARS):
    """Shrink a tool result until its JSON fits in max_chars (long lists are cut, with a note)."""
    text = json.dumps(value, ensure_ascii=False, default=str)
    if len(text) <= max_chars:
        return value
    for keep in (20, 12, 8, 5, 3):
        shrunk = _cut_lists(value, keep)
        if len(json.dumps(shrunk, ensure_ascii=False, default=str)) <= max_chars:
            return shrunk
    return {"truncated_json": text[:max_chars]}


def _cut_lists(value, keep: int):
    if isinstance(value, dict):
        return {k: _cut_lists(v, keep) for k, v in value.items()}
    if isinstance(value, list):
        items = [_cut_lists(v, keep) for v in value[:keep]]
        if len(value) > keep:
            items.append(f"... {len(value) - keep} more rows not shown")
        return items
    return value


@dataclass
class AgentAnswer:
    question: str
    text: str
    steps: list[ToolCall] = field(default_factory=list)
    model: str = ""
    seconds: float = 0.0
    usage: dict = field(default_factory=dict)
    stopped_early: bool = False

    def tools_used(self) -> list[str]:
        return [s.name for s in self.steps]

    def to_dict(self, include_results: bool = True) -> dict:
        steps = [s.to_dict() for s in self.steps]
        if not include_results:
            for s in steps:
                s.pop("result", None)
        return {"question": self.question, "answer": self.text, "steps": steps, "model": self.model,
                "seconds": round(self.seconds, 2), "usage": self.usage, "stopped_early": self.stopped_early}


class Agent:
    def __init__(self, llm: LLM, toolbox: Toolbox, max_steps: int = 8) -> None:
        self.llm = llm
        self.toolbox = toolbox
        self.max_steps = max_steps
        db = toolbox.db
        self.system = SYSTEM_PROMPT.format(first=db.first_date(), as_of=db.as_of(), schema=SCHEMA_DOC)

    def ask(self, question: str, history: list[tuple[str, str]] | None = None) -> AgentAnswer:
        started = time.perf_counter()
        contents: list[dict] = []
        for user_text, model_text in (history or [])[-6:]:
            contents.append({"role": "user", "parts": [{"text": user_text}]})
            contents.append({"role": "model", "parts": [{"text": model_text}]})
        contents.append({"role": "user", "parts": [{"text": question}]})
        declarations = self.toolbox.declarations()
        steps: list[ToolCall] = []
        text, stopped = "", False
        for step in range(self.max_steps + 1):
            last = step == self.max_steps
            content = self.llm.generate(contents, system=self.system, tools=declarations,
                                        tool_mode="NONE" if last else "AUTO")
            contents.append(content)
            calls = [p["functionCall"] for p in content.get("parts", []) if "functionCall" in p]
            if not calls or last:
                text = "".join(p.get("text", "") for p in content.get("parts", []) if not p.get("thought")).strip()
                stopped = last and bool(calls)
                break
            responses = []
            for fc in calls:
                call = self.toolbox.call(fc.get("name", ""), fc.get("args") or {})
                steps.append(call)
                payload = {"error": call.error} if call.error else {"result": compact(call.result)}
                response = {"name": call.name, "response": payload}
                if fc.get("id"):
                    response["id"] = fc["id"]
                responses.append({"functionResponse": response})
            contents.append({"role": "user", "parts": responses})
        if not text:
            # Out of steps (or an empty reply): ask once more, with no tools, to answer from what it has.
            contents.append({"role": "user", "parts": [{"text": "Answer the question now, using only the tool "
                                                                "results above. Do not call more tools."}]})
            content = self.llm.generate(contents, system=self.system)
            text = "".join(p.get("text", "") for p in content.get("parts", []) if not p.get("thought")).strip()
        usage = getattr(getattr(self.llm, "usage", None), "to_dict", lambda: {})()
        return AgentAnswer(question, text or "(no answer)", steps, getattr(self.llm, "name", "llm"),
                           time.perf_counter() - started, usage, stopped)
