"""Jev-based tool-call compaction for Hermes.

Port of https://github.com/tamaratran/fast-jev-compaction (MIT license) adapted for
integration into Hermes's main compression pipeline. Uses Jev (System One) to score
every tool call and result, keeping/dropping based on keepThreshold. Never rewrites
text, only deletes tool calls and truncates results.

Transport: OpenRouter's Decisions API with ~typesafe/jev-latest model.
Requires OPENROUTER_API_KEY environment variable.
"""
from __future__ import annotations

import json
import logging
import math
import os
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional
from urllib.error import HTTPError
from urllib.request import Request, urlopen

logger = logging.getLogger(__name__)

STATE_CONTEXT = (
    "A coding assistant conversation is being compacted to free context. `history` is the "
    "whole conversation so far, oldest first; tool outputs are replaced by a short `result` "
    "note and long texts may be abridged. Each question asks whether one tool call, or the "
    "full output of that call, still needs to stay in the history verbatim. Whatever is not "
    "kept is deleted permanently, but the assistant can always re-run a tool or re-read a file."
)

INPUT_CHARS = (1000, 200, 60)
TEXT_HEAD = 400
TEXT_TAIL = 150
REQUEST_OVERHEAD_TOKENS = 20
OPENROUTER_DECISIONS_URL = "https://openrouter.ai/api/alpha/decisions"
OPENROUTER_JEV_MODEL = "~typesafe/jev-latest"

_TOKEN_PIECES = re.compile(r"[A-Za-z]+|\d+|[^\sA-Za-z\d]")
_WS = re.compile(r"\s+")


def estimate_tokens(text: str) -> int:
    """Estimate tokens without tokenizer: letters/6, digits/2, symbols 0.9."""
    tokens = 0.0
    for piece in _TOKEN_PIECES.findall(text):
        c = piece[0]
        if c.isdigit():
            tokens += len(piece) / 2
        elif c.isascii() and c.isalpha():
            tokens += 1 + (len(piece) - 1) // 6
        else:
            tokens += 0.9
    return math.ceil(tokens)


def truncate(text: str, limit: int) -> str:
    """Truncate text to limit, adding ellipsis if needed."""
    return text if len(text) <= limit else text[: max(0, limit - 1)] + "…"


def abridge(text: str, head: int, tail: int) -> str:
    """Keep head and tail, omit middle with note."""
    if len(text) <= head + tail + 40:
        return text
    return f"{text[:head]}\n[… {len(text) - head - tail} chars omitted …]\n{text[-tail:]}"


def _json(value: Any) -> str:
    """JSON encode for Jev requests."""
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), default=str)


def message_text(m: Dict[str, Any]) -> str:
    """Extract text content from a message."""
    c = m.get("content")
    if isinstance(c, str):
        return c
    if isinstance(c, list):
        return "\n".join(p.get("text", "") for p in c if isinstance(p, dict) and p.get("type") == "text")
    return ""


@dataclass
class ToolCall:
    """Represents a tool call with its result for compaction decision."""
    id: str
    tool_call_id: str
    tool: str
    input: Any
    call_index: int
    result_index: int
    result_chars: int
    is_error: bool
    pinned: bool


@dataclass
class Decision:
    """Compaction decision for one tool call."""
    id: str
    tool: str
    keep_call: float
    keep_result: float
    action: str  # keep | drop_result | drop_call
    reason: str  # pinned | kept | result_dropped | call_dropped


@dataclass
class JevOptions:
    """Configuration for Jev-based compaction."""
    goal: str = ""
    keep_threshold: float = 0.5
    preserve_recent_messages: int = 6
    max_state_tokens: int = 25_000
    max_request_tokens: int = 30_000
    truncate_head_chars: int = 300


@dataclass
class JevUsage:
    """Usage statistics for a Jev compaction run."""
    requests: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    models: List[str] = field(default_factory=list)


def is_pinned(index: int, total: int, preserve: int) -> bool:
    """Check if a message index should be preserved (first or last N)."""
    return index == 0 or index >= total - preserve


def _parse_arguments(fn: Dict[str, Any]) -> Any:
    """Parse tool function arguments from string or dict."""
    args = fn.get("arguments")
    if isinstance(args, str):
        try:
            return json.loads(args)
        except Exception:
            return {"arguments": args}
    return args if args is not None else {}


def collect_tool_calls(messages: List[Dict[str, Any]], preserve: int) -> List[ToolCall]:
    """Collect all tool_use + tool_result pairs from messages."""
    # First pass: collect all results indexed by tool_call_id
    results: Dict[str, int] = {}
    for idx, m in enumerate(messages):
        if m.get("role") == "tool" and m.get("tool_call_id"):
            results[m["tool_call_id"]] = idx

    # Second pass: create ToolCall objects for calls that have results
    calls: List[ToolCall] = []
    total = len(messages)
    for idx, m in enumerate(messages):
        for tc in m.get("tool_calls") or []:
            tcid = tc.get("id")
            if tcid not in results:
                continue
            ridx = results[tcid]
            fn = tc.get("function") or {}
            rtext = message_text(messages[ridx])
            calls.append(ToolCall(
                id=f"t{len(calls) + 1}",
                tool_call_id=tcid,
                tool=fn.get("name") or tc.get("name") or "tool",
                input=_parse_arguments(fn),
                call_index=idx,
                result_index=ridx,
                result_chars=len(rtext),
                is_error=bool(re.match(r"\s*(\{\"error\"|Error\b|error:)", rtext[:40], re.I)),
                pinned=is_pinned(idx, total, preserve) or is_pinned(ridx, total, preserve),
            ))
    return calls


def _input_text(inp: Any, limit: int) -> str:
    """Convert input to limited-length string."""
    return truncate(_json(inp), limit)


def _result_note(call: ToolCall) -> str:
    """Generate a short note for a tool result."""
    return f"{'error' if call.is_error else 'ok'}, {call.result_chars} chars (omitted)"


def _compact_call(call: ToolCall) -> str:
    """Compress a tool call to a single line."""
    if isinstance(call.input, dict):
        parts = []
        for k, v in call.input.items():
            text = v if isinstance(v, str) else _json({k: v})[:200]
            flat = _WS.sub(" ", text)
            parts.append(f"{k}={flat}")
        inp = " ".join(parts)
    else:
        inp = _json(call.input)
    return f"{call.id} {call.tool} {truncate(inp, INPUT_CHARS[2])} → {'error' if call.is_error else 'ok'} {call.result_chars}ch"


def _history_entries(messages, calls: List[ToolCall], input_chars: int) -> List[Dict[str, Any]]:
    """Build history entries for Jev state."""
    by_msg: Dict[int, List[ToolCall]] = {}
    for c in calls:
        by_msg.setdefault(c.call_index, []).append(c)
    entries = []
    for i, m in enumerate(messages):
        tcs = [{
            "id": c.id,
            "tool": c.tool,
            "input": _input_text(c.input, input_chars),
            "result": _result_note(c)
        } for c in by_msg.get(i, [])]
        text = message_text(m)
        if m.get("role") == "tool":
            text = ""  # results represented by call notes, not verbatim
        if not text.strip() and not tcs:
            continue
        entry: Dict[str, Any] = {"i": i, "role": m.get("role"), "text": text}
        if tcs:
            entry["tool_calls"] = tcs
        entries.append(entry)
    return entries


def goal_from_messages(messages) -> str:
    """Extract goal from recent user messages."""
    prompts = [message_text(m) for m in messages if m.get("role") == "user" and message_text(m).strip()]
    return "\n".join(truncate(p, 500) for p in prompts[-3:])


def fit_state(messages, calls: List[ToolCall], opt: JevOptions) -> Dict[str, Any]:
    """Shrink the whole-history state in stages until it fits within max_state_tokens."""
    goal = opt.goal or goal_from_messages(messages)
    total = len(messages)

    def state_of(history):
        return {"context": STATE_CONTEXT, "goal": goal, "history": history}

    def entry_tokens(e):
        return estimate_tokens(_json(e)) + 1

    base = estimate_tokens(_json(state_of([])))
    history: List[Dict[str, Any]] = []
    per: List[int] = []
    tokens = 0

    def rebuild(limit):
        nonlocal history, per, tokens
        history = _history_entries(messages, calls, limit)
        per = [entry_tokens(e) for e in history]
        tokens = base + sum(per)

    def fits():
        return tokens <= opt.max_state_tokens

    def fitted(h, stage):
        return {"state": state_of(h), "tokens": tokens, "stage": stage}

    def shrink(index, change):
        nonlocal tokens
        change(history[index])
        now = entry_tokens(history[index])
        tokens += now - per[index]
        per[index] = now

    # Stage 1: try full inputs
    rebuild(INPUT_CHARS[0])
    if fits():
        return fitted(history, "full")

    # Stage 2-3: progressively truncate inputs
    for limit in INPUT_CHARS[1:]:
        rebuild(limit)
        if fits():
            return fitted(history, f"inputs<={limit}")

    # Stage 4: abridge long texts (head + tail)
    def pinned(e):
        return is_pinned(e["i"], total, opt.preserve_recent_messages)

    idx = list(range(len(history)))
    order = [i for i in idx if not pinned(history[i])] + [i for i in idx if pinned(history[i])]

    for i in order:
        if len(history[i]["text"]) <= TEXT_HEAD + TEXT_TAIL + 40:
            continue
        shrink(i, lambda e: e.__setitem__("text", abridge(e["text"], TEXT_HEAD, TEXT_TAIL)))
        if fits():
            return fitted(history, "texts abridged")

    # Stage 5: collapse old non-pinned messages
    for i in order:
        e = history[i]
        if pinned(e) or not e["text"]:
            continue
        original = len(message_text(messages[e["i"]]))
        shrink(i, lambda e, n=original: e.__setitem__("text", f"[… {n} chars omitted …]"))
        if fits():
            return fitted(history, "old messages collapsed")

    # Stage 6: compact old tool calls to single-line form
    by_msg: Dict[int, List[ToolCall]] = {}
    for c in calls:
        by_msg.setdefault(c.call_index, []).append(c)
    for i in order:
        e = history[i]
        own = by_msg.get(e["i"])
        if pinned(e) or not own:
            continue
        shrink(i, lambda e, own=own: e.__setitem__("tool_calls", [_compact_call(c) for c in own]))
        if fits():
            return fitted(history, "old calls compacted")

    # Stage 7: remove old messages with no content
    left = set()
    for i in order:
        e = history[i]
        if pinned(e) or e.get("tool_calls"):
            continue
        left.add(i)
        tokens -= per[i]
        if fits():
            return fitted([h for j, h in enumerate(history) if j not in left], "old messages left out")

    # Stage 8: merge adjacent call-only messages
    remaining = [h for j, h in enumerate(history) if j not in left]
    merged: List[Dict[str, Any]] = []

    def foldable(e):
        return not pinned(e) and not e["text"] and isinstance((e.get("tool_calls") or [None])[0], str)

    for e in remaining:
        prev = merged[-1] if merged else None
        if prev and foldable(prev) and foldable(e) and prev["role"] == e["role"]:
            prev["tool_calls"] = list(prev["tool_calls"]) + list(e["tool_calls"])
            continue
        merged.append(dict(e))
    history = merged
    per = [entry_tokens(e) for e in history]
    tokens = base + sum(per)
    if fits():
        return fitted(history, "old calls merged")

    raise ValueError(f"history too large for Jev (~{tokens} tokens after truncation, limit {opt.max_state_tokens})")


def questions_for(call: ToolCall) -> Dict[str, Any]:
    """Generate the two Jev questions for one tool call."""
    return {
        f"call_{call.id}": {
            "type": "noul",
            "instructions": (
                f"Tool call {call.id} ({call.tool}) should stay in the history: knowing this call "
                "was made, with its input, still matters for what the assistant does next"
            ),
        },
        f"result_{call.id}": {
            "type": "noul",
            "instructions": (
                f"The full output of tool call {call.id} ({call.tool}, {call.result_chars} chars) should "
                "stay in the history verbatim: the assistant still needs its contents and re-running "
                "the tool would not do"
            ),
        },
    }


def batch_calls(calls: List[ToolCall], state_tokens: int, opt: JevOptions) -> List[List[ToolCall]]:
    """Split calls into batches that fit within max_request_tokens with state and overhead."""
    budget = opt.max_request_tokens - state_tokens - REQUEST_OVERHEAD_TOKENS
    batches: List[List[ToolCall]] = []
    cur: List[ToolCall] = []
    cur_tokens = 0
    for c in calls:
        t = estimate_tokens(_json(questions_for(c)))
        if cur and cur_tokens + t > budget:
            batches.append(cur)
            cur, cur_tokens = [], 0
        if not cur and t > budget:
            raise ValueError(f"state leaves no room for questions (~{state_tokens} of {opt.max_request_tokens} tokens)")
        cur.append(c)
        cur_tokens += t
    if cur:
        batches.append(cur)
    return batches


def decide_call(call: ToolCall, keep_call: float, keep_result: float, opt: JevOptions) -> Decision:
    """Apply thresholds to decide call action."""
    if call.pinned:
        return Decision(id=call.id, tool=call.tool, keep_call=keep_call, keep_result=keep_result, action="keep", reason="pinned")
    if keep_result >= opt.keep_threshold:
        return Decision(id=call.id, tool=call.tool, keep_call=keep_call, keep_result=keep_result, action="keep", reason="kept")
    if keep_call >= opt.keep_threshold:
        return Decision(id=call.id, tool=call.tool, keep_call=keep_call, keep_result=keep_result, action="drop_result", reason="result_dropped")
    return Decision(id=call.id, tool=call.tool, keep_call=keep_call, keep_result=keep_result, action="drop_call", reason="call_dropped")


def _truncated_result(text: str, is_error: bool, head: int) -> str:
    """Truncate a result to head + note."""
    if len(text) <= head + 120:
        return text
    lead = f"{text[:head]}\n" if head > 0 else ""
    return (f"{lead}[fast-jev-compaction truncated {len(text) - head} chars of this tool result"
            f"{' (error)' if is_error else ''}; re-run the tool if needed]")


def apply_decisions(messages, decisions: List[Decision], calls: List[ToolCall], head: int):
    """Rebuild messages after applying compaction decisions."""
    by_id = {c.id: c for c in calls}
    actions: Dict[str, str] = {}
    for d in decisions:
        c = by_id.get(d.id)
        if c and d.action != "keep":
            actions[c.tool_call_id] = d.action
    errors = {c.tool_call_id: c.is_error for c in calls}
    kept = []
    for m in messages:
        if m.get("role") == "tool":
            act = actions.get(m.get("tool_call_id"))
            if act == "drop_call":
                continue
            if act == "drop_result":
                nm = dict(m)
                nm["content"] = _truncated_result(message_text(m), errors.get(m.get("tool_call_id"), False), head)
                kept.append(nm)
                continue
            kept.append(m)
            continue
        tcs = m.get("tool_calls") or []
        if not any(actions.get(tc.get("id")) == "drop_call" for tc in tcs):
            kept.append(m)
            continue
        remaining = [tc for tc in tcs if actions.get(tc.get("id")) != "drop_call"]
        if not remaining and not message_text(m).strip():
            continue
        nm = dict(m)
        if remaining:
            nm["tool_calls"] = remaining
        else:
            nm.pop("tool_calls", None)
        kept.append(nm)
    return kept


def openrouter_asker(
    api_key: Optional[str] = None,
    model: str = OPENROUTER_JEV_MODEL,
    url: str = OPENROUTER_DECISIONS_URL,
    timeout: float = 120.0,
) -> Callable:
    """`ask(state, questions) -> response dict` over OpenRouter Decisions API."""
    key = api_key or os.environ.get("OPENROUTER_API_KEY") or ""
    if not key:
        raise RuntimeError("OPENROUTER_API_KEY is not configured")

    def ask(state, questions):
        body = json.dumps({"model": model, "state": state, "questions": questions}).encode()
        req = Request(url, data=body, headers={
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/NousResearch/hermes-agent",
            "X-Title": "hermes compaction",
        })
        try:
            with urlopen(req, timeout=timeout) as r:
                return json.loads(r.read().decode())
        except HTTPError as e:
            raise RuntimeError(f"Jev request failed ({e.code}): {e.read().decode()[:200]}") from e

    return ask


class JevCompactor:
    """Jev-based compactor for Hermes context_compressor integration."""

    def __init__(
        self,
        asker: Optional[Callable] = None,
        options: Optional[JevOptions] = None,
        concurrency: int = 4,
    ):
        self.asker = asker or openrouter_asker()
        self.opt = options or JevOptions()
        self.concurrency = concurrency
        self.usage = JevUsage()
        self.stats: Dict[str, Any] = {}
        self.decisions: List[Decision] = []

    def _ask_batch(self, state, batch: List[ToolCall]) -> Dict[str, Dict[str, float]]:
        """Ask Jev about one batch of tool calls."""
        questions: Dict[str, Any] = {}
        for c in batch:
            questions.update(questions_for(c))
        resp = self.asker(state, questions)
        answers = resp.get("answers") if isinstance(resp, dict) else None
        if not isinstance(answers, dict):
            raise ValueError("Jev response is missing answers")
        usage = resp.get("usage") or {}
        self.usage.requests += 1
        self.usage.input_tokens += int(usage.get("input_tokens") or 0)
        self.usage.output_tokens += int(usage.get("output_tokens") or 0)
        self.usage.cost_usd += float(usage.get("cost") or 0.0)
        if resp.get("model") and resp["model"] not in self.usage.models:
            self.usage.models.append(resp["model"])

        def noul(name):
            a = answers.get(name)
            if not isinstance(a, dict) or not isinstance(a.get("noul"), (int, float)):
                raise ValueError(f"Invalid Jev answer for {name}: {a}")
            return float(a["noul"])

        return {c.id: {"keep_call": noul(f"call_{c.id}"), "keep_result": noul(f"result_{c.id}")} for c in batch}

    def compact(self, messages: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Run Jev compaction on messages. Returns {"compacted": [...], "stats": {...}}."""
        try:
            # Collect tool calls
            calls = collect_tool_calls(messages, self.opt.preserve_recent_messages)
            if not calls:
                logger.debug("No tool calls to compact with Jev")
                return {"compacted": messages, "stats": {"calls_found": 0}}

            # Fit state within token budget
            fitted = fit_state(messages, calls, self.opt)
            state = fitted["state"]
            state_tokens = fitted["tokens"]
            logger.debug(f"Jev state fitted: {fitted['stage']}, {state_tokens} tokens")
            self.stats["fit_stage"] = fitted["stage"]
            self.stats["state_tokens"] = state_tokens

            # Get candidates (non-pinned calls)
            candidates = [c for c in calls if not c.pinned]
            if not candidates:
                logger.debug("No candidate calls for Jev scoring")
                return {"compacted": messages, "stats": {"calls_found": len(calls), "candidates": 0}}

            # Batch and ask Jev
            batches = batch_calls(candidates, state_tokens, self.opt)
            logger.debug(f"Batched {len(candidates)} calls into {len(batches)} Jev request(s)")

            answers: Dict[str, Dict[str, float]] = {}
            with ThreadPoolExecutor(max_workers=self.concurrency) as executor:
                futures = [executor.submit(self._ask_batch, state, batch) for batch in batches]
                for future in futures:
                    answers.update(future.result())

            # Apply decisions
            self.decisions = [
                decide_call(c, answers[c.id]["keep_call"], answers[c.id]["keep_result"], self.opt)
                for c in candidates
            ] + [Decision(id=c.id, tool=c.tool, keep_call=1.0, keep_result=1.0, action="keep", reason="pinned") for c in calls if c.pinned]

            compacted = apply_decisions(messages, self.decisions, calls, self.opt.truncate_head_chars)
            self.stats["calls_found"] = len(calls)
            self.stats["candidates"] = len(candidates)
            self.stats["decisions"] = {
                "kept": sum(1 for d in self.decisions if d.action == "keep"),
                "result_dropped": sum(1 for d in self.decisions if d.action == "drop_result"),
                "call_dropped": sum(1 for d in self.decisions if d.action == "drop_call"),
            }
            logger.info(f"Jev compaction: {self.stats['decisions']}, usage: {self.usage.requests} req, "
                       f"{self.usage.input_tokens} in, {self.usage.output_tokens} out, ${self.usage.cost_usd:.4f}")
            return {"compacted": compacted, "stats": self.stats}

        except Exception as e:
            logger.exception("Jev compaction failed")
            raise
