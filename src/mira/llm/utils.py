"""Shared utilities for LLM output processing and request shaping."""

from __future__ import annotations

import json
import re

# Match a full <think>…</think> block. MiniMax has been seen to close with
# either </think> or </thinking>, so accept both.
_THINK_RE = re.compile(r"<think>.*?</think(?:ing)?>", re.DOTALL)


# Codex-class Responses API backends reject ``text.format`` of ``json_object``
# unless the request carries an explicit JSON-only instruction in a *user-role*
# content item. AxonHub moves system content into Responses ``instructions``
# and its channel selector may route a Chat Completions request to a
# Responses-only upstream, so the Chat builder needs the same guarantee: a
# stray ``json`` substring in a system prompt, function-call argument, or tool
# output does not count.
_JSON_HINT = "Respond with a JSON object only. No markdown fences, no prose."


def _ensure_json_hint(messages: list[dict]) -> list[dict]:
    """Return a copy of ``messages`` carrying ``_JSON_HINT`` in a user message.

    Works for both chat-shaped ``messages`` (``{"role", "content"}``) and
    Responses ``input`` items, whose user/system entries share that shape. The
    hint is appended to the text of the final user message, keeping the
    caller's original content intact; if there is no user message (or its final
    user message carries non-string content), a dedicated user message is
    added. System messages, function-call arguments, and tool outputs are never
    treated as satisfying the contract.

    Caller-owned messages are never mutated: every container is shallow-copied
    before it is touched. Idempotent — a conversation that already contains the
    hint literal is returned as a copy of itself unchanged.
    """
    copied = [dict(msg) for msg in messages]
    for msg in reversed(copied):
        if msg.get("role") != "user":
            continue
        content = msg.get("content")
        if isinstance(content, str):
            if _JSON_HINT in content:
                return copied
            msg["content"] = f"{content}\n\n{_JSON_HINT}" if content else _JSON_HINT
            return copied
        break
    copied.append({"role": "user", "content": _JSON_HINT})
    return copied


def strip_think_blocks(text: str | None) -> str:
    """Remove <think>… reasoning blocks from model output.

    Some models (e.g. MiniMax) output <think>… blocks as part of their
    thinking process before the actual response. These must be stripped
    before JSON parsing.
    """
    if not text:
        return ""
    result = _THINK_RE.sub("", text).strip()
    try:
        idx = next(i for i, c in enumerate(result) if c in "{[")
        obj, _ = json.JSONDecoder().raw_decode(result[idx:])
        return json.dumps(obj)
    except (StopIteration, json.JSONDecodeError):
        return result


_VALID_JSON_ESCAPES = frozenset('"\\/bfnrtu')
_JSON_WHITESPACE = frozenset(" \t\r\n")


def escape_lone_backslashes(text: str) -> str:
    """Double every backslash inside a string literal that does not start a valid JSON escape."""
    out: list[str] = []
    in_string = False
    i, n = 0, len(text)
    while i < n:
        ch = text[i]
        if not in_string:
            if ch == '"':
                in_string = True
            out.append(ch)
            i += 1
        elif ch == "\\":
            nxt = text[i + 1] if i + 1 < n else ""
            if nxt in _VALID_JSON_ESCAPES:
                out.append(ch + nxt)
                i += 2
            else:
                out.append("\\\\")
                i += 1
        else:
            if ch == '"':
                in_string = False
            out.append(ch)
            i += 1
    return "".join(out)


def strip_trailing_commas(text: str) -> str:
    """Drop a comma that is directly followed by ``}`` or ``]`` outside string literals."""
    out: list[str] = []
    in_string = False
    i, n = 0, len(text)
    while i < n:
        ch = text[i]
        if in_string:
            out.append(ch)
            if ch == "\\" and i + 1 < n:
                out.append(text[i + 1])
                i += 2
                continue
            if ch == '"':
                in_string = False
            i += 1
            continue
        if ch == '"':
            in_string = True
        elif ch == ",":
            j = i + 1
            while j < n and text[j] in _JSON_WHITESPACE:
                j += 1
            if j < n and text[j] in "}]":
                i += 1
                continue
        out.append(ch)
        i += 1
    return "".join(out)


def lenient_json_loads(text: str) -> object:
    """Parse model-emitted JSON, repairing lone backslashes and trailing commas; on failure raise the error for the original text."""
    try:
        return json.loads(text, strict=False)
    except json.JSONDecodeError as original:
        first_error = original
    with_escapes = escape_lone_backslashes(text)
    for candidate in (
        with_escapes,
        strip_trailing_commas(text),
        strip_trailing_commas(with_escapes),
    ):
        if candidate == text:
            continue
        try:
            return json.loads(candidate, strict=False)
        except json.JSONDecodeError:
            continue
    raise first_error


def json_error_context(text: str, error: json.JSONDecodeError, radius: int = 160) -> str:
    """Describe where a JSON parse failed: total length plus the text around the error position."""
    pos = max(0, min(error.pos, len(text)))
    window = text[max(0, pos - radius) : pos + radius]
    return f"{error.msg} at char {pos} of {len(text)}; around error: {window!r}"


def strip_code_fences(text: str | None) -> str:
    """Remove markdown code fences wrapping JSON.

    Handles ``None`` input, leading text before the opening fence
    (e.g. LLM analysis preamble), and trailing text after the closing fence.

    When the response contains multiple code blocks (e.g. ``python`` snippets
    in an analysis section followed by a ``json`` result block), only the
    explicitly-tagged ``json`` block is extracted.
    """
    if not text:
        return ""
    text = text.strip()
    # Prefer an explicitly-tagged ```json block anywhere in the response,
    # so we skip unrelated code blocks (```python, etc.) in LLM analysis.
    # Note: re.search scans the entire text, which may be slower for very large
    # responses, but is acceptable for typical LLM output sizes.
    json_match = re.search(r"```json\s*\n?(.*?)\n?\s*```", text, re.DOTALL)
    if json_match:
        return json_match.group(1).strip()
    # Fall back to a generic code fence at the start of the response
    match = re.match(r"^```\s*\n?(.*?)\n?\s*```", text, re.DOTALL)
    return match.group(1).strip() if match else text
