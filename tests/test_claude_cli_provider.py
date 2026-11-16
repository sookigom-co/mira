from __future__ import annotations

import json
import stat
from pathlib import Path

import pytest

from mira.config import LLMConfig, _strip_deployment_only_settings
from mira.dashboard.model_catalog import active_backend
from mira.exceptions import LLMError
from mira.llm import create_llm
from mira.llm.claude_cli import ClaudeCLIProvider
from mira.llm.codex_cli import CodexCLIProvider

FAKE_CLAUDE = """#!/usr/bin/env python3
import json, os, sys
prompt = sys.stdin.read()
log = os.environ.get("PATH").split(":")[0] + "/calls.jsonl"
with open(log, "a") as f:
    f.write(json.dumps({"argv": sys.argv[1:], "env": dict(os.environ), "prompt": prompt}) + "\\n")
mode_file = os.path.join(os.path.dirname(log), "mode")
mode = open(mode_file).read().strip()
if mode == "structured":
    assert "--json-schema" in sys.argv
    print(json.dumps({"type": "result", "is_error": False, "result": "",
                      "structured_output": {"comments": [], "summary": "say \\"hi\\""},
                      "usage": {"input_tokens": 5, "output_tokens": 2}}))
elif mode == "bad_then_ok":
    open(mode_file, "w").write("ok")
    print(json.dumps({"type": "result", "is_error": False,
                      "result": '{"files": [{"path": "a" "b"}]}', "usage": {}}))
elif mode == "ctrlchar":
    print(json.dumps({"type": "result", "is_error": False,
                      "result": '{"files": [{"path": "a", "summary": "line1\\nline2"}]}',
                      "usage": {}}))
elif mode == "ok":
    result = '```json\\n{"comments": [], "summary": "ok"}\\n```'
    print(json.dumps({"type": "result", "is_error": False, "result": result,
                      "usage": {"input_tokens": 100, "cache_read_input_tokens": 20,
                                "output_tokens": 7}}))
elif mode == "notlogged":
    print(json.dumps({"type": "result", "is_error": True,
                      "result": "Not logged in · Please run /login", "usage": {}}))
    sys.exit(1)
else:
    sys.stderr.write("boom")
    sys.exit(2)
"""


@pytest.fixture
def fake_claude(tmp_path, monkeypatch):
    bindir = tmp_path / "bin"
    bindir.mkdir()
    exe = bindir / "claude"
    exe.write_text(FAKE_CLAUDE)
    exe.chmod(exe.stat().st_mode | stat.S_IEXEC)
    (bindir / "mode").write_text("ok")
    monkeypatch.setenv("PATH", f"{bindir}:/usr/bin:/bin")
    monkeypatch.setenv("CLAUDE_CODE_OAUTH_TOKEN", "sk-ant-oat01-test")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "must-not-leak")
    monkeypatch.setenv("MIRA_WEBHOOK_SECRET", "must-not-leak")
    return bindir


def _calls(bindir: Path) -> list[dict]:
    return [json.loads(line) for line in (bindir / "calls.jsonl").read_text().splitlines()]


def _provider(**kw) -> ClaudeCLIProvider:
    return ClaudeCLIProvider(LLMConfig(provider="claude-cli", **kw))


def test_factory_and_backend():
    config = LLMConfig(provider="claude-cli", model="sonnet")
    assert isinstance(create_llm(config), ClaudeCLIProvider)
    assert active_backend(config) == "claude-cli"


def test_command_flags():
    cmd = _provider(model="sonnet", reasoning_effort="high")._claude_command()
    assert cmd[:2] == ["claude", "-p"]
    assert cmd[cmd.index("--tools") + 1] == ""
    assert cmd[cmd.index("--output-format") + 1] == "json"
    assert cmd[cmd.index("--effort") + 1] == "high"
    assert cmd[cmd.index("--model") + 1] == "sonnet"
    assert "--no-session-persistence" in cmd
    default = _provider(model="claude-default", reasoning_effort="off")._claude_command()
    assert "--model" not in default and "--effort" not in default


def test_rejects_command_with_arguments():
    with pytest.raises(ValueError):
        _provider(claude_command="claude --dangerously-skip-permissions")._claude_command()


def test_repo_overlay_cannot_set_execution_keys():
    cleaned = _strip_deployment_only_settings(
        {"llm": {"provider": "claude-cli", "claude_command": "/tmp/evil", "model": "haiku"}}
    )
    assert cleaned["llm"] == {"model": "haiku"}


@pytest.mark.asyncio
async def test_review_roundtrip_uses_oauth_only(fake_claude):
    provider = _provider(model="sonnet")
    out = await provider.review([{"role": "user", "content": "diff here"}])
    assert json.loads(out) == {"comments": [], "summary": "ok"}
    call = _calls(fake_claude)[0]
    assert call["env"]["CLAUDE_CODE_OAUTH_TOKEN"] == "sk-ant-oat01-test"
    assert "ANTHROPIC_API_KEY" not in call["env"]
    assert "MIRA_WEBHOOK_SECRET" not in call["env"]
    assert "diff here" in call["prompt"] and "Claude Code" in call["prompt"]
    assert provider.usage == {"prompt_tokens": 120, "completion_tokens": 7, "total_tokens": 127}


@pytest.mark.asyncio
async def test_not_logged_in_raises(fake_claude):
    (fake_claude / "mode").write_text("notlogged")
    with pytest.raises(LLMError) as exc:
        await _provider().complete([{"role": "user", "content": "x"}])
    assert "Not logged in" in str(exc.value)


@pytest.mark.asyncio
async def test_crash_raises(fake_claude):
    (fake_claude / "mode").write_text("crash")
    with pytest.raises(LLMError) as exc:
        await _provider().complete([{"role": "user", "content": "x"}])
    assert "boom" in str(exc.value)


@pytest.mark.asyncio
async def test_missing_token(fake_claude, monkeypatch):
    monkeypatch.delenv("CLAUDE_CODE_OAUTH_TOKEN")
    with pytest.raises(LLMError):
        await _provider().complete([{"role": "user", "content": "x"}])


@pytest.mark.asyncio
async def test_structured_output_preferred(fake_claude):
    (fake_claude / "mode").write_text("structured")
    out = await _provider(model="sonnet").review([{"role": "user", "content": "d"}])
    assert json.loads(out)["summary"] == 'say "hi"'
    argv = _calls(fake_claude)[0]["argv"]
    schema = json.loads(argv[argv.index("--json-schema") + 1])
    assert schema.get("type") == "object" and schema.get("properties")


@pytest.mark.asyncio
async def test_malformed_json_is_retried(fake_claude):
    (fake_claude / "mode").write_text("bad_then_ok")
    out = await _provider().complete([{"role": "user", "content": "x"}])
    assert json.loads(out)["summary"] == "ok"
    assert len(_calls(fake_claude)) == 2


@pytest.mark.asyncio
async def test_raw_control_chars_tolerated(fake_claude):
    (fake_claude / "mode").write_text("ctrlchar")
    out = await _provider().complete([{"role": "user", "content": "x"}])
    assert json.loads(out)["files"][0]["summary"] == "line1\nline2"


@pytest.mark.asyncio
async def test_review_prompt_names_claude_not_codex(fake_claude):
    await _provider(model="sonnet").review([{"role": "user", "content": "diff here"}])
    prompt = _calls(fake_claude)[0]["prompt"]
    assert "Claude Code" in prompt and "Codex" not in prompt


def test_tool_prompt_names_claude_not_codex():
    tools = [{"type": "function", "function": {"name": "submit_result", "parameters": {}}}]
    prompt = _provider()._tool_prompt([{"role": "user", "content": "x"}], tools)
    assert "Claude Code" in prompt and "Codex" not in prompt


def test_codex_prompt_wording_unchanged():
    prompt = CodexCLIProvider(LLMConfig(provider="codex-cli"))._messages_prompt([])
    assert prompt.startswith(
        "You are running as Mira's model backend through Codex CLI.\n"
        "Follow the Mira review instructions exactly. Do not mention Codex CLI.\n"
    )


def test_structured_output_can_be_disabled():
    cmd = _provider(claude_structured_output=False)._claude_command({"type": "object"})
    assert "--json-schema" not in cmd
