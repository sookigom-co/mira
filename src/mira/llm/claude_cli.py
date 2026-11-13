"""Claude Code CLI-backed provider using a Claude subscription login.

Mirrors :mod:`mira.llm.codex_cli`: Mira's chat messages and tool schemas are
rendered into one prompt, run through ``claude -p`` (headless mode), and the
final answer is returned as the same JSON string the HTTP providers produce.

Auth is the Claude Code OAuth token from ``claude setup-token``, passed to the
child process as ``CLAUDE_CODE_OAUTH_TOKEN``. No Anthropic API key is read or
forwarded, so usage counts against the signed-in Claude plan's limits.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import shlex
import tempfile
from pathlib import Path

from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from mira.config import LLMConfig
from mira.exceptions import LLMError
from mira.llm.codex_cli import _SAFE_ENV_KEYS, CodexCLIProvider

logger = logging.getLogger(__name__)

_EFFORTS = frozenset({"low", "medium", "high", "xhigh", "max"})
_DEFAULT_MODELS = frozenset({"", "default", "claude-default"})
_SYSTEM_PROMPT = (
    "You are the model backend of Mira, an automated pull-request reviewer. "
    "You have no tools. Follow the instructions in the user message exactly and "
    "return only the requested final answer."
)


class ClaudeCLIProvider(CodexCLIProvider):
    """LLM provider that shells out to Claude Code in headless mode (``claude -p``)."""

    def __init__(self, config: LLMConfig) -> None:
        super().__init__(config)
        self._last_usage: tuple[int, int] | None = None

    def _oauth_token(self) -> str:
        env_name = self.config.claude_oauth_token_env or "CLAUDE_CODE_OAUTH_TOKEN"
        token = os.environ.get(env_name, "").strip()
        if not token:
            raise LLMError("claude_oauth_token_missing", env=env_name)
        return token

    def _claude_env(self, runtime_home: str) -> dict[str, str]:
        """Minimal child env: no Mira secrets, no ANTHROPIC_API_KEY (would bill the API)."""
        env = {key: value for key, value in os.environ.items() if key in _SAFE_ENV_KEYS}
        env["HOME"] = runtime_home
        env["CLAUDE_CODE_OAUTH_TOKEN"] = self._oauth_token()
        env["DISABLE_AUTOUPDATER"] = "1"
        env["CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC"] = "1"
        return env

    def _claude_command(self) -> list[str]:
        command = self.config.claude_command or "claude"
        if any(char in command for char in (" ", "\t", "\n", ";", "|", "&")):
            raise ValueError(
                f"Invalid claude_command: {command!r}. "
                "Set it to a single executable path/name without arguments."
            )
        cmd = [
            command,
            "-p",
            "--output-format",
            "json",
            "--tools",
            "",
            "--no-session-persistence",
            "--setting-sources",
            "",
            "--strict-mcp-config",
            "--system-prompt",
            _SYSTEM_PROMPT,
        ]
        effort = self.config.reasoning_effort
        if effort in _EFFORTS:
            cmd.extend(["--effort", effort])
        if self.config.model not in _DEFAULT_MODELS:
            cmd.extend(["--model", self.config.model])
        return cmd

    def _parse_result(self, stdout_text: str, stderr_text: str, returncode: int | None) -> str:
        try:
            payload = (
                json.loads(stdout_text.strip().splitlines()[-1]) if stdout_text.strip() else {}
            )
        except (json.JSONDecodeError, IndexError):
            payload = {}

        if not isinstance(payload, dict) or "result" not in payload:
            detail = (stderr_text or stdout_text).strip()
            raise LLMError("claude_exit_failed", exit_code=returncode, detail=detail[-2000:])

        if payload.get("is_error") or returncode != 0:
            raise LLMError(
                "claude_exit_failed",
                exit_code=returncode,
                detail=str(payload.get("result", ""))[-2000:],
            )

        usage = payload.get("usage") or {}
        self._last_usage = (
            int(usage.get("input_tokens", 0))
            + int(usage.get("cache_creation_input_tokens", 0))
            + int(usage.get("cache_read_input_tokens", 0)),
            int(usage.get("output_tokens", 0)),
        )
        return str(payload.get("result", "")).strip()

    @retry(
        stop=stop_after_attempt(2),
        wait=wait_exponential(multiplier=1, min=1, max=5),
        retry=retry_if_exception_type(LLMError),
        reraise=True,
    )
    async def _run_codex(self, prompt: str) -> str:  # name kept: the base class calls it
        self._last_usage = None
        with tempfile.TemporaryDirectory(prefix="mira-claude-") as tmpdir:
            runtime_home = str(Path(tmpdir) / "home")
            Path(runtime_home).mkdir(mode=0o700)
            workdir = str(Path(tmpdir) / "work")
            Path(workdir).mkdir(mode=0o700)
            cmd = self._claude_command()
            logger.debug("Running Claude CLI provider: %s", shlex.join(cmd))
            try:
                proc = await asyncio.create_subprocess_exec(
                    *cmd,
                    stdin=asyncio.subprocess.PIPE,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    env=self._claude_env(runtime_home),
                    cwd=workdir,
                    start_new_session=os.name == "posix",
                )
            except FileNotFoundError as exc:
                raise LLMError(
                    "claude_command_not_found", command=self.config.claude_command
                ) from exc

            try:
                stdout, stderr = await asyncio.wait_for(
                    proc.communicate(prompt.encode("utf-8")),
                    timeout=self.config.claude_timeout_seconds,
                )
            except TimeoutError as exc:
                await self._terminate_process_tree(proc)
                raise LLMError(
                    "claude_timeout", seconds=self.config.claude_timeout_seconds
                ) from exc
            except BaseException:
                await self._terminate_process_tree(proc)
                raise

            return self._parse_result(
                stdout.decode("utf-8", errors="replace"),
                stderr.decode("utf-8", errors="replace"),
                proc.returncode,
            )

    def _messages_prompt(self, messages: list[dict]) -> str:
        return (
            super()
            ._messages_prompt(messages)
            .replace("through Codex CLI.", "through Claude Code.")
            .replace("Do not mention Codex CLI.", "Do not mention Claude Code.")
        )

    def _record_usage(self, prompt: str, result: str) -> None:
        real = self._last_usage
        if real:
            self.total_prompt_tokens += real[0]
            self.total_completion_tokens += real[1]
        else:
            self.total_prompt_tokens += self.count_tokens(prompt)
            self.total_completion_tokens += self.count_tokens(result)

    async def complete(
        self,
        messages: list[dict[str, str]],
        json_mode: bool = True,
        temperature: float | None = None,
        max_tokens: int | None = None,
        *,
        omit_temperature: bool = False,
    ) -> str:
        prompt = self._messages_prompt(messages)
        if json_mode:
            prompt += (
                "\n\n## Required output\n"
                "Return ONLY one valid JSON object. No markdown fences or explanatory text."
            )
        raw = await self._run_codex(prompt)
        result = self._extract_json_object(raw) if json_mode else raw
        self._record_usage(prompt, result)
        return result

    async def complete_with_tools(
        self,
        messages: list[dict[str, str]],
        tools: list[dict],
        temperature: float | None = None,
        force_tool: str | None = None,
    ) -> str:
        prompt = self._tool_prompt(messages, tools, force_tool=force_tool)
        raw = await self._run_codex(prompt)
        result = self._extract_json_object(raw)
        self._record_usage(prompt, result)
        return result
