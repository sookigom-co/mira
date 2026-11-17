from __future__ import annotations

import json

import pytest

from mira.llm.utils import (
    close_unbalanced_brackets,
    escape_lone_backslashes,
    json_error_context,
    lenient_json_loads,
    strip_trailing_commas,
)

# Shape seen in production: the file object closes but `files` and the root object do not.
_MISSING_ROOT_CLOSERS = (
    '{"files":[{"path":"a.php","language":"php","summary":"S","symbols":[{"name":"f",'
    '"kind":"function","signature":"function f() { }","description":"D"}],"imports":[],'
    '"symbol_references":[{"source":"f","calls":[{"path":"b.php","symbol":"g"}]}],'
    '"external_refs":[{"kind":"git_url","target":"https://x.test/r","description":"E"}]}'
)


class TestStripTrailingCommas:
    def test_drops_comma_before_closing_brace_and_bracket(self):
        assert strip_trailing_commas('{"a": [1, 2,], "b": {"c": 1,},}') == (
            '{"a": [1, 2], "b": {"c": 1}}'
        )

    def test_tolerates_whitespace_between_comma_and_closer(self):
        assert strip_trailing_commas('{"a": 1,\n  }') == '{"a": 1\n  }'

    def test_keeps_commas_inside_strings(self):
        text = '{"sig": ".navbar { a, }", "b": "x,]"}'
        assert strip_trailing_commas(text) == text

    def test_escaped_quote_does_not_end_string(self):
        text = '{"s": "say \\"hi,\\" }", "n": 1}'
        assert strip_trailing_commas(text) == text


class TestCloseUnbalancedBrackets:
    def test_appends_missing_closers_in_nesting_order(self):
        assert close_unbalanced_brackets('{"files":[{"a":[1,2') == '{"files":[{"a":[1,2]}]}'

    def test_balanced_text_unchanged(self):
        text = '{"a": [1, {"b": "c"}]}'
        assert close_unbalanced_brackets(text) == text

    def test_ignores_brackets_inside_strings(self):
        assert (
            close_unbalanced_brackets('{"sig": "f() { [", "n": 1') == '{"sig": "f() { [", "n": 1}'
        )
        assert close_unbalanced_brackets('{"s": "a\\"}]"') == '{"s": "a\\"}]"}'

    def test_unchanged_when_text_ends_inside_a_string(self):
        text = '{"summary": "cut off mid'
        assert close_unbalanced_brackets(text) == text

    def test_unchanged_when_closers_mismatch(self):
        text = '{"a": [1}'
        assert close_unbalanced_brackets(text) == text

    def test_trailing_whitespace_removed_before_closers(self):
        assert close_unbalanced_brackets('{"a": 1 \n') == '{"a": 1}'


class TestLenientJsonLoads:
    def test_valid_json_unchanged(self):
        assert lenient_json_loads('{"a": [1, {"b": "c"}]}') == {"a": [1, {"b": "c"}]}

    def test_repairs_missing_root_closers(self):
        data = lenient_json_loads(_MISSING_ROOT_CLOSERS)
        assert isinstance(data, dict)
        assert [f["path"] for f in data["files"]] == ["a.php"]
        assert data["files"][0]["external_refs"][0]["target"] == "https://x.test/r"

    def test_repairs_dangling_comma_and_missing_closers_together(self):
        data = lenient_json_loads('{"files": [{"path": "a", "symbols": [],},')
        assert data == {"files": [{"path": "a", "symbols": []}]}

    def test_missing_value_after_key_is_not_repaired(self):
        text = '{"files": [{"path": "a", "summary":'
        with pytest.raises(json.JSONDecodeError) as exc:
            lenient_json_loads(text)
        assert exc.value.pos == len(text)

    def test_repair_is_logged_with_original_error_position(self, caplog):
        with caplog.at_level("INFO", logger="mira.llm.utils"):
            lenient_json_loads(_MISSING_ROOT_CLOSERS)
        assert "Repaired model JSON (closed brackets" in caplog.text
        assert (
            f"at char {len(_MISSING_ROOT_CLOSERS)} of {len(_MISSING_ROOT_CLOSERS)}" in caplog.text
        )

    def test_valid_json_logs_nothing(self, caplog):
        with caplog.at_level("INFO", logger="mira.llm.utils"):
            lenient_json_loads('{"a": 1}')
        assert caplog.text == ""

    def test_repairs_trailing_commas(self):
        data = lenient_json_loads('{"files": [{"path": "a.scss", "symbols": [],},],}')
        assert data == {"files": [{"path": "a.scss", "symbols": []}]}

    def test_repairs_lone_backslashes(self):
        data = lenient_json_loads('{"summary": "uses \\App\\Models"}')
        assert data == {"summary": "uses \\App\\Models"}

    def test_repairs_both_defects_together(self):
        data = lenient_json_loads('{"files": [{"path": "C:\\src\\a.py", "summary": "s",},]}')
        assert data == {"files": [{"path": "C:\\src\\a.py", "summary": "s"}]}

    def test_allows_raw_control_characters(self):
        assert lenient_json_loads('{"s": "line1\nline2"}') == {"s": "line1\nline2"}

    def test_unrepairable_raises_error_positioned_on_original_text(self):
        text = '{"files": [{"path": "a" "b"}]}'
        with pytest.raises(json.JSONDecodeError) as exc:
            lenient_json_loads(text)
        assert exc.value.pos == text.index('"b"')

    def test_lone_backslash_helper_keeps_valid_escapes(self):
        assert escape_lone_backslashes('"a\\nb\\\\c\\d"') == '"a\\nb\\\\c\\\\d"'


class TestJsonErrorContext:
    def test_reports_position_length_and_window(self):
        padding = "x" * 600
        text = '{"summary": "' + padding + '", "k": }'
        with pytest.raises(json.JSONDecodeError) as exc:
            json.loads(text)
        context = json_error_context(text, exc.value, radius=20)
        assert f"at char {exc.value.pos} of {len(text)}" in context
        assert '"k": }' in context
        assert padding not in context
