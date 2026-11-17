from __future__ import annotations

import json

import pytest

from mira.llm.utils import (
    escape_lone_backslashes,
    json_error_context,
    lenient_json_loads,
    strip_trailing_commas,
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


class TestLenientJsonLoads:
    def test_valid_json_unchanged(self):
        assert lenient_json_loads('{"a": [1, {"b": "c"}]}') == {"a": [1, {"b": "c"}]}

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
