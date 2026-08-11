# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Regression tests for GLM-4.7 tool-call tolerance/repair.

GLM-4.7 occasionally emits a tool call in a hybrid/Anthropic XML wrapper that
the strict ``<arg_key>/<arg_value>`` grammar rejects, e.g.::

    <function_calls><invoke>
      <parameter>key</arg_key><arg_value>value</arg_value></parameter>
    </invoke></function_calls>

When that happens the structural parser drops the call and the raw XML leaks
into content. These tests cover recovery of the arguments and of the dropped
tool call itself.
"""

import json

import pytest

from vllm.parser.glm47_moe import (
    _extract_parameters,
    _glm47_arg_converter,
    _recover_tool_call,
)

pytestmark = pytest.mark.cpu_test


class _Req:
    """Minimal request stub providing a `tools` list."""

    def __init__(self, tools=None):
        self.tools = tools or []


def _tool(name):
    return type("T", (), {"function": type("F", (), {"name": name})()})()


# ---------------------------------------------------------------------------
# Argument extraction tolerance (hybrid / Anthropic forms + JSON decode)
# ---------------------------------------------------------------------------


class TestArgTolerance:
    def test_hybrid_param_arg_value(self):
        raw = "<parameter>questions</arg_key><arg_value>[1,2]</arg_value></parameter>"
        assert _extract_parameters(raw) == {"questions": [1, 2]}

    def test_anthropic_named_param(self):
        raw = '<parameter name="city">San Francisco</parameter>'
        assert _extract_parameters(raw) == {"city": "San Francisco"}

    def test_nested_json_value_decoded(self):
        raw = ("<arg_key>questions</arg_key>"
               "<arg_value>[{\"q\":\"a\"}]</arg_value>")
        assert _extract_parameters(raw) == {"questions": [{"q": "a"}]}

    def test_plain_string_value_not_mangled(self):
        raw = ("<arg_key>city</arg_key>"
               "<arg_value>Seattle</arg_value>")
        assert _extract_parameters(raw) == {"city": "Seattle"}

    def test_converter_accepts_hybrid(self):
        raw = "<parameter>k</arg_key><arg_value>[1,2]</arg_value></parameter>"
        assert json.loads(_glm47_arg_converter(raw, partial=False)) == {"k": [1, 2]}


# ---------------------------------------------------------------------------
# Dropped tool-call recovery
# ---------------------------------------------------------------------------


class TestToolCallRecovery:
    def test_recover_hybrid_garble_single_tool(self):
        # The exact malformed output reported from the field.
        content = ("<function_calls><invoke>"
                   "<parameter>questions</arg_key>"
                   "<arg_value>[{\"q\":\"a\"}]</arg_value>"
                   "</parameter></invoke></function_calls>")
        req = _Req([_tool("question")])
        recovered = _recover_tool_call(content, req)
        assert recovered is not None
        tool_calls, cleaned = recovered
        assert len(tool_calls) == 1
        assert tool_calls[0].name == "question"
        assert json.loads(tool_calls[0].arguments) == {"questions": [{"q": "a"}]}
        # With a name tag, no single-tool assumption is needed.
        assert "<arg_key>" not in (cleaned or "")

    def test_recover_with_explicit_name(self):
        content = ("<tool_call><name>get_weather</name><arguments>"
                   "<arg_key>city</arg_key><arg_value>Paris</arg_value>"
                   "</arguments></tool_call>")
        req = _Req([_tool("get_weather"), _tool("question")])
        recovered = _recover_tool_call(content, req)
        assert recovered is not None
        tool_calls, _cleaned = recovered
        assert tool_calls[0].name == "get_weather"
        assert json.loads(tool_calls[0].arguments) == {"city": "Paris"}

    def test_plain_content_not_recovered(self):
        req = _Req([_tool("question")])
        assert _recover_tool_call("I'll just ask you directly.", req) is None

    def test_no_name_and_multiple_tools_not_recovered(self):
        content = (
            "<function_calls><invoke>"
            "<parameter>x</arg_key><arg_value>1</arg_value></parameter>"
            "</invoke></function_calls>"
        )
        req = _Req([_tool("a"), _tool("b")])
        assert _recover_tool_call(content, req) is None


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
