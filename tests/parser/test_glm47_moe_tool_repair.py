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
tool call itself, including through the serving seam
(:meth:`Glm47MoeParser.extract_tool_calls_from_content`).
"""

import json

import pytest

from vllm.entrypoints.openai.engine.protocol import (
    ExtractedToolCallInformation,
)
from vllm.parser.engine.parser_engine import ParserEngine
from vllm.parser.glm47_moe import (
    Glm47MoeParser,
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
# Dropped tool-call recovery (name, arguments, prose)
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
        name, args, prose = recovered
        assert name == "question"
        assert args == {"questions": [{"q": "a"}]}
        # Pure tool markup -> no assistant prose to keep.
        assert prose is None

    def test_recover_with_explicit_name(self):
        content = ("<tool_call><name>get_weather</name><arguments>"
                   "<arg_key>city</arg_key><arg_value>Paris</arg_value>"
                   "</arguments></tool_call>")
        req = _Req([_tool("get_weather"), _tool("question")])
        recovered = _recover_tool_call(content, req)
        assert recovered is not None
        name, args, prose = recovered
        assert name == "get_weather"
        assert args == {"city": "Paris"}
        assert prose is None

    def test_preserves_leading_prose(self):
        content = ("Let me check the weather.\n"
                   "<function_calls><invoke>"
                   "<parameter>city</arg_key><arg_value>Oslo</arg_value>"
                   "</parameter></invoke></function_calls>")
        req = _Req([_tool("get_weather")])
        recovered = _recover_tool_call(content, req)
        assert recovered is not None
        _name, args, prose = recovered
        assert args == {"city": "Oslo"}
        assert prose == "Let me check the weather."

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

    def test_lone_marker_with_empty_args_not_recovered(self):
        # Single tool + a lone marker must not fabricate an empty-arg call.
        req = _Req([_tool("question")])
        assert (
            _recover_tool_call("See the <tool_call> tag in the docs.", req)
            is None
        )

    def test_prose_mentioning_parameter_not_recovered(self):
        req = _Req([_tool("question")])
        assert (
            _recover_tool_call(
                "The <parameter> element is part of the spec.", req
            )
            is None
        )

    def test_explicit_name_allows_empty_args(self):
        # A real <name> is itself a call even with no arguments.
        content = "<tool_call><name>get_time</name></tool_call>"
        req = _Req([_tool("get_time"), _tool("question")])
        recovered = _recover_tool_call(content, req)
        assert recovered is not None
        name, args, prose = recovered
        assert name == "get_time"
        assert args == {}
        assert prose is None


def _no_tools_stub(self, content, request):
    """Mimic the strict engine failing to recognize any tool call."""
    return ExtractedToolCallInformation(
        tools_called=False, tool_calls=[], content=content
    )


def _no_stream_marker(self, *args, **kwargs):
    """Mimic the strict engine producing no streamed tool calls."""
    return None


# ---------------------------------------------------------------------------
# Streaming repair: finish_streaming emits a well-formed DeltaToolCall
# ---------------------------------------------------------------------------


class TestFinishStreamingRepair:
    def _parser(self, monkeypatch):
        monkeypatch.setattr(
            ParserEngine, "finish_streaming", _no_stream_marker
        )
        return Glm47MoeParser.__new__(Glm47MoeParser)

    def test_repaired_stream_delta_has_id_and_type(self, monkeypatch):
        content = ("<function_calls><invoke><parameter>questions</arg_key>"
                   "<arg_value>[{\"q\":\"a\"}]</arg_value></parameter>"
                   "</invoke></function_calls>")
        p = self._parser(monkeypatch)
        p._repair_stream_text = content
        p._repair_stream_request = _Req([_tool("question")])
        delta = p.finish_streaming()
        assert delta is not None
        assert delta.tool_calls and len(delta.tool_calls) == 1
        call = delta.tool_calls[0]
        assert call.type == "function"
        assert call.id and call.id.startswith("chatcmpl-tool-")
        assert call.function.name == "question"
        assert json.loads(call.function.arguments) == {
            "questions": [{"q": "a"}]
        }

    def test_existing_stream_tool_calls_not_duplicated(self, monkeypatch):
        from vllm.entrypoints.openai.engine.protocol import (
            DeltaFunctionCall,
            DeltaMessage,
            DeltaToolCall,
        )

        def _has_calls(self, *a, **k):
            return DeltaMessage(
                tool_calls=[
                    DeltaToolCall(
                        index=0,
                        id="nope",
                        type="function",
                        function=DeltaFunctionCall(name="already"),
                    )
                ]
            )

        monkeypatch.setattr(ParserEngine, "finish_streaming", _has_calls)
        p = Glm47MoeParser.__new__(Glm47MoeParser)
        p._repair_stream_text = ("<function_calls><invoke>"
                                 "<parameter>x</arg_key>"
                                 "<arg_value>1</arg_value></parameter>"
                                 "</invoke></function_calls>")
        p._repair_stream_request = _Req([_tool("question")])
        delta = p.finish_streaming()
        # Existing tool call wins; recovery must not add a second one.
        assert len(delta.tool_calls) == 1
        assert delta.tool_calls[0].function.name == "already"


# ---------------------------------------------------------------------------
# Serving seam: Glm47MoeParser.extract_tool_calls_from_content
# ---------------------------------------------------------------------------


class TestExtractToolCallsFromContentRepair:
    """Repair applied on the exact method the OpenAI serving path calls."""

    def _parser(self, monkeypatch):
        monkeypatch.setattr(
            ParserEngine,
            "extract_tool_calls_from_content",
            _no_tools_stub,
        )
        # Avoid full engine setup; we only exercise the repair override.
        return Glm47MoeParser.__new__(Glm47MoeParser)

    def test_repairs_leaked_call(self, monkeypatch):
        content = ("<function_calls><invoke><parameter>questions</arg_key>"
                   "<arg_value>[{\"q\":\"a\"}]</arg_value></parameter>"
                   "</invoke></function_calls>")
        req = _Req([_tool("question")])
        info = self._parser(monkeypatch).extract_tool_calls_from_content(
            content, req
        )
        assert info.tools_called is True
        assert info.tool_calls[0].function.name == "question"
        assert json.loads(info.tool_calls[0].function.arguments) == {
            "questions": [{"q": "a"}]
        }
        assert info.content is None

    def test_existing_call_untouched(self, monkeypatch):
        # A successfully parsed call must not be rewritten.
        content = (
            "<tool_call><name>get_weather</name><arguments>"
            "<arg_key>city</arg_key><arg_value>Paris</arg_value>"
            "</arguments></tool_call>"
        )
        req = _Req([_tool("get_weather"), _tool("question")])
        info = self._parser(monkeypatch).extract_tool_calls_from_content(
            content, req
        )
        assert info.tools_called is True
        assert info.tool_calls[0].function.name == "get_weather"

    def test_plain_content_no_repair(self, monkeypatch):
        req = _Req([_tool("question")])
        info = self._parser(monkeypatch).extract_tool_calls_from_content(
            "No tool here.", req
        )
        assert info.tools_called is False

    def test_repair_id_on_toolcall_not_function(self, monkeypatch):
        # id belongs on ToolCall (serialized), not the nested FunctionCall
        # (an internal, exclude=True field).
        content = ("<function_calls><invoke><parameter>questions</arg_key>"
                   "<arg_value>[{\"q\":\"a\"}]</arg_value></parameter>"
                   "</invoke></function_calls>")
        req = _Req([_tool("question")])
        info = self._parser(monkeypatch).extract_tool_calls_from_content(
            content, req
        )
        assert info.tools_called is True
        call = info.tool_calls[0]
        assert call.id and call.id.startswith("chatcmpl-tool-repair-")
        assert getattr(call.function, "id", None) is None

    def test_invalid_recovered_name_dropped(self, monkeypatch):
        p = self._parser(monkeypatch)
        monkeypatch.setattr(
            Glm47MoeParser, "_is_valid_tool_name", lambda self, name: False
        )
        content = ("<function_calls><invoke><parameter>questions</arg_key>"
                   "<arg_value>[{\"q\":\"a\"}]</arg_value></parameter>"
                   "</invoke></function_calls>")
        req = _Req([_tool("question")])
        info = p.extract_tool_calls_from_content(content, req)
        # Not a defined tool -> repair declines, original content preserved.
        assert info.tools_called is False

    def test_repair_runs_schema_coercion(self, monkeypatch):
        p = self._parser(monkeypatch)
        monkeypatch.setattr(
            Glm47MoeParser,
            "_fix_arg_types",
            lambda self, args_json, name: '{"city": "COERCED"}',
        )
        content = ("<function_calls><invoke><parameter>city</arg_key>"
                   "<arg_value>Paris</arg_value></parameter>"
                   "</invoke></function_calls>")
        req = _Req([_tool("get_weather")])
        info = p.extract_tool_calls_from_content(content, req)
        assert info.tools_called is True
        assert info.tool_calls[0].function.arguments == '{"city": "COERCED"}'


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
