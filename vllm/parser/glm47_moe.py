# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GLM-4.7 parser for reasoning and tool calls.

GLM-4.7 uses XML-like tool calls::

    <tool_call>func_name<arg_key>key</arg_key><arg_value>value</arg_value></tool_call>

The function name can be followed directly by the first ``<arg_key>`` tag,
and tool calls may have no arguments.
"""

from __future__ import annotations

import functools
import json
import uuid
from typing import TYPE_CHECKING

import regex as re

from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.entrypoints.openai.responses.protocol import ResponsesRequest
from vllm.parser.engine.events import EventType
from vllm.parser.engine.parser_engine import ParserEngine
from vllm.parser.engine.parser_engine_config import (
    ParserEngineConfig,
    ParserState,
    Transition,
)

if TYPE_CHECKING:
    from vllm.entrypoints.openai.engine.protocol import (
        ExtractedToolCallInformation,
    )
    from vllm.tokenizers import TokenizerLike
    from vllm.tool_parsers.abstract_tool_parser import Tool

# Reasoning boundaries. Confirmed against the served GLM-5.2-NVFP4-AQLM
# chat_template.jinja, whose assistant-generation path emits ``' thinking'
# ... ' response'`` (line ~120) when thinking is enabled — i.e. the DeepSeek-ML
# style with a leading space, NOT the ``<think>/</think>`` XML form.
THINK_START = " thinking"
THINK_END = " response"
TOOL_CALL_START = "<tool_call>"
TOOL_CALL_END = "</tool_call>"
ARG_KEY_START = "<arg_key>"
ARG_KEY_END = "</arg_key>"
ARG_VALUE_START = "<arg_value>"
ARG_VALUE_END = "</arg_value>"

_ARG_RE = re.compile(
    r"<arg_key>(?P<key>.*?)</arg_key>\s*"
    r"<arg_value>(?P<value>.*?)</arg_value>",
    re.DOTALL,
)
_PARTIAL_ARG_RE = re.compile(
    r"<arg_key>(?P<key>.*?)</arg_key>\s*"
    r"<arg_value>(?P<value>.*)$",
    re.DOTALL,
)

# GLM can occasionally emit a tool call in a hybrid/Anthropic XML form that
# does not match the strict ``<arg_key>/<arg_value>`` grammar, for example::
#
#     <function_calls><invoke>
#       <parameter>key</arg_key><arg_value>value</arg_value></parameter>
#     </invoke></function_calls>
#
# or the plain Anthropic form ``<parameter name="key">value</parameter>``.
# With thinking enabled and complex tool schemas this happens intermittently;
# without tolerance the whole call is dropped and the raw XML leaks into the
# assistant content. These patterns let the converter / repair recover them.
_HYBRID_ARG_RE = re.compile(
    r"<parameter>"
    r"(?P<key>.*?)</arg_key>\s*<arg_value>(?P<value>.*?)</arg_value>"
    r"\s*</parameter>",
    re.DOTALL,
)
_ANTHROPIC_PARAM_RE = re.compile(
    r'<parameter\s+name="(?P<key>.*?)">(?P<value>.*?)</parameter>', re.DOTALL
)
_NAME_RE = re.compile(r"<name>\s*(?P<name>.*?)\s*</name>", re.DOTALL)
_TOOL_MARKERS = ("<tool_call", "<function_calls", "<invoke", "<arg_key",
                 "<arguments", "<parameter")


def _json_value(value: str) -> object:
    """Decode a value the model emitted as embedded JSON, else keep the str."""
    v = value.strip()
    if v and v[0] in "[{":
        try:
            return json.loads(v)
        except ValueError:
            return v
    return v


def _extract_parameters(raw_args: str) -> dict[str, object]:
    """Best-effort extraction of keyword arguments from any supported form."""
    params: dict[str, object] = {}
    for match in _ARG_RE.finditer(raw_args):
        params[match.group("key").strip()] = _json_value(match.group("value"))
    for match in _HYBRID_ARG_RE.finditer(raw_args):
        params[match.group("key").strip()] = _json_value(match.group("value"))
    for match in _ANTHROPIC_PARAM_RE.finditer(raw_args):
        params[match.group("key").strip()] = _json_value(match.group("value"))
    return params


def _recover_tool_call(
    content: str,
    request,
) -> tuple[str, dict[str, object], str | None] | None:
    """Recover a dropped tool call from leaked native-XML content.

    When the structural state machine never enters tool state (because the
    model emitted a non-conformant wrapper), tool extraction yields no calls
    and leaves the raw XML in the content. If the content clearly contains a
    tool call this rebuilds it so the client can execute it instead of showing
    the raw XML.

    Returns ``(name, arguments, prose)`` where ``prose`` is any genuine
    assistant text that precedes the leaked tool markup (``None`` when the
    content was purely the tool call), or ``None`` when not repairable.
    """
    if not content or not any(m in content for m in _TOOL_MARKERS):
        return None

    positions = [content.find(m) for m in _TOOL_MARKERS if m in content]
    prose = content[:min(positions)].strip() if positions else None
    if prose == "":
        prose = None

    name = None
    match = _NAME_RE.search(content)
    if match and match.group("name").strip():
        name = match.group("name").strip()
    if not name:
        # Some wrapper forms carry no explicit <name>; accept it only when a
        # single tool is defined so the name is unambiguous.
        defined = [
            t.function.name for t in (getattr(request, "tools", None) or [])
            if getattr(getattr(t, "function", None), "name", None)
        ]
        if len(defined) == 1:
            name = defined[0]
        else:
            return None

    return name, _extract_parameters(content), prose


def _glm47_arg_converter(raw_args: str, partial: bool) -> str:
    params: dict[str, object] = {}

    for match in _ARG_RE.finditer(raw_args):
        params[match.group("key").strip()] = match.group("value")

    if partial:
        remaining = _ARG_RE.sub("", raw_args)
        match = _PARTIAL_ARG_RE.search(remaining)
        if match:
            key = match.group("key").strip()
            if key:
                params[key] = match.group("value")
    else:
        # Complete args: accept the hybrid / Anthropic forms and JSON-decode
        # nested values that the strict converter would otherwise mangle.
        params.update(_extract_parameters(raw_args))

    return json.dumps(params, ensure_ascii=False)


@functools.cache
def glm47_moe_config(thinking: bool = True) -> ParserEngineConfig:
    arg_tag_transitions = {
        (ParserState.TOOL_ARGS, terminal): Transition(
            ParserState.TOOL_ARGS,
            (EventType.ARG_VALUE_CHUNK,),
        )
        for terminal in (
            "ARG_KEY_START",
            "ARG_KEY_END",
            "ARG_VALUE_START",
            "ARG_VALUE_END",
        )
    }

    reasoning_terminals = (
        {
            "THINK_START": THINK_START,
            "THINK_END": THINK_END,
        }
        if thinking
        else {}
    )
    reasoning_token_id_terminals = (
        {
            "THINK_START": THINK_START,
            "THINK_END": THINK_END,
        }
        if thinking
        else {}
    )
    reasoning_transitions = (
        {
            (ParserState.CONTENT, "THINK_START"): Transition(
                ParserState.REASONING,
                (EventType.REASONING_START,),
            ),
            (ParserState.REASONING, "THINK_END"): Transition(
                ParserState.CONTENT,
                (EventType.REASONING_END,),
            ),
            (ParserState.CONTENT, "THINK_END"): Transition(
                ParserState.CONTENT,
                (),
            ),
        }
        if thinking
        else {}
    )

    return ParserEngineConfig(
        name="glm47_moe",
        initial_state=ParserState.REASONING if thinking else ParserState.CONTENT,
        terminals={
            **reasoning_terminals,
            "TOOL_START": TOOL_CALL_START,
            "TOOL_END": TOOL_CALL_END,
            "ARG_KEY_START": ARG_KEY_START,
            "ARG_KEY_END": ARG_KEY_END,
            "ARG_VALUE_START": ARG_VALUE_START,
            "ARG_VALUE_END": ARG_VALUE_END,
        },
        token_id_terminals={
            **reasoning_token_id_terminals,
            "TOOL_START": TOOL_CALL_START,
            "TOOL_END": TOOL_CALL_END,
        },
        transitions={
            **reasoning_transitions,
            (ParserState.REASONING, "THINK_START"): Transition(
                ParserState.REASONING,
                (),
            ),
            (ParserState.REASONING, "TOOL_START"): Transition(
                ParserState.TOOL_NAME,
                (EventType.REASONING_END, EventType.TOOL_CALL_START),
            ),
            (ParserState.CONTENT, "TOOL_START"): Transition(
                ParserState.TOOL_NAME,
                (EventType.TOOL_CALL_START,),
            ),
            (ParserState.TOOL_NAME, "ARG_KEY_START"): Transition(
                ParserState.TOOL_ARGS,
                (EventType.ARG_VALUE_CHUNK,),
            ),
            (ParserState.TOOL_NAME, "TOOL_END"): Transition(
                ParserState.CONTENT,
                (EventType.TOOL_CALL_END,),
            ),
            (ParserState.TOOL_ARGS, "TOOL_END"): Transition(
                ParserState.CONTENT,
                (EventType.TOOL_CALL_END,),
            ),
            **arg_tag_transitions,
        },
        arg_converter=_glm47_arg_converter,
        stream_arg_deltas=True,
        tool_args_json=False,
        validate_tool_names=True,
    )


class Glm47MoeParser(ParserEngine):
    """GLM-4.7 parser backed by the declarative parser engine."""

    def __init__(
        self,
        tokenizer: TokenizerLike,
        tools: list[Tool] | None = None,
        **kwargs,
    ) -> None:
        chat_kwargs = kwargs.get("chat_template_kwargs", {}) or {}
        thinking = chat_kwargs.get("thinking", None)
        enable_thinking = chat_kwargs.get("enable_thinking", None)
        self.thinking_enabled = (
            True
            if thinking is None and enable_thinking is None
            else bool(thinking) or bool(enable_thinking)
        )
        kwargs.setdefault(
            "parser_engine_config",
            glm47_moe_config(thinking=self.thinking_enabled),
        )
        super().__init__(tokenizer, tools, **kwargs)

    def extract_tool_calls_from_content(
        self,
        content: str,
        request: ChatCompletionRequest,
    ) -> ExtractedToolCallInformation:
        """Non-streaming tool extraction used by the OpenAI serving path.

        Serving drives `Glm47MoeParserToolAdapter.extract_tool_calls` here (via
        a `DelegatingParser`), NOT through :meth:`parse`. This is therefore the
        seam where a call dropped by the strict state machine must be repaired;
        otherwise the raw tool XML leaks straight into assistant content.
        """
        from vllm.entrypoints.openai.engine.protocol import (
            ExtractedToolCallInformation,
            FunctionCall,
            ToolCall,
        )

        info = super().extract_tool_calls_from_content(content, request)
        if info.tools_called:
            return info
        recovered = _recover_tool_call(content, request)
        if recovered is None:
            return info
        name, args, prose = recovered
        return ExtractedToolCallInformation(
            tools_called=True,
            tool_calls=[
                ToolCall(
                    function=FunctionCall(
                        id=f"chatcmpl-tool-repair-{uuid.uuid4().hex[:12]}",
                        name=name,
                        arguments=json.dumps(args, ensure_ascii=False),
                    )
                )
            ],
            content=prose,
        )

    def parse(
        self,
        model_output: str,
        request,
        enable_auto_tools: bool = False,
        model_output_token_ids=(),
    ) -> tuple[str | None, str | None, list | None]:
        reasoning, content, tool_calls = super().parse(
            model_output, request, enable_auto_tools, model_output_token_ids
        )
        if not tool_calls and content:
            from vllm.entrypoints.openai.engine.protocol import FunctionCall

            recovered = _recover_tool_call(content, request)
            if recovered is not None:
                name, args, prose = recovered
                tool_calls = [
                    FunctionCall(
                        id=f"chatcmpl-tool-repair-{uuid.uuid4().hex[:12]}",
                        name=name,
                        arguments=json.dumps(args, ensure_ascii=False),
                    )
                ]
                content = prose
        return reasoning, content, tool_calls

    def extract_tool_calls_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
        previous_token_ids,
        current_token_ids,
        delta_token_ids,
        request,
    ):
        # Retain the last request + accumulated text so finish_streaming can
        # repair a call that the streaming decoder never recognized.
        self._repair_stream_text = current_text or getattr(
            self, "_repair_stream_text", ""
        )
        self._repair_stream_request = request
        return super().extract_tool_calls_streaming(
            previous_text,
            current_text,
            delta_text,
            previous_token_ids,
            current_token_ids,
            delta_token_ids,
            request,
        )

    def finish_streaming(self):
        from vllm.entrypoints.openai.engine.protocol import (
            DeltaFunctionCall,
            DeltaMessage,
            DeltaToolCall,
        )

        delta = super().finish_streaming()
        text = getattr(self, "_repair_stream_text", "") or ""
        request = getattr(self, "_repair_stream_request", None)
        if text and request is not None and not (
            delta is not None and getattr(delta, "tool_calls", None)
        ):
            recovered = _recover_tool_call(text, request)
            if recovered is not None:
                name, args, prose = recovered
                if delta is None:
                    delta = DeltaMessage()
                if not getattr(delta, "tool_calls", None):
                    delta.tool_calls = [
                        DeltaToolCall(
                            index=0,
                            function=DeltaFunctionCall(
                                name=name,
                                arguments=json.dumps(args, ensure_ascii=False),
                            ),
                        )
                    ]
                if prose:
                    delta.content = prose
                elif hasattr(delta, "content"):
                    delta.content = None
                return delta
        return delta

    def _emit_name_delta(self, idx: int, deltas, name: str | None) -> None:
        if name is not None:
            name = name.strip()
        super()._emit_name_delta(idx, deltas, name)

    def _handle_tool_end(self, event, deltas) -> None:
        idx = event.tool_index
        if 0 <= idx < len(self._tool_slots):
            self._tool_slots[idx].name = self._tool_slots[idx].name.strip()
        super()._handle_tool_end(event, deltas)

    def is_reasoning_end(self, input_ids: list[int]) -> bool:
        if not self.thinking_enabled:
            return True
        return super().is_reasoning_end(input_ids)

    def extract_content_ids(self, input_ids: list[int]) -> list[int]:
        if not self.thinking_enabled:
            return input_ids
        return super().extract_content_ids(input_ids)

    def extract_reasoning(
        self,
        model_output: str,
        request: ChatCompletionRequest | ResponsesRequest,
    ) -> tuple[str | None, str | None]:
        if not self.thinking_enabled:
            return None, model_output
        return super().extract_reasoning(model_output, request)
