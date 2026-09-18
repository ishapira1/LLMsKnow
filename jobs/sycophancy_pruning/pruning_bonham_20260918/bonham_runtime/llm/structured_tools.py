from __future__ import annotations

import copy
import hashlib
import json
from typing import Any, Dict, List, Mapping, Optional, Sequence


class StructuredToolTranscriptError(ValueError):
    """Raised when a native tool transcript cannot be represented faithfully."""


_ROLE_ALIASES = {
    "human": "user",
    "user": "user",
    "ai": "assistant",
    "assistant": "assistant",
    "system": "system",
    "developer": "developer",
    "tool": "tool",
}
_ALLOWED_ROLES = frozenset(_ROLE_ALIASES.values())


def _nonempty_string(value: Any, *, field: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise StructuredToolTranscriptError(f"{field} must be a non-empty string")
    return text


def _normalize_tool_call(call: Any, *, message_index: int, call_index: int) -> Dict[str, Any]:
    if not isinstance(call, Mapping):
        raise StructuredToolTranscriptError(
            f"messages[{message_index}].tool_calls[{call_index}] must be an object"
        )
    call_type = str(call.get("type", "function") or "function").strip()
    if call_type != "function":
        raise StructuredToolTranscriptError(
            "Only native function tool calls are supported; "
            f"observed type={call_type!r}"
        )
    function = call.get("function")
    if not isinstance(function, Mapping):
        raise StructuredToolTranscriptError(
            f"messages[{message_index}].tool_calls[{call_index}].function must be an object"
        )
    name = _nonempty_string(
        function.get("name"),
        field=f"messages[{message_index}].tool_calls[{call_index}].function.name",
    )
    arguments = function.get("arguments", {})
    if isinstance(arguments, str):
        try:
            parsed_arguments = json.loads(arguments)
        except json.JSONDecodeError as exc:
            raise StructuredToolTranscriptError(
                f"Tool-call arguments for {name!r} are not valid JSON"
            ) from exc
        if not isinstance(parsed_arguments, Mapping):
            raise StructuredToolTranscriptError(
                f"Tool-call arguments for {name!r} must decode to an object"
            )
        arguments = dict(parsed_arguments)
    elif isinstance(arguments, Mapping):
        arguments = copy.deepcopy(dict(arguments))
    else:
        raise StructuredToolTranscriptError(
            f"Tool-call arguments for {name!r} must be an object or JSON object string"
        )

    normalized: Dict[str, Any] = {
        "type": "function",
        "function": {"name": name, "arguments": arguments},
    }
    call_id = str(call.get("id", "") or "").strip()
    if call_id:
        normalized["id"] = call_id
    return normalized


def normalize_chat_messages(messages: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    """Convert legacy messages while preserving native tool structure.

    Legacy ``type=human|assistant|system`` messages remain supported. Native
    ``role`` messages retain ``tool_calls``, ``tool_call_id``, and tool names;
    unknown roles fail closed instead of being silently converted to user text.
    """

    normalized: List[Dict[str, Any]] = []
    for index, raw_message in enumerate(messages):
        if not isinstance(raw_message, Mapping):
            raise StructuredToolTranscriptError(f"messages[{index}] must be an object")
        raw_role = raw_message.get("role")
        if raw_role is None:
            raw_role = raw_message.get("type")
        role_key = str(raw_role or "").strip().lower()
        role = _ROLE_ALIASES.get(role_key)
        if role not in _ALLOWED_ROLES:
            raise StructuredToolTranscriptError(
                f"messages[{index}] has unsupported role/type {raw_role!r}"
            )

        content_value = raw_message.get("content")
        if content_value is None:
            content = ""
        elif isinstance(content_value, str):
            content = content_value
        else:
            raise StructuredToolTranscriptError(
                f"messages[{index}].content must be a string or null"
            )
        message: Dict[str, Any] = {"role": role, "content": content}

        if "tool_calls" in raw_message:
            if role != "assistant":
                raise StructuredToolTranscriptError(
                    f"messages[{index}].tool_calls is only valid on assistant messages"
                )
            calls = raw_message.get("tool_calls")
            if not isinstance(calls, list) or not calls:
                raise StructuredToolTranscriptError(
                    f"messages[{index}].tool_calls must be a non-empty list"
                )
            message["tool_calls"] = [
                _normalize_tool_call(call, message_index=index, call_index=call_index)
                for call_index, call in enumerate(calls)
            ]

        if role == "tool":
            if not content.strip():
                raise StructuredToolTranscriptError(
                    f"messages[{index}] tool result must have non-empty content"
                )
            tool_call_id = _nonempty_string(
                raw_message.get("tool_call_id"),
                field=f"messages[{index}].tool_call_id",
            )
            name = _nonempty_string(
                raw_message.get("name"),
                field=f"messages[{index}].name",
            )
            message["tool_call_id"] = tool_call_id
            message["name"] = name
        elif "tool_call_id" in raw_message or "name" in raw_message:
            raise StructuredToolTranscriptError(
                f"messages[{index}] tool_call_id/name is only valid on tool messages"
            )

        normalized.append(message)

    return normalized


def normalize_tool_definitions(tools: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    if not isinstance(tools, Sequence) or isinstance(tools, (str, bytes)) or not tools:
        raise StructuredToolTranscriptError("tools must be a non-empty sequence")
    normalized: List[Dict[str, Any]] = []
    names: set[str] = set()
    for index, raw_tool in enumerate(tools):
        if not isinstance(raw_tool, Mapping):
            raise StructuredToolTranscriptError(f"tools[{index}] must be an object")
        tool_type = str(raw_tool.get("type", "function") or "function").strip()
        if tool_type != "function":
            raise StructuredToolTranscriptError(
                f"tools[{index}] has unsupported type={tool_type!r}"
            )
        function = raw_tool.get("function")
        if not isinstance(function, Mapping):
            raise StructuredToolTranscriptError(f"tools[{index}].function must be an object")
        name = _nonempty_string(function.get("name"), field=f"tools[{index}].function.name")
        if name in names:
            raise StructuredToolTranscriptError(f"Duplicate tool definition name {name!r}")
        names.add(name)
        description = str(function.get("description", "") or "")
        parameters = function.get("parameters")
        if not isinstance(parameters, Mapping):
            raise StructuredToolTranscriptError(
                f"tools[{index}].function.parameters must be a JSON-schema object"
            )
        normalized.append(
            {
                "type": "function",
                "function": {
                    "name": name,
                    "description": description,
                    "parameters": copy.deepcopy(dict(parameters)),
                },
            }
        )
    return normalized


def validate_tool_transcript(
    messages: Sequence[Mapping[str, Any]],
    tools: Sequence[Mapping[str, Any]],
) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    normalized_messages = normalize_chat_messages(messages)
    normalized_tools = normalize_tool_definitions(tools)
    tool_names = {tool["function"]["name"] for tool in normalized_tools}

    outstanding: Dict[str, str] = {}
    saw_call = False
    saw_result = False
    for index, message in enumerate(normalized_messages):
        if message["role"] == "assistant" and message.get("tool_calls"):
            for call in message["tool_calls"]:
                name = call["function"]["name"]
                if name not in tool_names:
                    raise StructuredToolTranscriptError(
                        f"messages[{index}] calls undefined tool {name!r}"
                    )
                call_id = str(call.get("id", "") or "").strip()
                if not call_id:
                    raise StructuredToolTranscriptError(
                        f"messages[{index}] tool call {name!r} requires an id"
                    )
                if call_id in outstanding:
                    raise StructuredToolTranscriptError(f"Duplicate tool call id {call_id!r}")
                outstanding[call_id] = name
                saw_call = True
        elif message["role"] == "tool":
            call_id = message["tool_call_id"]
            expected_name = outstanding.pop(call_id, None)
            if expected_name is None:
                raise StructuredToolTranscriptError(
                    f"messages[{index}] refers to unknown or already-consumed tool_call_id {call_id!r}"
                )
            if message["name"] != expected_name:
                raise StructuredToolTranscriptError(
                    f"messages[{index}] tool name {message['name']!r} does not match "
                    f"call {call_id!r} name {expected_name!r}"
                )
            saw_result = True

    if outstanding:
        raise StructuredToolTranscriptError(
            f"Tool transcript has calls without results: {sorted(outstanding)}"
        )
    if not saw_call or not saw_result:
        raise StructuredToolTranscriptError(
            "Native tool transcript requires at least one assistant tool call and matching tool result"
        )
    return normalized_messages, normalized_tools


def audit_native_tool_template(
    tokenizer: Any,
    messages: Sequence[Mapping[str, Any]],
    tools: Sequence[Mapping[str, Any]],
    *,
    add_generation_prompt: bool = True,
) -> Dict[str, Any]:
    """Render and authenticate a native tool transcript before an evaluation run."""

    normalized_messages, normalized_tools = validate_tool_transcript(messages, tools)
    apply_chat_template = getattr(tokenizer, "apply_chat_template", None)
    if not callable(apply_chat_template):
        raise StructuredToolTranscriptError("Tokenizer lacks apply_chat_template")
    try:
        rendered = apply_chat_template(
            normalized_messages,
            tools=normalized_tools,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
        )
    except Exception as exc:
        tokenizer_name = getattr(tokenizer, "name_or_path", tokenizer.__class__.__name__)
        raise StructuredToolTranscriptError(
            f"Tokenizer {tokenizer_name!r} cannot render a native tool transcript: "
            f"{type(exc).__name__}: {exc}"
        ) from exc
    if not isinstance(rendered, str) or not rendered.strip():
        raise StructuredToolTranscriptError("Native tool rendering returned no text")

    tool_results = [
        message["content"] for message in normalized_messages if message["role"] == "tool"
    ]
    function_names = [
        call["function"]["name"]
        for message in normalized_messages
        for call in message.get("tool_calls", [])
    ]
    missing = [value for value in function_names + tool_results if value not in rendered]
    if missing:
        raise StructuredToolTranscriptError(
            "Native tool template silently omitted required transcript content: "
            f"{missing!r}"
        )
    return {
        "supported": True,
        "rendered_sha256": hashlib.sha256(rendered.encode("utf-8")).hexdigest(),
        "message_count": len(normalized_messages),
        "tool_definition_count": len(normalized_tools),
        "tool_call_ids": [
            call["id"]
            for message in normalized_messages
            for call in message.get("tool_calls", [])
        ],
    }


__all__ = [
    "StructuredToolTranscriptError",
    "audit_native_tool_template",
    "normalize_chat_messages",
    "normalize_tool_definitions",
    "validate_tool_transcript",
]

