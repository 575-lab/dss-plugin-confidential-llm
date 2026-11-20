# Copyright 2026 Dataiku SAS
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Translate between DSS LLM Mesh completion objects and the OpenAI chat-completions format."""

import json

from .errors import UnsupportedRequestError

SUPPORTED_ROLES = ("system", "user", "assistant", "tool")

# DSS completion setting -> OpenAI request field. topK has no OpenAI equivalent and is ignored.
SETTINGS_MAP = {
    "temperature": "temperature",
    "topP": "top_p",
    "maxOutputTokens": "max_tokens",
    "stopSequences": "stop",
    "frequencyPenalty": "frequency_penalty",
    "presencePenalty": "presence_penalty",
    "seed": "seed",
}

# OpenAI finish_reason values DSS knows (it serializes its enum in lowercase); anything else is "unknown".
FINISH_REASONS = ("stop", "length", "tool_calls", "content_filter")


def _message_text(message):
    if message.get("content") is not None:
        return message["content"]

    texts = []
    for part in message.get("parts") or []:
        part_type = str(part.get("type", "")).upper()
        if part_type == "TEXT":
            texts.append(part.get("text") or "")
        else:
            raise UnsupportedRequestError(
                "This connection serves a text-only model; message part type {!r} is not supported.".format(part.get("type")))
    return "".join(texts)


def _to_openai_tool_call(call):
    function = call.get("function") or {}
    arguments = function.get("arguments")
    if not isinstance(arguments, str):
        arguments = json.dumps(arguments if arguments is not None else {})
    return {"id": call.get("id"), "type": "function", "function": {"name": function.get("name"), "arguments": arguments}}


def _to_openai_tool_outputs(message):
    outputs = message.get("toolOutputs") or []
    if not outputs:
        raise UnsupportedRequestError("A tool message without tool outputs is not supported.")
    result = []
    for output in outputs:
        content = output.get("output")
        if content is None:
            content = _message_text({"parts": output.get("parts")})
        elif not isinstance(content, str):
            content = json.dumps(content)
        result.append({"role": "tool", "tool_call_id": output.get("callId"), "content": content})
    return result


def to_openai_messages(messages):
    """Convert DSS messages to OpenAI ones.

    Many chat templates (Qwen among them) reject a system message anywhere but first, while DSS can send
    several (e.g. Prompt Studio instructions plus a user-provided system prompt). All system messages are
    therefore merged, in order, into a single leading system message.
    """
    system_texts = []
    result = []
    for message in messages:
        role = message.get("role")
        if role not in SUPPORTED_ROLES:
            raise UnsupportedRequestError("Message role {!r} is not supported.".format(role))
        if role == "tool":
            result.extend(_to_openai_tool_outputs(message))
            continue
        text = _message_text(message)
        if role == "system":
            if text:
                system_texts.append(text)
        elif role == "assistant" and message.get("toolCalls"):
            result.append({"role": "assistant", "content": text or None,
                           "tool_calls": [_to_openai_tool_call(c) for c in message["toolCalls"]]})
        else:
            result.append({"role": role, "content": text})
    if system_texts:
        result.insert(0, {"role": "system", "content": "\n\n".join(system_texts)})
    return result


def _to_openai_tools(tools):
    result = []
    for tool in tools:
        function = dict(tool.get("function") or {})
        if not function.get("name"):
            raise UnsupportedRequestError("A tool without a function name is not supported.")
        function.pop("compatible", None)  # DSS-only field
        if function.get("parameters") is None:
            function["parameters"] = {"type": "object", "properties": {}}
        if function.get("description") is None:
            function.pop("description", None)
        if function.get("strict") is None:
            function.pop("strict", None)
        result.append({"type": "function", "function": function})
    return result


def _to_openai_tool_choice(choice):
    choice_type = str((choice or {}).get("type", "")).lower()
    if choice_type in ("auto", "none", "required"):
        return choice_type
    if choice_type == "tool_name" and choice.get("name"):
        return {"type": "function", "function": {"name": choice["name"]}}
    return None


def build_request(model, query, settings, stream=False):
    settings = settings or {}
    body = {"model": model, "messages": to_openai_messages(query.get("messages") or []), "stream": bool(stream)}
    for dss_name, openai_name in SETTINGS_MAP.items():
        value = settings.get(dss_name)
        if value is not None and value != []:
            body[openai_name] = value
    # DSS puts tool definitions in the completion settings (older versions may put them on the query).
    tools = settings.get("tools") or query.get("tools")
    if tools:
        body["tools"] = _to_openai_tools(tools)
        tool_choice = _to_openai_tool_choice(settings.get("toolChoice") or query.get("toolChoice"))
        if tool_choice is not None:
            body["tool_choice"] = tool_choice
    return body


def _finish_reason(value):
    return value if value in FINISH_REASONS else "unknown"


def _to_dss_tool_call(call, index=None):
    function = call.get("function") or {}
    result = {"type": "function", "function": {}}
    if call.get("id"):
        result["id"] = call["id"]
    if function.get("name"):
        result["function"]["name"] = function["name"]
    arguments = function.get("arguments")
    if arguments is not None:
        result["function"]["arguments"] = arguments if isinstance(arguments, str) else json.dumps(arguments)
    if index is not None:
        result["index"] = index
    return result


def to_dss_response(openai_response):
    """Map a non-streamed chat-completions response to a DSS completion response."""
    try:
        choice = openai_response["choices"][0]
        message = choice["message"]
        text = message.get("content") or ""
    except (KeyError, IndexError, TypeError, AttributeError) as exc:
        raise ValueError("Unexpected response from the model server.") from exc

    response = {"text": text}
    tool_calls = message.get("tool_calls") or []
    if tool_calls:
        response["toolCalls"] = [_to_dss_tool_call(c) for c in tool_calls]
    if choice.get("finish_reason"):
        response["finishReason"] = _finish_reason(choice["finish_reason"])
    elif tool_calls:
        response["finishReason"] = "tool_calls"
    usage = openai_response.get("usage") or {}
    if "prompt_tokens" in usage:
        response["promptTokens"] = usage["prompt_tokens"]
    if "completion_tokens" in usage:
        response["completionTokens"] = usage["completion_tokens"]
    return response


def stream_delta_text(event):
    """Return the answer text carried by one streamed event, or '' (e.g. reasoning-only or role events)."""
    try:
        delta = event["choices"][0].get("delta") or {}
    except (KeyError, IndexError, TypeError, AttributeError):
        return ""
    return delta.get("content") or ""


def stream_event_to_dss(event):
    """Map one streamed chat-completions event to ``(chunk or None, finish_reason or None)``.

    Tool-call deltas keep their ``index`` so DSS merges argument fragments the way OpenAI clients do.
    """
    try:
        choice = event["choices"][0]
        delta = choice.get("delta") or {}
    except (KeyError, IndexError, TypeError, AttributeError):
        return None, None

    chunk = {}
    text = delta.get("content")
    if text:
        chunk["text"] = text
    tool_calls = delta.get("tool_calls") or []
    if tool_calls:
        chunk["toolCalls"] = [_to_dss_tool_call(c, c.get("index", i)) for i, c in enumerate(tool_calls)]
    finish = choice.get("finish_reason")
    return (chunk or None), (_finish_reason(finish) if finish else None)
