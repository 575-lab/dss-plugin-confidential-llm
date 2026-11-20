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

from .errors import UnsupportedRequestError

SUPPORTED_ROLES = ("system", "user", "assistant")

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
            raise UnsupportedRequestError("Message role {!r} is not supported (tool calling is not implemented).".format(role))
        if message.get("toolCalls") or message.get("toolOutputs"):
            raise UnsupportedRequestError("Tool calls are not supported by this connection.")
        text = _message_text(message)
        if role == "system":
            if text:
                system_texts.append(text)
        else:
            result.append({"role": role, "content": text})
    if system_texts:
        result.insert(0, {"role": "system", "content": "\n\n".join(system_texts)})
    return result


def build_request(model, query, settings, stream=False):
    if query.get("tools"):
        raise UnsupportedRequestError("Tool calling is not supported by this connection.")

    body = {"model": model, "messages": to_openai_messages(query.get("messages") or []), "stream": bool(stream)}
    for dss_name, openai_name in SETTINGS_MAP.items():
        value = (settings or {}).get(dss_name)
        if value is not None and value != []:
            body[openai_name] = value
    return body


def to_dss_response(openai_response):
    """Map a non-streamed chat-completions response to a DSS completion response."""
    try:
        choice = openai_response["choices"][0]
        text = choice["message"].get("content") or ""
    except (KeyError, IndexError, TypeError, AttributeError) as exc:
        raise ValueError("Unexpected response from the model server.") from exc

    response = {"text": text}
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
