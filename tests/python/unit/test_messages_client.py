import json

import pytest

from confidentialllm import messages
from confidentialllm.client import iter_sse_json
from confidentialllm.errors import UnsupportedRequestError


def test_content_and_text_parts_are_mapped():
    query = {"messages": [
        {"role": "system", "content": "be brief"},
        {"role": "user", "parts": [{"type": "TEXT", "text": "Hello "}, {"type": "TEXT", "text": "world"}]},
        {"role": "assistant", "content": "hi"},
    ]}
    body = messages.build_request("m", query, {})
    assert body["messages"] == [
        {"role": "system", "content": "be brief"},
        {"role": "user", "content": "Hello world"},
        {"role": "assistant", "content": "hi"},
    ]
    assert body["model"] == "m" and body["stream"] is False


def test_system_messages_are_merged_into_one_leading_message():
    # Qwen's chat template raises "System message must be at the beginning" otherwise.
    query = {"messages": [
        {"role": "user", "content": "q1"},
        {"role": "system", "content": "rule A"},
        {"role": "assistant", "content": "a1"},
        {"role": "system", "parts": [{"type": "TEXT", "text": "rule B"}]},
        {"role": "system", "content": ""},
        {"role": "user", "content": "q2"},
    ]}
    assert messages.build_request("m", query, {})["messages"] == [
        {"role": "system", "content": "rule A\n\nrule B"},
        {"role": "user", "content": "q1"},
        {"role": "assistant", "content": "a1"},
        {"role": "user", "content": "q2"},
    ]


def test_no_system_message_is_added_when_none_given():
    query = {"messages": [{"role": "user", "content": "hi"}]}
    assert messages.build_request("m", query, {})["messages"] == [{"role": "user", "content": "hi"}]


def test_settings_are_mapped_and_unset_ones_omitted():
    body = messages.build_request("m", {"messages": []}, {
        "temperature": 0.2, "topP": 0.9, "maxOutputTokens": 50, "stopSequences": ["END"],
        "topK": 40, "frequencyPenalty": None, "seed": 0,
    }, stream=True)
    assert body["temperature"] == 0.2 and body["top_p"] == 0.9 and body["max_tokens"] == 50
    assert body["stop"] == ["END"] and body["seed"] == 0 and body["stream"] is True
    assert "top_k" not in body and "frequency_penalty" not in body


def test_images_and_unknown_roles_are_rejected_explicitly():
    with pytest.raises(UnsupportedRequestError, match="text-only"):
        messages.build_request("m", {"messages": [{"role": "user", "parts": [{"type": "IMAGE_URI", "imageUrl": "x"}]}]}, {})
    with pytest.raises(UnsupportedRequestError, match="role"):
        messages.build_request("m", {"messages": [{"role": "function", "content": "x"}]}, {})
    with pytest.raises(UnsupportedRequestError, match="tool outputs"):
        messages.build_request("m", {"messages": [{"role": "tool", "content": "x"}]}, {})
    with pytest.raises(UnsupportedRequestError, match="function name"):
        messages.build_request("m", {"messages": []}, {"tools": [{"type": "function", "function": {}}]})


def test_tools_from_settings_are_translated():
    settings = {"tools": [
        {"type": "function", "function": {"name": "a", "description": "A", "parameters": {"type": "object", "properties": {"x": {"type": "string"}}}, "compatible": True}},
        {"type": "function", "function": {"name": "b", "parameters": None, "strict": None}},
    ], "toolChoice": {"type": "tool_name", "name": "a"}}
    body = messages.build_request("m", {"messages": []}, settings)
    assert body["tools"] == [
        {"type": "function", "function": {"name": "a", "description": "A", "parameters": {"type": "object", "properties": {"x": {"type": "string"}}}}},
        {"type": "function", "function": {"name": "b", "parameters": {"type": "object", "properties": {}}}},
    ]
    assert body["tool_choice"] == {"type": "function", "function": {"name": "a"}}
    for dss, openai in (("auto", "auto"), ("none", "none"), ("required", "required")):
        assert messages.build_request("m", {"messages": []}, dict(settings, toolChoice={"type": dss}))["tool_choice"] == openai
    assert "tool_choice" not in messages.build_request("m", {"messages": []}, {"tools": settings["tools"]})
    assert "tools" not in messages.build_request("m", {"messages": []}, {"tools": []})


def test_tool_call_history_is_translated():
    query = {"messages": [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "q"},
        {"role": "assistant", "toolCalls": [{"type": "function", "id": "c1", "function": {"name": "a", "arguments": "{\"x\": 1}"}},
                                            {"type": "function", "id": "c2", "function": {"name": "b", "arguments": {"y": 2}}}]},
        {"role": "tool", "toolOutputs": [{"callId": "c1", "output": "r1"}, {"callId": "c2", "parts": [{"type": "TEXT", "text": "r2"}]}]},
        {"role": "system", "content": "late"},
    ]}
    assert messages.build_request("m", query, {})["messages"] == [
        {"role": "system", "content": "sys\n\nlate"},
        {"role": "user", "content": "q"},
        {"role": "assistant", "content": None, "tool_calls": [
            {"id": "c1", "type": "function", "function": {"name": "a", "arguments": "{\"x\": 1}"}},
            {"id": "c2", "type": "function", "function": {"name": "b", "arguments": "{\"y\": 2}"}}]},
        {"role": "tool", "tool_call_id": "c1", "content": "r1"},
        {"role": "tool", "tool_call_id": "c2", "content": "r2"},
    ]


def test_tool_calls_in_response_are_mapped():
    r = messages.to_dss_response({"choices": [{"finish_reason": "tool_calls", "message": {"content": None, "tool_calls": [
        {"id": "c1", "type": "function", "function": {"name": "a", "arguments": "{}"}}]}}]})
    assert r == {"text": "", "finishReason": "tool_calls",
                 "toolCalls": [{"type": "function", "id": "c1", "function": {"name": "a", "arguments": "{}"}}]}
    assert messages.to_dss_response({"choices": [{"finish_reason": "weird", "message": {"content": "x"}}]})["finishReason"] == "unknown"


def test_stream_event_mapping():
    assert messages.stream_event_to_dss({"choices": [{"delta": {"content": "Hi"}}]}) == ({"text": "Hi"}, None)
    assert messages.stream_event_to_dss({"choices": [{"delta": {"reasoning": "t"}}]}) == (None, None)
    assert messages.stream_event_to_dss({"choices": [{"delta": {}, "finish_reason": "length"}]}) == (None, "length")
    assert messages.stream_event_to_dss({}) == (None, None)
    # Ollama sends whole tool calls without an index: position in the list is used.
    chunk, _ = messages.stream_event_to_dss({"choices": [{"delta": {"tool_calls": [
        {"id": "c1", "function": {"name": "a", "arguments": "{}"}}, {"id": "c2", "function": {"name": "b", "arguments": "{}"}}]}}]})
    assert [c["index"] for c in chunk["toolCalls"]] == [0, 1]


def test_response_mapping_with_and_without_usage():
    r = messages.to_dss_response({"choices": [{"message": {"content": "ok"}}],
                                  "usage": {"prompt_tokens": 3, "completion_tokens": 5}})
    assert r == {"text": "ok", "promptTokens": 3, "completionTokens": 5}
    assert messages.to_dss_response({"choices": [{"message": {"content": None}}]}) == {"text": ""}
    with pytest.raises(ValueError):
        messages.to_dss_response({"error": "boom"})


def test_stream_delta_ignores_reasoning_and_role_events():
    assert messages.stream_delta_text({"choices": [{"delta": {"content": "Hel"}}]}) == "Hel"
    assert messages.stream_delta_text({"choices": [{"delta": {"role": "assistant"}}]}) == ""
    assert messages.stream_delta_text({"choices": [{"delta": {"reasoning": "thinking..."}}]}) == ""
    assert messages.stream_delta_text({"choices": []}) == ""
    assert messages.stream_delta_text({}) == ""


def sse(*objs, done=True):
    out = b"".join(b"data: " + json.dumps(o).encode() + b"\n\n" for o in objs)
    return out + (b"data: [DONE]\n\n" if done else b"")


def test_sse_events_split_across_arbitrary_chunk_boundaries():
    raw = sse({"n": 1}, {"n": 2}, {"n": 3})
    for size in (1, 2, 5, 7, len(raw)):
        chunks = [raw[i:i + size] for i in range(0, len(raw), size)]
        assert [e["n"] for e in iter_sse_json(chunks)] == [1, 2, 3]


def test_sse_stops_at_done_and_ignores_comments_and_crlf():
    raw = b": keep-alive\r\n\r\ndata: {\"n\": 1}\r\n\r\ndata: [DONE]\n\ndata: {\"n\": 99}\n\n"
    assert [e["n"] for e in iter_sse_json([raw])] == [1]


def test_sse_final_event_without_trailing_newline():
    assert [e["n"] for e in iter_sse_json([b'data: {"n": 7}'])] == [7]


def test_thinking_and_default_max_tokens():
    body = messages.build_request("m", {"messages": []}, {}, thinking="off", default_max_tokens=8192)
    assert body["reasoning_effort"] == "none" and body["max_tokens"] == 8192
    body = messages.build_request("m", {"messages": []}, {"maxOutputTokens": 100}, default_max_tokens=8192)
    assert body["max_tokens"] == 100 and "reasoning_effort" not in body
    assert "max_tokens" not in messages.build_request("m", {"messages": []}, {}, default_max_tokens=None)
