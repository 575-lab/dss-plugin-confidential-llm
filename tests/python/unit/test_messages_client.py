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


def test_settings_are_mapped_and_unset_ones_omitted():
    body = messages.build_request("m", {"messages": []}, {
        "temperature": 0.2, "topP": 0.9, "maxOutputTokens": 50, "stopSequences": ["END"],
        "topK": 40, "frequencyPenalty": None, "seed": 0,
    }, stream=True)
    assert body["temperature"] == 0.2 and body["top_p"] == 0.9 and body["max_tokens"] == 50
    assert body["stop"] == ["END"] and body["seed"] == 0 and body["stream"] is True
    assert "top_k" not in body and "frequency_penalty" not in body


def test_images_tools_and_unknown_roles_are_rejected_explicitly():
    with pytest.raises(UnsupportedRequestError, match="text-only"):
        messages.build_request("m", {"messages": [{"role": "user", "parts": [{"type": "IMAGE_URI", "imageUrl": "x"}]}]}, {})
    with pytest.raises(UnsupportedRequestError, match="Tool"):
        messages.build_request("m", {"messages": [], "tools": [{"name": "t"}]}, {})
    with pytest.raises(UnsupportedRequestError, match="role"):
        messages.build_request("m", {"messages": [{"role": "tool", "content": "x"}]}, {})
    with pytest.raises(UnsupportedRequestError, match="Tool calls"):
        messages.build_request("m", {"messages": [{"role": "assistant", "content": "x", "toolCalls": [{}]}]}, {})


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
