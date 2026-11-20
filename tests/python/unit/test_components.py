"""Exercise the DSS components (llm.py, runnable.py) with a stand-in `dataiku` package.

These files only run inside DSS, so without this nothing would catch errors in them (a hashing bug in the
session cache reached a real DSS before this existed).
"""
import asyncio
import contextlib
import importlib.util
import json
import os
import sys
import types

import pytest

from confidentialllm import attest, client, session as session_module
from confidentialllm.attest import AttestationResult
from confidentialllm.config import EndpointConfig

ROOT = os.path.join(os.path.dirname(__file__), "..", "..", "..")
H = "ab" * 48
CA_PEM = None


class FakeSpan:
    def __init__(self):
        self.attributes = {}


class FakeTrace:
    def __init__(self):
        self.spans = []

    @contextlib.contextmanager
    def subspan(self, name):
        span = FakeSpan()
        self.spans.append((name, span))
        yield span


@pytest.fixture(autouse=True)
def fake_dataiku(monkeypatch):
    class BaseLLM:  # what dataiku.llm.python.BaseLLM provides: a plain base class
        pass

    class Runnable:
        pass

    pkg = types.ModuleType("dataiku")
    llm_pkg, llm_python = types.ModuleType("dataiku.llm"), types.ModuleType("dataiku.llm.python")
    llm_python.BaseLLM = BaseLLM
    runnables = types.ModuleType("dataiku.runnables")
    runnables.Runnable = Runnable
    for name, module in {"dataiku": pkg, "dataiku.llm": llm_pkg, "dataiku.llm.python": llm_python,
                         "dataiku.runnables": runnables}.items():
        monkeypatch.setitem(sys.modules, name, module)
    session_module._SESSIONS.clear()


def load(relative_path, name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(ROOT, relative_path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def preset(**overrides):
    base = {"gateway_url": "https://gw.example.com:8443", "pins": json.dumps({"mr_td": H}),
            "gateway_token": "tok", "ca_certificate": _ca()}
    base.update(overrides)
    return base


def _ca():
    """One real self-signed certificate for the whole module (so equal presets really are equal)."""
    global CA_PEM
    if CA_PEM is None:
        from test_tls import make_cert
        import pathlib, tempfile
        CA_PEM = make_cert(pathlib.Path(tempfile.mkdtemp()), "c")[0].read_text()
    return CA_PEM


# ---- the cache key: this is the bug that reached DSS ----

def test_cache_key_is_hashable_with_pins_and_all_fields():
    cfg = EndpointConfig.from_preset(preset(pins=json.dumps([{"mr_td": H}, {"mr_td": "cd" * 48, "rtmr0": H}])))
    hash(cfg.cache_key)  # raised TypeError: unhashable type: 'dict'


def test_get_session_is_shared_for_equal_settings_and_separate_for_different_pins():
    a = EndpointConfig.from_preset(preset())
    b = EndpointConfig.from_preset(preset())
    c = EndpointConfig.from_preset(preset(pins=json.dumps({"mr_td": "cd" * 48})))
    assert session_module.get_session(a) is session_module.get_session(b)
    assert session_module.get_session(a) is not session_module.get_session(c)


def test_pin_order_inside_a_set_does_not_split_sessions():
    one = EndpointConfig.from_preset(preset(pins='{"mr_td": "%s", "rtmr0": "%s"}' % (H, H)))
    two = EndpointConfig.from_preset(preset(pins='{"rtmr0": "%s", "mr_td": "%s"}' % (H, H)))
    assert session_module.get_session(one) is session_module.get_session(two)


# ---- llm.py ----

def make_llm(**cfg):
    module = load("python-llms/confidential-llm/llm.py", "llm_under_test")
    llm = module.ConfidentialLLM()
    llm.set_config({"model": "m", "endpoint": preset(**cfg)}, {})
    return module, llm


def fake_result():
    return AttestationResult(b"k", 1.0, True, {"tcb_status": "UpToDate", "key_fingerprint": "f",
                                              "measurements": {"mr_td": H}, "gpus": [{"gpu": "GPU-0"}]})


def test_llm_set_config_with_a_real_preset_does_not_raise():
    make_llm()  # exactly what DSS does at "Failed to start custom LLM server"


def test_llm_set_config_rejects_missing_pins_and_missing_model():
    module = load("python-llms/confidential-llm/llm.py", "llm_under_test2")
    with pytest.raises(Exception, match="pinned"):
        module.ConfidentialLLM().set_config({"model": "m", "endpoint": {"gateway_url": "https://gw"}}, {})
    with pytest.raises(ValueError, match="model"):
        module.ConfidentialLLM().set_config({"endpoint": preset()}, {})


def test_llm_process_returns_text_and_records_attestation_in_the_trace(monkeypatch):
    module, llm = make_llm()
    monkeypatch.setattr(llm.session, "current", fake_result)
    seen = {}

    def complete(session, body):
        seen["body"] = body
        return {"choices": [{"message": {"content": "pong"}}], "usage": {"prompt_tokens": 3, "completion_tokens": 1}}

    monkeypatch.setattr(client, "complete", complete)
    trace = FakeTrace()
    out = llm.process({"messages": [{"role": "user", "content": "ping"}]}, {"temperature": 0.1}, trace)
    assert out == {"text": "pong", "promptTokens": 3, "completionTokens": 1}
    assert seen["body"]["model"] == "m" and seen["body"]["temperature"] == 0.1 and seen["body"]["stream"] is False
    span = trace.spans[0][1]
    assert span.attributes["attestation_pinned"] is True and span.attributes["attestation_tcb_status"] == "UpToDate"
    assert "ping" not in json.dumps(span.attributes, default=str)  # prompts never go into traces


def test_llm_stream_yields_text_chunks_only(monkeypatch):
    module, llm = make_llm()
    monkeypatch.setattr(llm.session, "current", fake_result)

    def stream(session, body):
        assert body["stream"] is True
        yield {"choices": [{"delta": {"role": "assistant"}}]}
        yield {"choices": [{"delta": {"reasoning": "thinking"}}]}
        yield {"choices": [{"delta": {"content": "Hel"}}]}
        yield {"choices": [{"delta": {"content": "lo"}}]}

    monkeypatch.setattr(client, "stream", stream)

    async def collect():
        return [c async for c in llm.aprocess_stream({"messages": [{"role": "user", "content": "x"}]}, {}, FakeTrace())]

    assert asyncio.run(collect()) == [{"chunk": {"text": "Hel"}}, {"chunk": {"text": "lo"}}, {"footer": {"finishReason": "stop"}}]


def test_llm_stream_forwards_tools_and_tool_call_deltas(monkeypatch):
    module, llm = make_llm()
    monkeypatch.setattr(llm.session, "current", fake_result)
    seen = {}

    def stream(session, body):
        seen["body"] = body
        yield {"choices": [{"delta": {"tool_calls": [
            {"index": 0, "id": "c1", "type": "function", "function": {"name": "get_flow", "arguments": ""}}]}}]}
        yield {"choices": [{"delta": {"tool_calls": [{"index": 0, "function": {"arguments": "{\"k\": 1}"}}]}}]}
        yield {"choices": [{"delta": {}, "finish_reason": "tool_calls"}]}

    monkeypatch.setattr(client, "stream", stream)
    settings = {"tools": [{"type": "function", "function": {"name": "get_flow", "parameters": None}}],
                "toolChoice": {"type": "auto"}}

    async def collect():
        return [c async for c in llm.aprocess_stream({"messages": [{"role": "user", "content": "x"}]}, settings, FakeTrace())]

    out = asyncio.run(collect())
    assert seen["body"]["tools"][0]["function"]["name"] == "get_flow" and seen["body"]["tool_choice"] == "auto"
    assert out == [
        {"chunk": {"toolCalls": [{"type": "function", "id": "c1", "index": 0, "function": {"name": "get_flow", "arguments": ""}}]}},
        {"chunk": {"toolCalls": [{"type": "function", "index": 0, "function": {"arguments": "{\"k\": 1}"}}]}},
        {"footer": {"finishReason": "tool_calls"}},
    ]


def test_llm_stream_closes_the_underlying_stream_on_early_exit(monkeypatch):
    module, llm = make_llm()
    monkeypatch.setattr(llm.session, "current", fake_result)
    state = {"closed": False}

    def stream(session, body):
        try:
            for i in range(100):
                yield {"choices": [{"delta": {"content": "x"}}]}
        finally:
            state["closed"] = True

    monkeypatch.setattr(client, "stream", stream)

    async def first_only():
        agen = llm.aprocess_stream({"messages": [{"role": "user", "content": "x"}]}, {}, FakeTrace())
        await agen.__anext__()
        await agen.aclose()

    asyncio.run(first_only())
    assert state["closed"]


# ---- runnable.py (macro) ----

def make_macro(**cfg):
    module = load("python-runnables/attest-endpoint/runnable.py", "macro_under_test")
    return module.MyRunnable("PROJ", {"endpoint": preset(**cfg)}, {})


def test_macro_reports_success_and_matching_pins(monkeypatch):
    monkeypatch.setattr(attest, "attest_endpoint", lambda cfg: AttestationResult(b"k", 1.0, False, {
        "tcb_status": "UpToDate", "key_fingerprint": "f", "gpus": [{"gpu": "GPU-0", "model": "GH100", "driver": "1", "vbios": "2"}],
        "measurements": {"mr_td": H, "rtmr0": H, "rtmr1": H, "rtmr2": H, "rtmr3": H}}))
    html = make_macro().run(None)
    assert "Attestation verified" in html and "Matches the pinned measurements" in html and H in html


def test_macro_shows_measurements_even_when_pins_do_not_match(monkeypatch):
    other = "cd" * 48
    monkeypatch.setattr(attest, "attest_endpoint", lambda cfg: AttestationResult(b"k", 1.0, False, {
        "tcb_status": "UpToDate", "key_fingerprint": "f", "gpus": [{"gpu": "GPU-0", "model": "GH100", "driver": "1", "vbios": "2"}],
        "measurements": {"mr_td": other, "rtmr0": other, "rtmr1": other, "rtmr2": other, "rtmr3": other}}))
    html = make_macro().run(None)
    assert "Does NOT match" in html and other in html


def test_macro_works_without_pins_and_reports_failures_readably(monkeypatch):
    from confidentialllm.errors import AttestationError

    def boom(cfg):
        raise AttestationError("quote did not verify <script>")

    monkeypatch.setattr(attest, "attest_endpoint", boom)
    html = make_macro(pins="").run(None)
    assert "Attestation failed" in html and "<script>" not in html  # error text is HTML-escaped
