import base64
import threading
import time
from types import SimpleNamespace

import pytest

from confidentialllm import attest, binding, gpu, tdx
from confidentialllm.attest import AttestationResult
from confidentialllm.config import EndpointConfig
from confidentialllm.errors import AttestationError
from confidentialllm.session import AttestedSession

H = "ab" * 48


def make_config(**overrides):
    preset = {"gateway_url": "https://gw", "pins": '{"mr_td": "%s"}' % H, "attestation_ttl_minutes": 1}
    preset.update(overrides)
    return EndpointConfig.from_preset(preset)


class Clock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now


def fake_attest(counter, key=b"k1"):
    def fn(config):
        counter.append(1)
        return AttestationResult(key_config=key, attested_at=time.time(), pinned=True, info={"key_fingerprint": "f"})
    return fn


def test_session_caches_until_ttl_then_reattests():
    calls, clock = [], Clock()
    s = AttestedSession(make_config(), attest_fn=fake_attest(calls), clock=clock)
    s.current(); s.current()
    assert len(calls) == 1
    clock.now += 59
    s.current()
    assert len(calls) == 1
    clock.now += 2  # past the 60s ttl
    s.current()
    assert len(calls) == 2


def test_invalidate_forces_reattestation():
    calls = []
    s = AttestedSession(make_config(), attest_fn=fake_attest(calls), clock=Clock())
    s.current(); s.invalidate(); s.current()
    assert len(calls) == 2


def test_late_failure_of_old_key_does_not_discard_newer_attestation():
    keys = iter([b"old", b"new"])
    s = AttestedSession(make_config(), attest_fn=lambda c: AttestationResult(next(keys), time.time(), True), clock=Clock())
    old = s.current()
    s.invalidate(old.key_config)      # first caller fails with the old key and re-attests
    new = s.current()
    s.invalidate(old.key_config)      # a slow second caller reports the same old key late
    assert s.current() is new         # the good, newer attestation survives


def test_failed_attestation_is_not_cached():
    attempts = []

    def flaky(config):
        attempts.append(1)
        if len(attempts) == 1:
            raise AttestationError("boom")
        return AttestationResult(b"k", time.time(), True)

    s = AttestedSession(make_config(), attest_fn=flaky, clock=Clock())
    with pytest.raises(AttestationError):
        s.current()
    assert s.current().key_config == b"k"
    assert len(attempts) == 2


def test_concurrent_callers_share_one_attestation():
    calls, gate = [], threading.Event()

    def slow(config):
        calls.append(1)
        gate.wait(2)
        return AttestationResult(b"k", time.time(), True)

    s = AttestedSession(make_config(), attest_fn=slow, clock=Clock())
    threads = [threading.Thread(target=s.current) for _ in range(8)]
    for t in threads:
        t.start()
    time.sleep(0.2)
    gate.set()
    for t in threads:
        t.join()
    assert len(calls) == 1


# ---- verify_evidence: the whole flow with Intel/NVIDIA verification stubbed out ----

NONCE = b"\x07" * 32
KEY_CONFIG = b"hpke-key-config"


def fake_report(report_data, mr_td=bytes.fromhex(H)):
    zero = bytes(48)
    return SimpleNamespace(report_data=report_data, mr_td=mr_td, rt_mr0=zero, rt_mr1=zero, rt_mr2=zero, rt_mr3=zero)


@pytest.fixture
def stub(monkeypatch):
    def install(report_data=None, status="UpToDate", mr_td=bytes.fromhex(H), gpu_summary=None, gpu_error=None):
        digest = binding.compute_digest(NONCE, KEY_CONFIG)
        rd = digest + bytes(32) if report_data is None else report_data

        async def verify_quote(quote, allowed):
            if status not in allowed:
                raise AttestationError("tcb")
            return SimpleNamespace(report=fake_report(rd, mr_td)), status

        def verify_tokens(evidence, digest_hex, resolver=None):
            if gpu_error:
                raise gpu_error
            assert digest_hex == digest.hex()
            return gpu_summary or [{"gpu": "GPU-0"}]

        monkeypatch.setattr(tdx, "verify_quote", verify_quote)
        monkeypatch.setattr(gpu, "verify_tokens", verify_tokens)
    return install


EVIDENCE = {"tdx_quote_b64": base64.b64encode(b"quote").decode(), "gpu": {"x": 1}}


def test_full_verification_succeeds_and_reports_identity(stub):
    stub()
    result = attest.verify_evidence(make_config(), NONCE, KEY_CONFIG, EVIDENCE)
    assert result.key_config == KEY_CONFIG and result.pinned is True
    assert result.info["tcb_status"] == "UpToDate" and result.info["measurements"]["mr_td"] == H


def test_quote_bound_to_a_different_key_is_rejected(stub):
    stub()
    with pytest.raises(AttestationError, match="commit"):
        attest.verify_evidence(make_config(), NONCE, b"attacker-key", EVIDENCE)  # evidence for another key


def test_replayed_quote_for_another_nonce_is_rejected(stub):
    stub()
    with pytest.raises(AttestationError, match="commit"):
        attest.verify_evidence(make_config(), b"\x08" * 32, KEY_CONFIG, EVIDENCE)


def test_pin_mismatch_is_rejected(stub):
    stub(mr_td=bytes.fromhex("cd" * 48))
    with pytest.raises(AttestationError, match="pinned"):
        attest.verify_evidence(make_config(), NONCE, KEY_CONFIG, EVIDENCE)


def test_disallowed_tcb_is_rejected(stub):
    stub(status="OutOfDate")
    with pytest.raises(AttestationError, match="tcb"):
        attest.verify_evidence(make_config(), NONCE, KEY_CONFIG, EVIDENCE)


def test_gpu_failure_blocks_even_when_cpu_is_fine(stub):
    stub(gpu_error=AttestationError("gpu bad"))
    with pytest.raises(AttestationError, match="gpu bad"):
        attest.verify_evidence(make_config(), NONCE, KEY_CONFIG, EVIDENCE)


@pytest.mark.parametrize("evidence", [{}, {"tdx_quote_b64": "!!!not base64"}, {"tdx_quote_b64": None}])
def test_missing_or_garbled_quote_is_rejected(stub, evidence):
    stub()
    with pytest.raises(AttestationError, match="no valid TDX quote"):
        attest.verify_evidence(make_config(), NONCE, KEY_CONFIG, evidence)


def test_unpinned_mode_marks_result_unpinned(stub):
    stub()
    cfg = EndpointConfig.from_preset({"gateway_url": "https://gw", "allow_unpinned": True})
    assert attest.verify_evidence(cfg, NONCE, KEY_CONFIG, EVIDENCE).pinned is False
