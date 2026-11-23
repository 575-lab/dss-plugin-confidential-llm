"""Demo mode: simulated evidence from spark-demo/demo_gateway.py, checked by the production code path."""

import base64
import importlib.util
import os
import secrets
import time

import jwt
import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ec

from confidentialllm import attest, demo
from confidentialllm.config import EndpointConfig
from confidentialllm.errors import AttestationError, ConfigError

GATEWAY_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "..", "spark-demo", "demo_gateway.py")


def _gateway():
    pytest.importorskip("aiohttp")
    pytest.importorskip("pyhpke")
    spec = importlib.util.spec_from_file_location("demo_gateway", GATEWAY_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _pem(private_key):
    return private_key.public_key().public_bytes(
        serialization.Encoding.PEM, serialization.PublicFormat.SubjectPublicKeyInfo).decode()


@pytest.fixture
def setup(tmp_path):
    gw = _gateway()
    root, root_pem = gw.load_or_create_demo_root(tmp_path)
    _, key_config = gw.new_hpke_key()
    measurements = gw.simulated_measurements("fw", set())
    state = {"key_config": key_config, "tamper": set(), "demo_root": root, "tcb_status": "UpToDate",
             "measurements": measurements, "gpus": [("Test GPU", "1.0")]}
    pins = {k: measurements[k] for k in ("mr_td", "rtmr0", "rtmr1", "rtmr2")}
    config = EndpointConfig.from_preset({"gateway_url": "http://127.0.0.1:1", "demo_trust_root": root_pem,
                                         "pins": [pins]})
    return gw, state, config


def _verify(gw, state, config, nonce=None, evidence_nonce=None):
    nonce = nonce or secrets.token_bytes(32)
    evidence = gw.make_evidence(state, evidence_nonce or nonce)
    return attest.verify_evidence(config, nonce, state["key_config"], evidence)


def test_demo_evidence_passes(setup):
    result = _verify(*setup)
    assert result.pinned and result.info["demo_mode"] is True
    assert result.info["gpus"][0]["model"].startswith("Test GPU")


def test_binding_uses_production_digest(setup):
    gw, _, _ = setup
    from confidentialllm.binding import compute_digest
    nonce, key = secrets.token_bytes(32), b"k" * 41
    assert gw.binding_digest(nonce, key) == compute_digest(nonce, key)


@pytest.mark.parametrize("tamper, message", [
    ("measurements", "does not match any pinned measurement"),
    ("binding", "does not commit to this request's nonce"),
    ("gpu", "dbgstat"),
])
def test_tampering_is_refused(setup, tamper, message):
    gw, state, config = setup
    state["tamper"] = {tamper}
    if tamper == "measurements":
        state["measurements"] = gw.simulated_measurements("fw", {tamper})
    with pytest.raises(AttestationError, match=message):
        _verify(gw, state, config)


def test_replayed_evidence_is_refused(setup):
    with pytest.raises(AttestationError, match="nonce"):
        _verify(*setup, evidence_nonce=secrets.token_bytes(32))


def test_tcb_status_enforced(setup):
    gw, state, config = setup
    state["tcb_status"] = "OutOfDate"
    with pytest.raises(AttestationError, match="OutOfDate"):
        _verify(gw, state, config)


def test_other_signing_key_is_refused(setup):
    gw, state, config = setup
    state["demo_root"] = ec.generate_private_key(ec.SECP384R1())
    with pytest.raises(AttestationError, match="demo trust root"):
        _verify(gw, state, config)


def test_demo_quote_cannot_pass_as_gpu_token(setup):
    """Issuers differ, so a quote JWT can't be replayed as a GPU token (or vice versa)."""
    gw, state, config = setup
    nonce = secrets.token_bytes(32)
    evidence = gw.make_evidence(state, nonce)
    quote_jwt = base64.b64decode(evidence["tdx_quote_b64"]).decode()
    evidence["gpu"]["nras_overall_jwt"] = quote_jwt
    with pytest.raises(AttestationError, match="GPU token"):
        attest.verify_evidence(config, nonce, state["key_config"], evidence)


def test_expired_quote_is_refused(setup):
    _, state, config = setup
    root = state["demo_root"]
    old = int(time.time()) - 1000
    quote = jwt.encode({"iss": demo.DEMO_QUOTE_ISSUER, "iat": old, "exp": old + 10}, root, algorithm="ES384")
    with pytest.raises(AttestationError, match="demo trust root"):
        demo.verify_quote(quote.encode(), {"UpToDate"}, config.demo_trust_root)


def test_trust_root_validation():
    with pytest.raises(ConfigError):
        EndpointConfig.from_preset({"gateway_url": "http://127.0.0.1:1", "allow_unpinned": True,
                                    "demo_trust_root": "not a key"})
    p256 = _pem(ec.generate_private_key(ec.SECP256R1()))
    with pytest.raises(ConfigError, match="P-384"):
        EndpointConfig.from_preset({"gateway_url": "http://127.0.0.1:1", "allow_unpinned": True,
                                    "demo_trust_root": p256})


def test_production_mode_is_default():
    config = EndpointConfig.from_preset({"gateway_url": "http://127.0.0.1:1", "allow_unpinned": True})
    assert config.demo_mode is False


def test_demo_root_changes_session_cache_key(setup):
    _, _, config = setup
    import dataclasses
    assert config.cache_key != dataclasses.replace(config, demo_trust_root=None).cache_key


def test_ehbp_roundtrip_with_real_client(setup):
    """The gateway's server-side EHBP must interoperate with tinfoil-ehbp (the client the plugin uses)."""
    gw, state, _ = setup
    ehbp = pytest.importorskip("ehbp")
    private, key_config = gw.new_hpke_key()
    identity = ehbp.ServerIdentity.unmarshal_public_config(key_config)
    plaintext = os.urandom(200_000)  # several 64 KiB request frames
    encrypted = identity.encrypt_request_body(plaintext)

    suite = gw._suite()
    recipient = suite.create_recipient_context(encrypted.encapsulated_key, suite.kem.deserialize_private_key(private),
                                               info=gw.HPKE_REQUEST_INFO)
    reader, opened = gw.FrameReader(), b""
    for frame in reader.push(encrypted.body):
        opened += recipient.open(frame, b"")
    assert opened == plaintext

    sealer = gw.ResponseSealer(recipient.export(gw.EXPORT_LABEL, 32), encrypted.encapsulated_key)
    body = sealer.seal(b"hello ") + sealer.seal(b"world")
    assert encrypted.token.decrypt_response_body(sealer.response_nonce, body) == b"hello world"
