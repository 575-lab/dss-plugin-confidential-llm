import base64
import datetime
import hashlib
import json

import pytest
from cryptography.hazmat.primitives.asymmetric import ec

import vtpm_fixtures as fx
from confidentialllm import attest, binding, gpu, vtpm
from confidentialllm.config import EndpointConfig, parse_pins
from confidentialllm.errors import AttestationError, ConfigError

NONCE = b"\x07" * 32
KEY_CONFIG = b"hpke-key-config"
DIGEST = binding.compute_digest(NONCE, KEY_CONFIG)


def verify(evidence, ca, **kw):
    kw.setdefault("now", fx.NOW)
    return vtpm.verify_evidence(evidence, kw.pop("digest", DIGEST), roots=[ca.root], intermediates=kw.pop("intermediates", []), **kw)


# ---- the happy path ----

def test_valid_evidence_verifies_and_reports_identity():
    evidence, ca = fx.make_evidence(DIGEST)
    measurements, platform = verify(evidence, ca)
    expected = fx.replay_all(fx.default_events())
    assert measurements["pcr0"] == expected[0].hex() and measurements["pcr4"] == expected[4].hex()
    assert set(measurements) == {"pcr0", "pcr4", "pcr7", "pcr8", "pcr9", "instance_id"}
    assert measurements["instance_id"] == "987654321012345678"
    assert platform["technology"] == "AMD_SEV" and platform["secure_boot"] is True
    assert platform["instance"]["zone"] == "us-central1-b" and platform["instance"]["instance_name"] == "g4-test"


@pytest.mark.parametrize("tech,name", [(1, "AMD_SEV"), (2, "AMD_SEV_ES"), (3, "INTEL_TDX"), (4, "AMD_SEV_SNP")])
def test_every_confidential_technology_is_reported(tech, name):
    evidence, ca = fx.make_evidence(DIGEST, technology=tech)
    assert verify(evidence, ca)[1]["technology"] == name


def test_intermediate_may_come_from_the_bundle_instead_of_the_gateway():
    evidence, ca = fx.make_evidence(DIGEST, send_intermediate=False)
    with pytest.raises(AttestationError, match="chain"):
        verify(evidence, ca)
    assert verify(evidence, ca, intermediates=[ca.intermediate])[1]["technology"] == "AMD_SEV"


def test_non_production_or_missing_instance_info_is_not_reported():
    evidence, ca = fx.make_evidence(DIGEST, instance=fx.instance_info_der(is_production=False))
    measurements, platform = verify(evidence, ca)
    assert "instance_id" not in measurements and platform["instance"] is None
    evidence, ca = fx.make_evidence(DIGEST, instance=None)
    assert verify(evidence, ca)[1]["instance"] is None


# ---- binding: nonce and gateway key ----

def test_quote_for_another_key_or_nonce_is_rejected():
    evidence, ca = fx.make_evidence(DIGEST)
    other = binding.compute_digest(b"\x08" * 32, KEY_CONFIG)
    with pytest.raises(AttestationError, match="commit"):
        verify(evidence, ca, digest=other)
    with pytest.raises(AttestationError, match="commit"):
        verify(evidence, ca, digest=binding.compute_digest(NONCE, b"attacker-key"))


# ---- trust chain and signature ----

def test_ak_from_another_ca_is_rejected():
    evidence, _ = fx.make_evidence(DIGEST)
    with pytest.raises(AttestationError, match="chain"):
        verify(evidence, fx.CA())


def test_expired_ak_certificate_is_rejected():
    evidence, ca = fx.make_evidence(DIGEST)
    with pytest.raises(AttestationError, match="validity"):
        verify(evidence, ca, now=fx.NOW + datetime.timedelta(days=4000))


def test_quote_signed_by_a_different_key_is_rejected():
    evidence, ca = fx.make_evidence(DIGEST)
    other = fx.sign_quote(ec.generate_private_key(ec.SECP256R1()), base64.b64decode(evidence["quote_b64"]))
    evidence["signature_b64"] = fx.b64(other)
    with pytest.raises(AttestationError, match="signature"):
        verify(evidence, ca)


def test_modified_quote_is_rejected():
    evidence, ca = fx.make_evidence(DIGEST)
    quote = bytearray(base64.b64decode(evidence["quote_b64"]))
    quote[-1] ^= 1  # flip a bit of the PCR digest
    evidence["quote_b64"] = fx.b64(bytes(quote))
    with pytest.raises(AttestationError, match="signature"):
        verify(evidence, ca)


# ---- PCR values and event log ----

def test_pcr_values_that_do_not_match_the_quote_are_rejected():
    evidence, ca = fx.make_evidence(DIGEST, pcr_tweak={4: b"\x11" * 32})
    with pytest.raises(AttestationError, match="PCR digest"):
        verify(evidence, ca)


def test_missing_pcr_value_is_rejected():
    evidence, ca = fx.make_evidence(DIGEST)
    del evidence["pcrs"]["9"]
    with pytest.raises(AttestationError, match="PCR 9"):
        verify(evidence, ca)


def test_event_log_that_does_not_replay_to_the_quote_is_rejected():
    evidence, ca = fx.make_evidence(DIGEST)
    # A log claiming SEV-SNP, but the quoted PCR0 came from the real (SEV) boot.
    evidence["event_log_b64"] = fx.b64(fx.build_event_log(fx.default_events(technology=4)))
    with pytest.raises(AttestationError, match="replay to the quoted PCR 0"):
        verify(evidence, ca)


def test_non_confidential_vm_is_rejected():
    evidence, ca = fx.make_evidence(DIGEST, technology=0)
    with pytest.raises(AttestationError, match="not a Confidential VM"):
        verify(evidence, ca)


def test_missing_nonhost_event_is_rejected():
    events = [e for e in fx.default_events() if e[1] != vtpm.EV_NONHOST_INFO]
    evidence, ca = fx.make_evidence(DIGEST, events=events)
    with pytest.raises(AttestationError, match="NonHostInfo"):
        verify(evidence, ca)


def test_nonhost_event_after_the_separator_is_ignored():
    events = [e for e in fx.default_events() if e[1] != vtpm.EV_NONHOST_INFO]
    events.append((0, vtpm.EV_NONHOST_INFO, vtpm.GCE_NONHOST_SIGNATURE + b"\x04" + bytes(15), None))
    evidence, ca = fx.make_evidence(DIGEST, events=events)
    with pytest.raises(AttestationError, match="NonHostInfo"):
        verify(evidence, ca)


def test_nonhost_event_whose_data_does_not_match_its_digest_is_rejected():
    events = fx.default_events()
    events[2] = (0, vtpm.EV_NONHOST_INFO, vtpm.GCE_NONHOST_SIGNATURE + b"\x04" + bytes(15),
                 hashlib.sha256(vtpm.GCE_NONHOST_SIGNATURE + b"\x00" + bytes(15)).digest())
    evidence, ca = fx.make_evidence(DIGEST, events=events)
    with pytest.raises(AttestationError, match="digest"):
        verify(evidence, ca)


def test_secure_boot_off_is_rejected():
    evidence, ca = fx.make_evidence(DIGEST, secure_boot=False)
    with pytest.raises(AttestationError, match="Secure Boot"):
        verify(evidence, ca)


def test_startup_locality_sets_the_initial_pcr0():
    events = [vtpm.Event(0, vtpm.EV_NO_ACTION, None, vtpm.STARTUP_LOCALITY_SIGNATURE + b"\x03"),
              vtpm.Event(0, 1, b"\x22" * 32, b"")]
    assert vtpm.replay(events, (0,))[0] == hashlib.sha256(bytes(31) + b"\x03" + b"\x22" * 32).digest()


def test_quote_without_sha256_bank_is_rejected():
    pcrs = fx.replay_all(fx.default_events())
    quote = vtpm.parse_quote(fx.build_quote(DIGEST, pcrs, alg=vtpm.TPM_ALG_SHA1))
    with pytest.raises(AttestationError, match="SHA-256"):
        vtpm.check_pcr_digest(quote, pcrs, {0})


@pytest.mark.parametrize("field", ["ak_cert_der_b64", "quote_b64", "signature_b64", "event_log_b64"])
def test_missing_or_garbled_fields_are_rejected(field):
    evidence, ca = fx.make_evidence(DIGEST)
    evidence[field] = "!!!"
    with pytest.raises(AttestationError):
        verify(evidence, ca)


@pytest.mark.parametrize("raw", [b"", b"\x00" * 10, b"\xff\x54\x43\x47\x80\x17" + bytes(40)])
def test_garbage_quotes_raise_attestation_errors(raw):
    with pytest.raises(AttestationError):
        vtpm.parse_quote(raw)


def test_bundled_google_roots_load():
    roots, intermediates = vtpm.default_roots()
    assert any("EK/AK CA Root" in r.subject.rfc4514_string() for r in roots)
    assert len(intermediates) == 3


# ---- pins and configuration ----

P = "ab" * 32


def vtpm_preset(**overrides):
    preset = {"gateway_url": "https://gw", "tee_type": "gce-vtpm",
              "pins": json.dumps({"pcr0": P, "pcr4": P, "pcr7": P})}
    preset.update(overrides)
    return preset


def test_vtpm_pins_parse_and_normalize():
    pins = parse_pins(json.dumps({"pcr0": P.upper(), "pcr4": P, "pcr7": P, "instance_id": 42}), "gce-vtpm")
    assert pins == ({"pcr0": P, "pcr4": P, "pcr7": P, "instance_id": "42"},)


@pytest.mark.parametrize("pin", [
    {"pcr0": P, "pcr4": P},                                   # pcr7 missing
    {"pcr0": P, "pcr4": P, "pcr7": "ab" * 48},                # TDX-sized value
    {"pcr0": P, "pcr4": P, "pcr7": P, "mr_td": "ab" * 48},    # TDX key mixed in
    {"pcr0": P, "pcr4": P, "pcr7": P, "instance_id": "g4-test"},
    {"pcr0": P, "pcr4": P, "pcr7": P, "instance_id": True},
])
def test_bad_vtpm_pins_rejected(pin):
    with pytest.raises(ConfigError):
        parse_pins(json.dumps(pin), "gce-vtpm")


def test_tdx_pins_are_rejected_for_a_vtpm_preset_and_vice_versa():
    with pytest.raises(ConfigError):
        EndpointConfig.from_preset(vtpm_preset(pins=json.dumps({"mr_td": "ab" * 48})))
    with pytest.raises(ConfigError):
        EndpointConfig.from_preset(vtpm_preset(tee_type="tdx"))


def test_unknown_tee_type_rejected_and_default_is_tdx():
    with pytest.raises(ConfigError, match="TEE type"):
        EndpointConfig.from_preset(vtpm_preset(tee_type="sev-snp"))
    cfg = EndpointConfig.from_preset({"gateway_url": "https://gw", "pins": json.dumps({"mr_td": "ab" * 48})})
    assert cfg.tee_type == "tdx"


def test_tee_type_and_gpu_models_are_part_of_the_cache_key():
    a = EndpointConfig.from_preset(vtpm_preset())
    b = EndpointConfig.from_preset(vtpm_preset(allowed_gpu_models=["GB20X"]))
    assert a.cache_key != b.cache_key and b.allowed_gpu_models == ("GB20X",)
    hash(b.cache_key)


# ---- the whole flow through attest.verify_evidence ----

@pytest.fixture
def gce(monkeypatch):
    """Real vTPM verification against a fake CA; NVIDIA stubbed out."""
    evidence, ca = fx.make_evidence(DIGEST)
    real = vtpm.verify_evidence
    monkeypatch.setattr(vtpm, "verify_evidence",
                        lambda ev, digest, pinned_pcrs=(): real(ev, digest, pinned_pcrs, roots=[ca.root], intermediates=[], now=fx.NOW))
    monkeypatch.setattr(gpu, "verify_tokens", lambda ev, d, r=None, allowed_models=(): [{"gpu": "GPU-0", "model": "GB20X"}])
    measurements, _ = real(evidence, DIGEST, roots=[ca.root], now=fx.NOW)
    return {"tee": "gce-vtpm", "vtpm": evidence, "gpu": {"x": 1}}, measurements


def test_pinned_vtpm_endpoint_verifies(gce):
    evidence, m = gce
    cfg = EndpointConfig.from_preset(vtpm_preset(pins=json.dumps({k: m[k] for k in ("pcr0", "pcr4", "pcr7", "instance_id")})))
    result = attest.verify_evidence(cfg, NONCE, KEY_CONFIG, evidence)
    assert result.pinned is True and result.info["tee"] == "gce-vtpm"
    assert result.info["platform"]["technology"] == "AMD_SEV" and "vTPM" in result.info["tcb_status"]


def test_vtpm_pin_mismatch_names_the_pcr(gce):
    evidence, m = gce
    cfg = EndpointConfig.from_preset(vtpm_preset(pins=json.dumps({"pcr0": m["pcr0"], "pcr4": P, "pcr7": m["pcr7"]})))
    with pytest.raises(AttestationError, match="vTPM identity.*pcr4"):
        attest.verify_evidence(cfg, NONCE, KEY_CONFIG, evidence)


def test_instance_id_pin_binds_to_one_vm(gce):
    evidence, m = gce
    pin = {k: m[k] for k in ("pcr0", "pcr4", "pcr7")}
    pin["instance_id"] = "1"
    with pytest.raises(AttestationError, match="instance_id"):
        attest.verify_evidence(EndpointConfig.from_preset(vtpm_preset(pins=json.dumps(pin))), NONCE, KEY_CONFIG, evidence)


def test_evidence_of_another_tee_type_is_rejected(gce):
    evidence, _ = gce
    tdx_cfg = EndpointConfig.from_preset({"gateway_url": "https://gw", "allow_unpinned": True})
    with pytest.raises(AttestationError, match="configured for Intel TDX"):
        attest.verify_evidence(tdx_cfg, NONCE, KEY_CONFIG, evidence)
    vtpm_cfg = EndpointConfig.from_preset(vtpm_preset(pins="", allow_unpinned=True))
    with pytest.raises(AttestationError, match="Intel TDX evidence"):
        attest.verify_evidence(vtpm_cfg, NONCE, KEY_CONFIG, {"tdx_quote_b64": "eA==", "gpu": {}})  # no "tee": an old TDX gateway


def test_gpu_model_allow_list():
    gpu.check_gpu_model("GPU-0", "GB20X A01 GSP BROM", ("GB20X",))
    gpu.check_gpu_model("GPU-0", "anything", ())
    with pytest.raises(AttestationError, match="not one of the allowed"):
        gpu.check_gpu_model("GPU-0", "GH100 A01 GSP BROM", ("GB20X",))
