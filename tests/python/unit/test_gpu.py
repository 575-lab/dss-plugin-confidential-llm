import time

import jwt
import pytest
from cryptography.hazmat.primitives.asymmetric import ec

from confidentialllm import gpu
from confidentialllm.errors import AttestationError

NONCE = "ab" * 32
OK_CHAIN = {"x-nvidia-cert-status": "valid", "x-nvidia-cert-ocsp-status": "good"}


def good_device_claims(**overrides):
    claims = dict(gpu.REQUIRED_DEVICE_CLAIMS)
    claims.update({
        "iss": gpu.NRAS_ISSUER, "iat": int(time.time()), "exp": int(time.time()) + 3600,
        "eat_nonce": NONCE, "hwmodel": "GH100",
        "x-nvidia-gpu-driver-version": "580.178.04", "x-nvidia-gpu-vbios-version": "96.00.D9.00.01",
        **{chain: dict(OK_CHAIN) for chain in gpu.CERT_CHAINS},
    })
    claims.update(overrides)
    return claims


def good_overall(**overrides):
    claims = {"iss": gpu.NRAS_ISSUER, "iat": int(time.time()), "exp": int(time.time()) + 3600,
              "eat_nonce": NONCE, "x-nvidia-overall-att-result": True}
    claims.update(overrides)
    return claims


@pytest.fixture(scope="module")
def key():
    return ec.generate_private_key(ec.SECP384R1())


def sign(claims, key):
    return jwt.encode(claims, key, algorithm="ES384")


def resolver_for(key):
    return lambda token: key.public_key()


def bundle(key, overall=None, device=None, name="GPU-0"):
    return {"nras_overall_jwt": sign(overall or good_overall(), key),
            "nras_device_jwts": {name: sign(device or good_device_claims(), key)}}


def test_valid_tokens_pass_and_summarize(key):
    summary = gpu.verify_tokens(bundle(key), NONCE, resolver_for(key))
    assert summary == [{"gpu": "GPU-0", "model": "GH100", "driver": "580.178.04", "vbios": "96.00.D9.00.01"}]


def test_token_signed_by_other_key_is_rejected(key):
    other = ec.generate_private_key(ec.SECP384R1())
    with pytest.raises(AttestationError, match="not a valid NVIDIA-signed"):
        gpu.verify_tokens(bundle(other), NONCE, resolver_for(key))


def test_wrong_issuer_rejected(key):
    with pytest.raises(AttestationError):
        gpu.verify_tokens(bundle(key, device=good_device_claims(iss="https://evil.example")), NONCE, resolver_for(key))


def test_expired_token_rejected(key):
    old = good_device_claims(iat=int(time.time()) - 7200, exp=int(time.time()) - 3600)
    with pytest.raises(AttestationError):
        gpu.verify_tokens(bundle(key, device=old), NONCE, resolver_for(key))


def test_unsigned_alg_none_token_rejected(key):
    forged = jwt.encode(good_device_claims(), key=None, algorithm="none")
    with pytest.raises(AttestationError):
        gpu.verify_tokens({"nras_overall_jwt": sign(good_overall(), key), "nras_device_jwts": {"GPU-0": forged}},
                          NONCE, resolver_for(key))


def test_nonce_mismatch_rejected_on_overall_and_device(key):
    with pytest.raises(AttestationError, match="overall GPU token's nonce"):
        gpu.verify_tokens(bundle(key, overall=good_overall(eat_nonce="00" * 32)), NONCE, resolver_for(key))
    with pytest.raises(AttestationError, match="nonce"):
        gpu.verify_tokens(bundle(key, device=good_device_claims(eat_nonce="00" * 32)), NONCE, resolver_for(key))


def test_overall_result_must_be_true(key):
    with pytest.raises(AttestationError, match="overall attestation result"):
        gpu.verify_tokens(bundle(key, overall=good_overall(**{"x-nvidia-overall-att-result": False})), NONCE, resolver_for(key))


@pytest.mark.parametrize("claim,bad", [
    ("secboot", False), ("dbgstat", "enabled"), ("measres", "fail"), ("x-nvidia-gpu-arch-check", False),
    ("x-nvidia-gpu-attestation-report-nonce-match", False),
    ("x-nvidia-gpu-driver-rim-signature-verified", False),
    ("x-nvidia-gpu-vbios-rim-version-match", False),
    ("x-nvidia-gpu-attestation-report-cert-chain-fwid-match", False),
])
def test_each_required_claim_is_enforced(key, claim, bad):
    with pytest.raises(AttestationError, match=claim):
        gpu.verify_tokens(bundle(key, device=good_device_claims(**{claim: bad})), NONCE, resolver_for(key))


def test_missing_claim_is_rejected(key):
    claims = good_device_claims()
    del claims["secboot"]
    with pytest.raises(AttestationError, match="secboot"):
        gpu.verify_tokens(bundle(key, device=claims), NONCE, resolver_for(key))


@pytest.mark.parametrize("chain", gpu.CERT_CHAINS)
@pytest.mark.parametrize("broken", [{"x-nvidia-cert-status": "revoked", "x-nvidia-cert-ocsp-status": "good"},
                                    {"x-nvidia-cert-status": "valid", "x-nvidia-cert-ocsp-status": "revoked"}, {}])
def test_certificate_chains_must_be_valid_and_ocsp_good(key, chain, broken):
    with pytest.raises(AttestationError, match="certificate chain"):
        gpu.verify_tokens(bundle(key, device=good_device_claims(**{chain: broken})), NONCE, resolver_for(key))


@pytest.mark.parametrize("evidence", [None, {}, {"nras_overall_jwt": "x"}, {"nras_overall_jwt": "x", "nras_device_jwts": {}}])
def test_missing_gpu_evidence_rejected(evidence):
    with pytest.raises(AttestationError, match="no NVIDIA GPU attestation"):
        gpu.verify_tokens(evidence, NONCE, lambda t: None)


# ---- default key resolver: JWKS caching, rotation, unknown kid ----

from jwt.algorithms import ECAlgorithm


def _keyset(*pairs):
    """pairs of (kid, private_key) -> a PyJWKSet, like NVIDIA's JWKS document."""
    jwk_list = []
    for kid, private in pairs:
        jwk = ECAlgorithm.to_jwk(private.public_key(), as_dict=True)
        jwk.update({"kid": kid, "alg": "ES384", "use": "sig"})
        jwk_list.append(jwk)
    return jwt.PyJWKSet.from_dict({"keys": jwk_list})


def _token(private, kid):
    return jwt.encode({"x": 1}, private, algorithm="ES384", headers={"kid": kid})


@pytest.fixture
def jwks(monkeypatch):
    state = {"fetches": 0, "set": None}
    gpu._jwks.update({"keys": None, "fetched_at": 0.0})

    def fetch():
        state["fetches"] += 1
        return state["set"]

    monkeypatch.setattr(gpu, "_fetch_jwks", fetch)
    return state


def test_resolver_returns_matching_key_and_caches(jwks):
    k1 = ec.generate_private_key(ec.SECP384R1())
    jwks["set"] = _keyset(("kid-1", k1))
    token = _token(k1, "kid-1")
    key = gpu._default_key_resolver(token)
    gpu._default_key_resolver(token)
    assert key.public_numbers() == k1.public_key().public_numbers()
    assert jwks["fetches"] == 1


def test_resolver_refetches_once_when_nvidia_rotates_keys(jwks):
    k1, k2 = ec.generate_private_key(ec.SECP384R1()), ec.generate_private_key(ec.SECP384R1())
    jwks["set"] = _keyset(("kid-1", k1))
    gpu._default_key_resolver(_token(k1, "kid-1"))
    jwks["set"] = _keyset(("kid-1", k1), ("kid-2", k2))  # rotation
    key = gpu._default_key_resolver(_token(k2, "kid-2"))
    assert key.public_numbers() == k2.public_key().public_numbers()
    assert jwks["fetches"] == 2


def test_resolver_rejects_unknown_kid_after_a_single_refetch(jwks):
    k1, evil = ec.generate_private_key(ec.SECP384R1()), ec.generate_private_key(ec.SECP384R1())
    jwks["set"] = _keyset(("kid-1", k1))
    with pytest.raises(AttestationError, match="No NVIDIA signing key"):
        gpu._default_key_resolver(_token(evil, "attacker-kid"))
    assert jwks["fetches"] == 1


def test_resolver_requires_a_kid(jwks):
    k1 = ec.generate_private_key(ec.SECP384R1())
    jwks["set"] = _keyset(("kid-1", k1))
    with pytest.raises(AttestationError, match="no key id"):
        gpu._default_key_resolver(jwt.encode({"x": 1}, k1, algorithm="ES384"))


def test_resolver_refreshes_a_stale_cache(jwks, monkeypatch):
    k1 = ec.generate_private_key(ec.SECP384R1())
    jwks["set"] = _keyset(("kid-1", k1))
    token = _token(k1, "kid-1")
    gpu._default_key_resolver(token)
    gpu._jwks["fetched_at"] -= gpu.JWKS_TTL_SECONDS + 1
    gpu._default_key_resolver(token)
    assert jwks["fetches"] == 2
