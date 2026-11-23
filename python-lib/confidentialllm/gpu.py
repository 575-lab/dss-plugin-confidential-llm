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

"""NVIDIA GPU attestation: verify the NRAS-signed EAT tokens produced by ``nvattest --verifier remote``."""

import hmac
import threading
import time

from .errors import AttestationError

JWKS_URL = "https://nras.attestation.nvidia.com/.well-known/jwks.json"
NRAS_ISSUER = "https://nras.attestation.nvidia.com"

# Claims that must hold for every GPU token. Names are those emitted by nvattest (claims v4).
REQUIRED_DEVICE_CLAIMS = {
    "secboot": True,
    "dbgstat": "disabled",
    "measres": "success",
    "x-nvidia-gpu-arch-check": True,
    "x-nvidia-gpu-attestation-report-nonce-match": True,
    "x-nvidia-gpu-attestation-report-signature-verified": True,
    "x-nvidia-gpu-attestation-report-cert-chain-fwid-match": True,
    "x-nvidia-gpu-driver-rim-signature-verified": True,
    "x-nvidia-gpu-vbios-rim-signature-verified": True,
    "x-nvidia-gpu-driver-rim-version-match": True,
    "x-nvidia-gpu-vbios-rim-version-match": True,
}
CERT_CHAINS = (
    "x-nvidia-gpu-attestation-report-cert-chain",
    "x-nvidia-gpu-driver-rim-cert-chain",
    "x-nvidia-gpu-vbios-rim-cert-chain",
)

JWKS_TTL_SECONDS = 3600

_jwks_lock = threading.Lock()
_jwks = {"keys": None, "fetched_at": 0.0}


def _fetch_jwks():
    """Download NVIDIA's signing keys with httpx (bundled CA store), not PyJWT's urllib client.

    PyJWT's own fetcher uses the system certificate store, which is missing or stale on some DSS hosts.
    """
    import httpx
    import jwt

    response = httpx.get(JWKS_URL, timeout=15, follow_redirects=False)
    response.raise_for_status()
    return jwt.PyJWKSet.from_dict(response.json())


def _find_key(keyset, kid):
    for jwk in keyset.keys:
        if jwk.key_id == kid:
            return jwk.key
    return None


def _default_key_resolver(token):
    """Return NVIDIA's public key for ``token`` (by ``kid``), cached for an hour, refreshed once on an unknown kid."""
    import jwt

    kid = jwt.get_unverified_header(token).get("kid")
    if not kid:
        raise AttestationError("A GPU token has no key id.")

    with _jwks_lock:
        fresh = _jwks["keys"] is not None and time.monotonic() - _jwks["fetched_at"] < JWKS_TTL_SECONDS
        if fresh:
            key = _find_key(_jwks["keys"], kid)
            if key is not None:
                return key
        # Stale cache, or NVIDIA rotated keys: fetch once. Never trust a key we could not find by kid.
        _jwks["keys"], _jwks["fetched_at"] = _fetch_jwks(), time.monotonic()
        key = _find_key(_jwks["keys"], kid)
    if key is None:
        raise AttestationError("No NVIDIA signing key matches the GPU token's key id.")
    return key


def _decode(token, key_resolver, issuer=NRAS_ISSUER):
    import jwt

    try:
        key = key_resolver(token)
        return jwt.decode(
            token, key, algorithms=["ES384"], issuer=issuer,
            options={"require": ["exp", "iat", "iss"]},
        )
    except Exception as exc:
        raise AttestationError("A GPU token is not a valid NVIDIA-signed token: {}".format(exc)) from exc


def check_device_claims(name, claims, digest_hex):
    if not hmac.compare_digest(str(claims.get("eat_nonce")), digest_hex):
        raise AttestationError("{}: the GPU token's nonce does not match this request.".format(name))
    for claim, expected in REQUIRED_DEVICE_CLAIMS.items():
        if claims.get(claim) != expected:
            raise AttestationError("{}: GPU claim {}={!r}, expected {!r}.".format(name, claim, claims.get(claim), expected))
    for chain in CERT_CHAINS:
        status = claims.get(chain) or {}
        if status.get("x-nvidia-cert-status") != "valid" or status.get("x-nvidia-cert-ocsp-status") != "good":
            raise AttestationError("{}: certificate chain {} is not valid and OCSP-good.".format(name, chain))


def verify_tokens(gpu, digest_hex, key_resolver=None, issuer=NRAS_ISSUER):
    """Verify the NVIDIA tokens and return a summary per GPU.

    ``gpu`` is {"nras_overall_jwt": str, "nras_device_jwts": {"GPU-0": str, ...}} as sent by the gateway.
    """
    key_resolver = key_resolver or _default_key_resolver
    if not isinstance(gpu, dict) or "nras_overall_jwt" not in gpu or not gpu.get("nras_device_jwts"):
        raise AttestationError("The gateway returned no NVIDIA GPU attestation.")

    overall = _decode(gpu["nras_overall_jwt"], key_resolver, issuer)
    if overall.get("x-nvidia-overall-att-result") is not True:
        raise AttestationError("NVIDIA's overall attestation result is not true.")
    if not hmac.compare_digest(str(overall.get("eat_nonce")), digest_hex):
        raise AttestationError("The overall GPU token's nonce does not match this request.")

    summary = []
    for name, token in gpu["nras_device_jwts"].items():
        claims = _decode(token, key_resolver, issuer)
        check_device_claims(name, claims, digest_hex)
        summary.append({
            "gpu": name,
            "model": claims.get("hwmodel"),
            "driver": claims.get("x-nvidia-gpu-driver-version"),
            "vbios": claims.get("x-nvidia-gpu-vbios-version"),
        })
    return summary
