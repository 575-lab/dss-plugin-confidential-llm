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

import asyncio
import base64
import concurrent.futures
import hashlib
import logging
import secrets
import time
from dataclasses import dataclass, field

import httpx

from . import gpu, tdx
from .binding import NONCE_SIZE, compute_digest
from .config import tls_verify
from .errors import AttestationError, GatewayError

LOGGER = logging.getLogger(__name__)

KEYS_PATH = "/.well-known/hpke-keys"
KEYS_MEDIA_TYPE = "application/ohttp-keys"
ATTESTATION_PATH = "/attestation"
ATTESTATION_TIMEOUT_SECONDS = 180  # includes the gateway's call to NVIDIA's attestation service


@dataclass(frozen=True)
class AttestationResult:
    """What a successful attestation proved. ``key_config`` is the only key requests may be encrypted to."""

    key_config: bytes
    attested_at: float
    pinned: bool
    info: dict = field(default_factory=dict)


def _run_async(coro):
    """Run a coroutine to completion from sync code, even when the caller already has a running event loop."""
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(asyncio.run, coro).result()


def _headers(config):
    return {"Authorization": "Bearer {}".format(config.gateway_token)} if config.gateway_token else {}


def fetch_evidence(config, nonce, http_client=None):
    """Download the HPKE key config and the attestation evidence bound to ``nonce``."""
    owns_client = http_client is None
    client = http_client or httpx.Client(timeout=httpx.Timeout(ATTESTATION_TIMEOUT_SECONDS, connect=15), verify=tls_verify(config))
    try:
        keys = client.get(config.base_url + KEYS_PATH, headers=_headers(config), follow_redirects=False)
        if keys.status_code // 100 != 2:
            raise GatewayError("Fetching the gateway's key failed with HTTP {}.".format(keys.status_code), keys.status_code)
        if keys.headers.get("content-type", "") != KEYS_MEDIA_TYPE:
            raise GatewayError("The gateway returned an unexpected content type for its key.")

        att = client.get(config.base_url + ATTESTATION_PATH, params={"nonce": nonce.hex()},
                         headers=_headers(config), follow_redirects=False)
        if att.status_code // 100 != 2:
            raise GatewayError("Fetching attestation evidence failed with HTTP {}.".format(att.status_code), att.status_code)
        return keys.content, att.json()
    except httpx.HTTPError as exc:
        raise GatewayError("Could not reach the gateway at {}: {}".format(config.base_url, exc)) from exc
    finally:
        if owns_client:
            client.close()


def verify_evidence(config, nonce, key_config, evidence, key_resolver=None):
    """Verify everything and return an AttestationResult. Raises AttestationError on any failure."""
    digest = compute_digest(nonce, key_config)

    try:
        quote = base64.b64decode(evidence["tdx_quote_b64"], validate=True)
    except (KeyError, ValueError, TypeError) as exc:
        raise AttestationError("The gateway returned no valid TDX quote.") from exc

    parsed, tcb_status = _run_async(tdx.verify_quote(quote, config.allowed_tcb))
    tdx.check_report_data(parsed.report.report_data, digest)
    measurements = tdx.read_measurements(parsed.report)
    pinned = tdx.check_pins(measurements, config.pins, config.allow_unpinned)

    gpus = gpu.verify_tokens(evidence.get("gpu"), digest.hex(), key_resolver)

    return AttestationResult(
        key_config=key_config,
        attested_at=time.time(),
        pinned=pinned,
        info={
            "tcb_status": tcb_status,
            "measurements": measurements,
            "gpus": gpus,
            "key_fingerprint": hashlib.sha256(key_config).hexdigest()[:16],
        },
    )


def attest_endpoint(config, http_client=None, key_resolver=None):
    nonce = secrets.token_bytes(NONCE_SIZE)  # fresh per attestation: evidence cannot be replayed
    key_config, evidence = fetch_evidence(config, nonce, http_client)
    return verify_evidence(config, nonce, key_config, evidence, key_resolver)
