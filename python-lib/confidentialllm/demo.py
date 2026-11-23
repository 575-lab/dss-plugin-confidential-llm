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

"""DEMO MODE: verify SIMULATED evidence from spark-demo/demo_gateway.py.

The demo gateway runs on a machine without confidential-computing hardware. Instead of an Intel-signed
TDX quote and NVIDIA-signed GPU tokens, it sends look-alike evidence signed by a demo P-384 key. This
module verifies that evidence against the demo public key an administrator pasted into the preset.

Every check after signature verification is the production code path (binding digest, pins, TCB
allow-list, GPU claims), so the demo shows exactly how the plugin behaves. What it can NOT show is
that hardware is involved: anyone holding the demo private key can produce passing evidence.
"""

from types import SimpleNamespace

from .errors import AttestationError

DEMO_QUOTE_ISSUER = "demo-tdx-simulator"
DEMO_NRAS_ISSUER = "demo-nras-simulator"
_MEASUREMENT_FIELDS = (("mr_td", "mr_td"), ("rtmr0", "rt_mr0"), ("rtmr1", "rt_mr1"),
                       ("rtmr2", "rt_mr2"), ("rtmr3", "rt_mr3"))


def load_trust_root(pem):
    """Parse and validate the pasted demo public key (ECDSA P-384, PEM). Returns the key object."""
    from cryptography.hazmat.primitives.asymmetric import ec
    from cryptography.hazmat.primitives.serialization import load_pem_public_key

    try:
        key = load_pem_public_key(str(pem).strip().encode())
    except Exception as exc:
        raise ValueError("The demo trust root is not a PEM public key: {}".format(exc)) from exc
    if not isinstance(key, ec.EllipticCurvePublicKey) or key.curve.name != "secp384r1":
        raise ValueError("The demo trust root must be an ECDSA P-384 public key.")
    return key


def _bytes(claims, name, size):
    try:
        value = bytes.fromhex(claims[name])
    except (KeyError, TypeError, ValueError) as exc:
        raise AttestationError("The demo quote has no valid '{}'.".format(name)) from exc
    if len(value) != size:
        raise AttestationError("The demo quote's '{}' must be {} bytes.".format(name, size))
    return value


def verify_quote(quote, allowed_tcb, trust_root_pem):
    """Counterpart of tdx.verify_quote for demo evidence. Returns (parsed, tcb_status) in the same shape."""
    import jwt

    try:
        token = quote.decode("ascii") if isinstance(quote, (bytes, bytearray)) else str(quote)
        claims = jwt.decode(token, load_trust_root(trust_root_pem), algorithms=["ES384"],
                            issuer=DEMO_QUOTE_ISSUER, options={"require": ["exp", "iat", "iss"]})
    except Exception as exc:
        raise AttestationError("The demo quote is not signed by the configured demo trust root: {}".format(exc)) from exc

    status = str(claims.get("tcb_status"))
    if status not in allowed_tcb:
        raise AttestationError("Intel TCB status {!r} is not allowed (allowed: {}; advisories: []).".format(
            status, sorted(allowed_tcb)))

    report = SimpleNamespace(report_data=_bytes(claims, "report_data", 64))
    for claim, attribute in _MEASUREMENT_FIELDS:
        setattr(report, attribute, _bytes(claims, claim, 48))
    return SimpleNamespace(report=report, demo=True), status


def gpu_key_resolver(trust_root_pem):
    key = load_trust_root(trust_root_pem)
    return lambda token: key
