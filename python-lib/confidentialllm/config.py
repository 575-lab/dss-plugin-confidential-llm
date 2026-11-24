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

import ipaddress
import json
import re
import ssl
from dataclasses import dataclass
from urllib.parse import urlsplit

from .errors import ConfigError

TEE_TDX = "tdx"
TEE_GCE_VTPM = "gce-vtpm"
TEE_TYPES = (TEE_TDX, TEE_GCE_VTPM)
TEE_LABELS = {TEE_TDX: "Intel TDX", TEE_GCE_VTPM: "GCE Confidential VM (vTPM)"}

PIN_KEYS = ("mr_td", "rtmr0", "rtmr1", "rtmr2")  # Intel TDX
VTPM_PIN_KEYS = ("pcr0", "pcr4", "pcr7", "pcr8", "pcr9", "instance_id")  # GCE vTPM
_HEX96 = re.compile(r"^[0-9a-f]{96}$")  # TDX measurements are 48 bytes
_HEX64 = re.compile(r"^[0-9a-f]{64}$")  # SHA-256 PCRs are 32 bytes
_DIGITS = re.compile(r"^[0-9]{1,20}$")  # GCE instance ids are 64-bit integers


def _check_tdx_pin(entry):
    unknown = set(entry) - set(PIN_KEYS)
    if unknown:
        raise ConfigError("Unknown TDX pin fields: {}. Allowed: {}.".format(sorted(unknown), list(PIN_KEYS)))
    if "mr_td" not in entry:
        raise ConfigError("Every TDX pin set must include 'mr_td'.")
    for key, value in entry.items():
        if not isinstance(value, str) or not _HEX96.match(value.lower()):
            raise ConfigError("Pin '{}' must be 96 hex characters (a 48-byte TDX measurement).".format(key))
    return {key: value.lower() for key, value in entry.items()}


def _check_vtpm_pin(entry):
    unknown = set(entry) - set(VTPM_PIN_KEYS)
    if unknown:
        raise ConfigError("Unknown vTPM pin fields: {}. Allowed: {}.".format(sorted(unknown), list(VTPM_PIN_KEYS)))
    missing = [key for key in ("pcr0", "pcr4", "pcr7") if key not in entry]
    if missing:
        raise ConfigError("Every vTPM pin set must include pcr0, pcr4 and pcr7 (missing: {}).".format(", ".join(missing)))
    out = {}
    for key, value in entry.items():
        if key == "instance_id":
            value = str(value) if isinstance(value, int) and not isinstance(value, bool) else value
            if not isinstance(value, str) or not _DIGITS.match(value):
                raise ConfigError("Pin 'instance_id' must be the numeric GCE instance id.")
            out[key] = value
        elif not isinstance(value, str) or not _HEX64.match(value.lower()):
            raise ConfigError("Pin '{}' must be 64 hex characters (a SHA-256 PCR value).".format(key))
        else:
            out[key] = value.lower()
    return out

DEFAULT_ALLOWED_TCB = ("UpToDate",)
DEFAULT_TTL_MINUTES = 10
DEFAULT_TIMEOUT_SECONDS = 600


def parse_pins(raw, tee_type=TEE_TDX):
    """Parse the pinned measurements for ``tee_type``.

    Accepts a JSON object, or a list of objects when several identities are acceptable (for
    example during a rolling update). Returns a tuple of dicts; an empty tuple means "no pins".
    """
    if raw is None or (isinstance(raw, str) and not raw.strip()):
        return ()
    try:
        data = json.loads(raw) if isinstance(raw, str) else raw
    except ValueError as exc:
        raise ConfigError("Pinned measurements are not valid JSON: {}".format(exc)) from exc

    if isinstance(data, dict):
        data = [data]
    if not isinstance(data, list) or not data:
        raise ConfigError("Pinned measurements must be a JSON object or a non-empty list of objects.")

    check = _check_vtpm_pin if tee_type == TEE_GCE_VTPM else _check_tdx_pin
    pins = []
    for entry in data:
        if not isinstance(entry, dict):
            raise ConfigError("Each pinned measurement set must be a JSON object.")
        pins.append(check(entry))
    return tuple(pins)


def _normalize_url(url):
    if not url or not isinstance(url, str):
        raise ConfigError("Gateway URL is required.")
    parts = urlsplit(url.strip())
    if parts.scheme not in ("http", "https") or not parts.hostname:
        raise ConfigError("Gateway URL must start with http:// or https:// and include a host.")
    if parts.username or parts.password:
        raise ConfigError("Do not put credentials in the gateway URL; use the gateway token field.")
    if parts.query or parts.fragment:
        raise ConfigError("Gateway URL must not contain a query string or fragment.")
    return "{}://{}{}".format(parts.scheme, parts.netloc, parts.path.rstrip("/"))


def _is_loopback_host(host):
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def parse_ca_certificate(pem):
    """Validate a PEM certificate the administrator pasted. Returns the stripped PEM or None."""
    if pem is None or not str(pem).strip():
        return None
    pem = str(pem).strip()
    try:
        ssl.create_default_context(cadata=pem)
    except (ssl.SSLError, ValueError) as exc:
        raise ConfigError("The trusted certificate is not a valid PEM certificate: {}".format(exc)) from exc
    return pem


def tls_verify(config):
    """What httpx should pass as ``verify``.

    With a trusted certificate configured, ONLY that certificate is trusted (the system store is not consulted),
    so the gateway must present exactly the certificate the administrator reviewed.
    """
    if config.ca_certificate:
        return ssl.create_default_context(cadata=config.ca_certificate)
    return True


def _positive_int(value, default, label):
    if value in (None, ""):
        return default
    try:
        number = int(float(value))
    except (TypeError, ValueError) as exc:
        raise ConfigError("{} must be a number.".format(label)) from exc
    if number <= 0:
        raise ConfigError("{} must be greater than zero.".format(label))
    return number


@dataclass(frozen=True)
class EndpointConfig:
    base_url: str
    pins: tuple
    allow_unpinned: bool
    allowed_tcb: frozenset
    ttl_seconds: int
    timeout_seconds: int
    gateway_token: str = None
    ca_certificate: str = None
    tee_type: str = TEE_TDX
    allowed_gpu_models: tuple = ()

    @property
    def cache_key(self):
        """Sessions are shared between connections with the same trust settings."""
        # Must be hashable (it is a dict key): pins are dicts, so freeze each one into sorted items.
        frozen_pins = tuple(tuple(sorted(pin.items())) for pin in self.pins)
        return (self.base_url, self.tee_type, frozen_pins, self.allowed_tcb, self.allowed_gpu_models,
                self.ttl_seconds, self.gateway_token, self.ca_certificate)

    @classmethod
    def from_preset(cls, preset, *, discovery=False):
        """Build a validated config from the parameter-set preset.

        ``discovery=True`` is only for the "Attest endpoint" macro, which has to run without pins to
        show an administrator the measurements to pin. It is never used by the LLM connection.
        """
        preset = preset or {}
        tee_type = preset.get("tee_type") or TEE_TDX  # presets made before 0.2.0 have no TEE type: they are TDX
        if tee_type not in TEE_TYPES:
            raise ConfigError("Unknown TEE type {!r}. Allowed: {}.".format(tee_type, list(TEE_TYPES)))
        pins = parse_pins(preset.get("pins"), tee_type)
        allow_unpinned = bool(preset.get("allow_unpinned", False))

        if not pins and not (allow_unpinned or discovery):
            raise ConfigError(
                "No pinned measurements are configured. Without them any genuine {} machine would be "
                "accepted. Run the 'Attest confidential endpoint' macro, review the measurements, and paste "
                "them into the preset (or explicitly allow unpinned endpoints for testing).".format(TEE_LABELS[tee_type])
            )

        allowed_tcb = preset.get("allowed_tcb") or list(DEFAULT_ALLOWED_TCB)
        if isinstance(allowed_tcb, str):
            allowed_tcb = [item.strip() for item in allowed_tcb.split(",") if item.strip()]
        allowed_tcb = frozenset(allowed_tcb)
        if not allowed_tcb:
            raise ConfigError("At least one allowed Intel TCB status is required.")
        if any("revoked" in status.lower() for status in allowed_tcb):
            raise ConfigError("A revoked TCB can never be allowed.")

        gpu_models = preset.get("allowed_gpu_models") or []
        if isinstance(gpu_models, str):
            gpu_models = gpu_models.split(",")
        gpu_models = tuple(sorted({str(m).strip() for m in gpu_models if str(m).strip()}))

        base_url = _normalize_url(preset.get("gateway_url"))
        parts = urlsplit(base_url)
        token = preset.get("gateway_token") or None
        ca_certificate = parse_ca_certificate(preset.get("ca_certificate"))
        if token and parts.scheme == "http" and not _is_loopback_host(parts.hostname):
            raise ConfigError(
                "Refusing to send the gateway token over plain http to {}. Use an https gateway URL "
                "(with the trusted certificate if it is self-signed).".format(parts.hostname))
        if ca_certificate and parts.scheme != "https":
            raise ConfigError("A trusted certificate only applies to an https gateway URL.")

        return cls(
            base_url=base_url,
            pins=pins,
            allow_unpinned=allow_unpinned or discovery,
            allowed_tcb=allowed_tcb,
            ttl_seconds=_positive_int(preset.get("attestation_ttl_minutes"), DEFAULT_TTL_MINUTES, "Attestation lifetime") * 60,
            timeout_seconds=_positive_int(preset.get("request_timeout_seconds"), DEFAULT_TIMEOUT_SECONDS, "Request timeout"),
            gateway_token=token,
            ca_certificate=ca_certificate,
            tee_type=tee_type,
            allowed_gpu_models=gpu_models,
        )
