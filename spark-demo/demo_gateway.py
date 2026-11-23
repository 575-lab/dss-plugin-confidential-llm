#!/usr/bin/env python3
"""DEMO confidential gateway for a machine WITHOUT confidential-computing hardware (e.g. a DGX Spark).

What is real:
  * The HPKE key, the EHBP body encryption (requests and responses), the nonce/key binding.
  * Forwarding to a local OpenAI-compatible model server (Ollama, vLLM, ...).

What is SIMULATED:
  * The "TDX quote" and the "NVIDIA GPU tokens". They are signed by a demo key that this script
    creates, not by Intel or NVIDIA. They prove nothing about hardware. The plugin only accepts
    them in its explicit demo mode, where an administrator pastes this demo key's public half.

Endpoints (same shape as the real gateway the plugin expects):
  GET  /.well-known/hpke-keys          HPKE public key config (application/ohttp-keys)
  GET  /attestation?nonce=<64 hex>     {"tdx_quote_b64": ..., "gpu": {...}} bound to the nonce + key
  POST /v1/chat/completions            EHBP-encrypted body only, forwarded to the model server
"""

import argparse
import base64
import datetime
import hashlib
import ipaddress
import json
import logging
import os
import platform
import secrets
import ssl
import struct
import subprocess
import sys
import time
from pathlib import Path

import jwt
from aiohttp import ClientSession, ClientTimeout, web
from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives.asymmetric.x25519 import X25519PrivateKey
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.hkdf import HKDFExpand
from cryptography.hazmat.primitives import hmac as crypto_hmac
from cryptography.x509.oid import NameOID
from pyhpke import AEADId, CipherSuite, KDFId, KEMId

LOG = logging.getLogger("demo-gateway")

# --- Wire constants: must match the EHBP spec / tinfoil-ehbp and the plugin's binding.py -------------
ENCAPSULATED_KEY_HEADER = "Ehbp-Encapsulated-Key"
RESPONSE_NONCE_HEADER = "Ehbp-Response-Nonce"
KEYS_MEDIA_TYPE = "application/ohttp-keys"
KEY_CONFIG_PROBLEM_TYPE = "urn:ietf:params:ehbp:error:key-config"
KEM_X25519, KDF_SHA256, AEAD_AES256GCM = 0x0020, 0x0001, 0x0002
HPKE_REQUEST_INFO = b"ehbp request"
EXPORT_LABEL = b"ehbp response"
BINDING_LABEL = b"ehbp-attest-v1\x00"

# --- Demo evidence format (understood only by the plugin's demo mode) ---------------------------------
DEMO_QUOTE_ISSUER = "demo-tdx-simulator"
DEMO_NRAS_ISSUER = "demo-nras-simulator"
DEMO_KID = "demo-root-1"

MAX_FRAME = 64 * 1024 * 1024


# ======================================================================================================
# Keys
# ======================================================================================================

def load_or_create_demo_root(key_dir):
    """The demo 'hardware root': an ECDSA P-384 key that signs the fake quote and fake GPU tokens.

    Kept on disk so the plugin's pasted trust root survives gateway restarts.
    """
    key_dir.mkdir(parents=True, exist_ok=True)
    private_path = key_dir / "demo_root_private.pem"
    public_path = key_dir / "demo_root_public.pem"
    if private_path.exists():
        private_key = serialization.load_pem_private_key(private_path.read_bytes(), password=None)
    else:
        private_key = ec.generate_private_key(ec.SECP384R1())
        private_path.write_bytes(private_key.private_bytes(
            serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8, serialization.NoEncryption()))
        os.chmod(private_path, 0o600)
    public_pem = private_key.public_key().public_bytes(
        serialization.Encoding.PEM, serialization.PublicFormat.SubjectPublicKeyInfo)
    public_path.write_bytes(public_pem)
    return private_key, public_pem.decode()


def new_hpke_key():
    """A fresh X25519 key per gateway start (like a real enclave: the private key never leaves memory)."""
    private = X25519PrivateKey.generate()
    raw_private = private.private_bytes(serialization.Encoding.Raw, serialization.PrivateFormat.Raw,
                                        serialization.NoEncryption())
    raw_public = private.public_key().public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw)
    key_config = (bytes([0]) + struct.pack(">H", KEM_X25519) + raw_public
                  + struct.pack(">HHH", 4, KDF_SHA256, AEAD_AES256GCM))
    return raw_private, key_config


def _suite():
    return CipherSuite.new(KEMId.DHKEM_X25519_HKDF_SHA256, KDFId.HKDF_SHA256, AEADId.AES256_GCM)


# ======================================================================================================
# Simulated measurements
# ======================================================================================================

def _sha384(*parts):
    h = hashlib.sha384()
    for part in parts:
        h.update(part if isinstance(part, bytes) else str(part).encode())
    return h.digest()


def _read(path):
    try:
        return Path(path).read_bytes()
    except OSError:
        return b""


def simulated_measurements(firmware_label, tamper):
    """48-byte values shaped like TDX measurements, derived from this machine's real boot state.

    mr_td / rtmr0 : the "firmware" (a label you choose; change it to simulate a firmware update)
    rtmr1         : the running kernel (/proc/version)  -> changes when you reboot into a new kernel
    rtmr2         : the kernel command line (/proc/cmdline)
    """
    values = {
        "mr_td": _sha384("demo-mr_td", firmware_label),
        "rtmr0": _sha384("demo-rtmr0", firmware_label),
        "rtmr1": _sha384("demo-rtmr1", _read("/proc/version") or platform.release()),
        "rtmr2": _sha384("demo-rtmr2", _read("/proc/cmdline")),
        "rtmr3": bytes(48),
    }
    if "measurements" in tamper:
        values["rtmr1"] = _sha384("demo-rtmr1", "an-unreviewed-kernel")
    return {key: value.hex() for key, value in values.items()}


def gpu_inventory():
    """Real GPU names/driver from nvidia-smi when available (informational only)."""
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=10, check=True).stdout
        gpus = [line.split(",") for line in out.strip().splitlines() if line.strip()]
        return [(name.strip(), driver.strip()) for name, driver in gpus] or [("unknown GPU", "unknown")]
    except Exception:  # no NVIDIA driver here: still produce a demo token
        return [("no GPU detected", "n/a")]


# ======================================================================================================
# Evidence
# ======================================================================================================

def binding_digest(nonce, key_config):
    return hashlib.sha256(BINDING_LABEL + nonce + key_config).digest()


def make_evidence(state, nonce):
    key_config = state["key_config"]
    tamper = state["tamper"]
    digest = binding_digest(nonce, key_config)
    if "binding" in tamper:  # evidence for some other key: the plugin must refuse it
        digest = binding_digest(nonce, b"some-other-key")
    now = int(time.time())
    root = state["demo_root"]

    quote_claims = {
        "iss": DEMO_QUOTE_ISSUER,
        "iat": now,
        "exp": now + 300,
        "tee": "DEMO-TDX (simulated, no hardware)",
        "tcb_status": state["tcb_status"],
        "report_data": (digest + bytes(32)).hex(),
        **state["measurements"],
    }
    quote = jwt.encode(quote_claims, root, algorithm="ES384", headers={"kid": DEMO_KID, "typ": "demo-tdx-quote"})

    device_tokens = {}
    for index, (name, driver) in enumerate(state["gpus"]):
        claims = {
            "iss": DEMO_NRAS_ISSUER,
            "iat": now,
            "exp": now + 300,
            "eat_nonce": digest.hex(),
            "hwmodel": "{} (SIMULATED attestation)".format(name),
            "x-nvidia-gpu-driver-version": driver,
            "x-nvidia-gpu-vbios-version": "demo",
            "secboot": True,
            "dbgstat": "enabled" if "gpu" in tamper else "disabled",
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
        for chain in ("x-nvidia-gpu-attestation-report-cert-chain", "x-nvidia-gpu-driver-rim-cert-chain",
                      "x-nvidia-gpu-vbios-rim-cert-chain"):
            claims[chain] = {"x-nvidia-cert-status": "valid", "x-nvidia-cert-ocsp-status": "good"}
        device_tokens["GPU-{}".format(index)] = jwt.encode(claims, root, algorithm="ES384", headers={"kid": DEMO_KID})

    overall = jwt.encode({"iss": DEMO_NRAS_ISSUER, "iat": now, "exp": now + 300, "eat_nonce": digest.hex(),
                          "x-nvidia-overall-att-result": True},
                         root, algorithm="ES384", headers={"kid": DEMO_KID})
    return {
        "demo": True,
        "tdx_quote_b64": base64.b64encode(quote.encode()).decode(),
        "gpu": {"nras_overall_jwt": overall, "nras_device_jwts": device_tokens},
    }


# ======================================================================================================
# EHBP server side
# ======================================================================================================

class FrameReader:
    """Split a byte stream into LEN(u32 BE) || CIPHERTEXT frames."""

    def __init__(self):
        self.buffer = bytearray()

    def push(self, data):
        self.buffer += data
        frames = []
        while len(self.buffer) >= 4:
            length = struct.unpack_from(">I", self.buffer)[0]
            if length > MAX_FRAME:
                raise ValueError("request frame too large")
            if len(self.buffer) < 4 + length:
                break
            frames.append(bytes(self.buffer[4:4 + length]))
            del self.buffer[:4 + length]
        return frames


def _hkdf_extract(salt, ikm):
    h = crypto_hmac.HMAC(salt, hashes.SHA256())
    h.update(ikm)
    return h.finalize()


def _hkdf_expand(prk, info, length):
    return HKDFExpand(algorithm=hashes.SHA256(), length=length, info=info).derive(prk)


class ResponseSealer:
    """OHTTP-style response encryption: AES-256-GCM, nonce = base XOR seq, length-prefixed frames."""

    def __init__(self, exported_secret, request_enc):
        self.response_nonce = secrets.token_bytes(32)
        prk = _hkdf_extract(request_enc + self.response_nonce, exported_secret)
        self.aead = AESGCM(_hkdf_expand(prk, b"key", 32))
        self.nonce_base = _hkdf_expand(prk, b"nonce", 12)
        self.seq = 0

    def seal(self, plaintext):
        nonce = bytearray(self.nonce_base)
        for i in range(8):
            nonce[11 - i] ^= (self.seq >> (8 * i)) & 0xFF
        self.seq += 1
        ciphertext = self.aead.encrypt(bytes(nonce), plaintext, b"")
        return struct.pack(">I", len(ciphertext)) + ciphertext


def key_config_problem(title):
    return web.json_response({"type": KEY_CONFIG_PROBLEM_TYPE, "title": title}, status=422,
                             content_type="application/problem+json")


async def decrypt_request(request, state):
    """Open the HPKE context and decrypt every request frame. Returns (plaintext, exported_secret, enc)."""
    enc_hex = request.headers.get(ENCAPSULATED_KEY_HEADER)
    enc = bytes.fromhex(enc_hex)
    suite = _suite()
    private_key = suite.kem.deserialize_private_key(state["hpke_private"])
    recipient = suite.create_recipient_context(enc, private_key, info=HPKE_REQUEST_INFO)
    exported_secret = recipient.export(EXPORT_LABEL, 32)
    reader = FrameReader()
    plaintext = bytearray()
    async for chunk in request.content.iter_any():
        for frame in reader.push(chunk):
            plaintext += recipient.open(frame, b"")
    if reader.buffer:
        raise ValueError("truncated request frame")
    return bytes(plaintext), exported_secret, enc


# ======================================================================================================
# HTTP handlers
# ======================================================================================================

def check_token(request, state):
    expected = state["token"]
    if not expected:
        return None
    given = request.headers.get("Authorization", "")
    if not secrets.compare_digest(given.encode(), "Bearer {}".format(expected).encode()):
        return web.json_response({"error": "unauthorized"}, status=401)
    return None


async def handle_keys(request):
    state = request.app["state"]
    if (denied := check_token(request, state)):
        return denied
    return web.Response(body=state["key_config"], content_type=KEYS_MEDIA_TYPE)


async def handle_attestation(request):
    state = request.app["state"]
    if (denied := check_token(request, state)):
        return denied
    try:
        nonce = bytes.fromhex(request.query.get("nonce", ""))
    except ValueError:
        nonce = b""
    if len(nonce) != 32:
        return web.json_response({"error": "nonce must be 64 hex characters"}, status=400)
    LOG.info("Attestation requested (nonce %s...)", nonce.hex()[:12])
    return web.json_response(make_evidence(state, nonce))


async def handle_chat(request):
    state = request.app["state"]
    if (denied := check_token(request, state)):
        return denied
    if ENCAPSULATED_KEY_HEADER not in request.headers:
        return web.json_response({"error": "plaintext requests are refused; use EHBP"}, status=400)

    try:
        body, exported_secret, enc = await decrypt_request(request, state)
    except Exception as exc:  # wrong/old key or corrupted body: tell the client to re-fetch the key
        LOG.warning("Could not decrypt request: %s", exc)
        return key_config_problem("request could not be decrypted with the current key")

    try:
        payload = json.loads(body)
    except ValueError:
        payload = {}
    if state["show_plaintext"]:
        LOG.info("Decrypted request (inside the 'enclave'): %s", body[:500].decode("utf-8", "replace"))
    else:
        LOG.info("Decrypted request: model=%s, %d messages, stream=%s",
                 payload.get("model"), len(payload.get("messages") or []), payload.get("stream"))

    sealer = ResponseSealer(exported_secret, enc)
    upstream_url = state["upstream"] + "/v1/chat/completions"
    # The model server sees the gateway's own Host (aiohttp sets it from the URL): Ollama refuses foreign hosts.
    async with ClientSession(timeout=ClientTimeout(total=None, sock_read=state["timeout"])) as session:
        try:
            async with session.post(upstream_url, data=body, headers={"Content-Type": "application/json"}) as up:
                response = web.StreamResponse(status=up.status, headers={
                    RESPONSE_NONCE_HEADER: sealer.response_nonce.hex(),
                    "Content-Type": up.headers.get("Content-Type", "application/json"),
                })
                await response.prepare(request)
                wrote = False
                async for chunk in up.content.iter_any():
                    if chunk:
                        await response.write(sealer.seal(chunk))
                        wrote = True
                if not wrote:
                    await response.write(sealer.seal(b"{}"))
                await response.write_eof()
                return response
        except OSError as exc:
            LOG.error("Model server unreachable at %s: %s", upstream_url, exc)
            response = web.StreamResponse(status=502, headers={RESPONSE_NONCE_HEADER: sealer.response_nonce.hex(),
                                                               "Content-Type": "application/json"})
            await response.prepare(request)
            await response.write(sealer.seal(json.dumps({"error": "model server unreachable"}).encode()))
            await response.write_eof()
            return response


# ======================================================================================================
# TLS (optional, self-signed)
# ======================================================================================================

def self_signed_tls(key_dir, hosts):
    key = ec.generate_private_key(ec.SECP256R1())
    names = []
    for host in hosts:
        try:
            names.append(x509.IPAddress(ipaddress.ip_address(host)))
        except ValueError:
            names.append(x509.DNSName(host))
    subject = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, hosts[0])])
    now = datetime.datetime.now(datetime.timezone.utc)
    cert = (x509.CertificateBuilder().subject_name(subject).issuer_name(subject).public_key(key.public_key())
            .serial_number(x509.random_serial_number()).not_valid_before(now - datetime.timedelta(minutes=5))
            .not_valid_after(now + datetime.timedelta(days=365))
            .add_extension(x509.SubjectAlternativeName(names), critical=False)
            .add_extension(x509.BasicConstraints(ca=True, path_length=None), critical=True)
            .sign(key, hashes.SHA256()))
    cert_path, key_path = key_dir / "gateway_cert.pem", key_dir / "gateway_key.pem"
    if not (cert_path.exists() and key_path.exists()):
        cert_path.write_bytes(cert.public_bytes(serialization.Encoding.PEM))
        key_path.write_bytes(key.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8,
                                               serialization.NoEncryption()))
        os.chmod(key_path, 0o600)
    context = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
    context.load_cert_chain(cert_path, key_path)
    return context, cert_path


# ======================================================================================================
# Main
# ======================================================================================================

def build_app(args):
    key_dir = Path(args.key_dir)
    demo_root, demo_root_pem = load_or_create_demo_root(key_dir)
    hpke_private, key_config = new_hpke_key()
    tamper = set(args.tamper or [])
    state = {
        "demo_root": demo_root,
        "demo_root_pem": demo_root_pem,
        "hpke_private": hpke_private,
        "key_config": key_config,
        "measurements": simulated_measurements(args.firmware_label, tamper),
        "gpus": gpu_inventory(),
        "tcb_status": args.tcb_status,
        "tamper": tamper,
        "upstream": args.upstream.rstrip("/"),
        "token": args.token or os.environ.get("GATEWAY_TOKEN") or None,
        "show_plaintext": args.show_plaintext,
        "timeout": args.timeout,
    }
    app = web.Application(client_max_size=MAX_FRAME)
    app["state"] = state
    app.router.add_get("/.well-known/hpke-keys", handle_keys)
    app.router.add_get("/attestation", handle_attestation)
    app.router.add_post("/v1/chat/completions", handle_chat)
    return app, state


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8443)
    parser.add_argument("--upstream", default=os.environ.get("UPSTREAM", "http://127.0.0.1:11434"),
                        help="OpenAI-compatible model server (Ollama: :11434, vLLM: :8000)")
    parser.add_argument("--token", help="require this bearer token (or set GATEWAY_TOKEN)")
    parser.add_argument("--tls-host", action="append",
                        help="serve https with a self-signed cert for this IP/hostname (repeatable)")
    parser.add_argument("--key-dir", default=str(Path(__file__).resolve().parent / "demo-keys"))
    parser.add_argument("--firmware-label", default="demo-firmware-v1",
                        help="change to simulate a firmware update (changes mr_td/rtmr0)")
    parser.add_argument("--tcb-status", default="UpToDate", help="e.g. OutOfDate to show the TCB check")
    parser.add_argument("--tamper", action="append", choices=["measurements", "binding", "gpu"],
                        help="break the evidence on purpose to show the plugin refusing")
    parser.add_argument("--show-plaintext", action="store_true",
                        help="log decrypted prompts (shows that only the gateway can read them)")
    parser.add_argument("--timeout", type=int, default=600)
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    app, state = build_app(args)

    ssl_context = None
    scheme = "http"
    if args.tls_host:
        ssl_context, cert_path = self_signed_tls(Path(args.key_dir), args.tls_host)
        scheme = "https"

    print("=" * 78)
    print(" DEMO CONFIDENTIAL GATEWAY - attestation is SIMULATED, not hardware-backed")
    print("=" * 78)
    print(" Listening:     {}://{}:{}".format(scheme, args.host, args.port))
    print(" Model server:  {}".format(state["upstream"]))
    print(" HPKE key:      {} (new on every start)".format(hashlib.sha256(state["key_config"]).hexdigest()[:16]))
    if state["tamper"] or state["tcb_status"] != "UpToDate":
        print(" TAMPERING:     {} tcb={}".format(sorted(state["tamper"]), state["tcb_status"]))
    if ssl_context:
        print(" TLS cert:      {}  (paste into the preset's trusted certificate)".format(cert_path))
    print("\n Demo trust root (paste into the preset's 'Demo trust root' field):\n")
    print(state["demo_root_pem"])
    print(" Measurements to pin (what the macro will show):\n")
    print(json.dumps({k: state["measurements"][k] for k in ("mr_td", "rtmr0", "rtmr1", "rtmr2")}, indent=2))
    print("=" * 78, flush=True)

    web.run_app(app, host=args.host, port=args.port, ssl_context=ssl_context, print=None)


if __name__ == "__main__":
    sys.exit(main())
