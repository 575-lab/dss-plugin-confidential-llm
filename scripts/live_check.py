#!/usr/bin/env python3
"""Run the plugin's library against a real gateway, the way DSS will: TLS + token + pins + encrypted chat.

Environment:
  GATEWAY_TOKEN   bearer token (never printed)
  GATEWAY_URL     default https://localhost:8443  (use the SSH tunnel locally)
  CA_FILE         gateway certificate PEM (required for https with a self-signed certificate)
  PINS_FILE       pinned measurements JSON (required; see the "Attest confidential endpoint" macro)
  TEE_TYPE        tdx (default) or gce-vtpm (GCP AMD SEV Confidential VM, e.g. G4)
  ALLOW_UNPINNED  set to 1 to attest without pins and print the measurements (discovery only)
  MODEL           default qwen3.8-27b-uncensored
"""
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python-lib"))

from confidentialllm import client, messages  # noqa: E402
from confidentialllm.config import EndpointConfig  # noqa: E402
from confidentialllm.errors import ConfidentialLLMError  # noqa: E402
from confidentialllm.session import AttestedSession  # noqa: E402

token = os.environ.get("GATEWAY_TOKEN")
if not token:
    sys.exit("Set GATEWAY_TOKEN.")

preset = {
    "gateway_url": os.environ.get("GATEWAY_URL", "https://localhost:8443"),
    "gateway_token": token,
    "ca_certificate": open(os.environ["CA_FILE"]).read() if os.environ.get("CA_FILE") else None,
    "pins": open(os.environ["PINS_FILE"]).read() if os.environ.get("PINS_FILE") else None,
    "tee_type": os.environ.get("TEE_TYPE", "tdx"),
    "allow_unpinned": os.environ.get("ALLOW_UNPINNED") == "1",
}
model = os.environ.get("MODEL", "qwen3.8-27b-uncensored")

try:
    session = AttestedSession(EndpointConfig.from_preset(preset))
    t = time.time()
    result = session.current()
    print("attested in %.1fs | tee=%s | pinned=%s | tcb=%s | gpus=%s" % (
        time.time() - t, result.info["tee"], result.pinned, result.info["tcb_status"], result.info["gpus"]))
    if "platform" in result.info:
        print("platform: %s" % result.info["platform"])
    print("measurements: %s" % result.info["measurements"])

    t = time.time()
    body = messages.build_request(model, {"messages": [{"role": "user", "content": "Reply with exactly: pong"}]}, {"maxOutputTokens": 600})
    print("complete: %r in %.1fs" % (messages.to_dss_response(client.complete(session, body)), time.time() - t))

    t = time.time()
    text = "".join(messages.stream_delta_text(e) for e in client.stream(
        session, messages.build_request(model, {"messages": [{"role": "user", "content": "Count to five in words."}]}, {"maxOutputTokens": 800}, stream=True)))
    print("stream: %r in %.1fs" % (text[:100], time.time() - t))
    print("\nOK: TLS, token, attestation, pins and encrypted chat all work.")
except ConfidentialLLMError as exc:
    sys.exit("FAILED: %s: %s" % (type(exc).__name__, exc))
