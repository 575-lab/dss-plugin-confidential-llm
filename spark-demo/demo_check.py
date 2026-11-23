#!/usr/bin/env python3
"""Exercise the plugin's own library against the demo gateway, without DSS.

  python3 spark-demo/demo_check.py --url http://127.0.0.1:8443 --model qwen2.5:3b
  python3 spark-demo/demo_check.py --url ... --pins pins.json     # enforce pins (else unpinned)
  python3 spark-demo/demo_check.py --url ... --print-pins > pins.json

The demo trust root defaults to spark-demo/demo-keys/demo_root_public.pem (written by the gateway).
"""

import argparse
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "python-lib"))

from confidentialllm import attest, client  # noqa: E402
from confidentialllm.config import EndpointConfig  # noqa: E402
from confidentialllm.errors import ConfidentialLLMError  # noqa: E402
from confidentialllm.session import AttestedSession  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8443")
    parser.add_argument("--model", default="qwen2.5:3b")
    parser.add_argument("--token")
    parser.add_argument("--ca-file", help="the gateway's self-signed certificate (https)")
    parser.add_argument("--trust-root", default=str(HERE / "demo-keys" / "demo_root_public.pem"))
    parser.add_argument("--pins", help="JSON file with pinned measurements")
    parser.add_argument("--print-pins", action="store_true")
    parser.add_argument("--prompt", default="In one sentence: what is remote attestation?")
    args = parser.parse_args()

    preset = {
        "gateway_url": args.url,
        "gateway_token": args.token,
        "ca_certificate": Path(args.ca_file).read_text() if args.ca_file else None,
        "demo_trust_root": Path(args.trust_root).read_text(),
        "pins": Path(args.pins).read_text() if args.pins else None,
        "allow_unpinned": not args.pins,
    }
    config = EndpointConfig.from_preset(preset)

    try:
        started = time.time()
        result = attest.attest_endpoint(config)
    except ConfidentialLLMError as exc:
        print("ATTESTATION REFUSED: {}".format(exc))
        return 2

    measurements = result.info["measurements"]
    if args.print_pins:
        print(json.dumps({k: measurements[k] for k in ("mr_td", "rtmr0", "rtmr1", "rtmr2")}, indent=2))
        return 0

    print("Attestation OK in {:.2f}s  (demo_mode={}, pinned={}, tcb={}, key={})".format(
        time.time() - started, result.info["demo_mode"], result.pinned, result.info["tcb_status"],
        result.info["key_fingerprint"]))
    for gpu in result.info["gpus"]:
        print("  {}: {}".format(gpu["gpu"], gpu["model"]))

    session = AttestedSession(config, attest_fn=lambda cfg: result)
    body = {"model": args.model, "messages": [{"role": "user", "content": args.prompt}]}

    try:
        answer = client.complete(session, dict(body, stream=False))
        print("\nEncrypted completion:\n  " + answer["choices"][0]["message"]["content"].strip())

        print("\nEncrypted stream:\n  ", end="")
        for event in client.stream(session, dict(body, stream=True)):
            for choice in event.get("choices", []):
                print((choice.get("delta") or {}).get("content") or "", end="", flush=True)
        print()
    except ConfidentialLLMError as exc:
        print("REQUEST FAILED: {}".format(exc))
        return 3
    return 0


if __name__ == "__main__":
    sys.exit(main())
