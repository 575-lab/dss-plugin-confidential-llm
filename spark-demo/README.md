# Spark demo: confidential LLM flow without confidential hardware

Runs the whole Confidential LLM flow on a DGX Spark (or any Linux box with a GPU and a model server):
attestation handshake, nonce/key binding, pin checks, encrypted prompts and answers. The **hardware
evidence is simulated**. It's signed by a demo key this folder creates, not by Intel or NVIDIA.

| Real | Simulated |
|---|---|
| HPKE key per gateway start, EHBP request/response encryption | "TDX quote" (a JWT signed by the demo key) |
| Nonce + key binding (`binding.py`), replay protection | "NVIDIA NRAS tokens" (signed by the same demo key) |
| Pin, TCB-status and GPU-claim checks (production code) | Measurements, derived from the Spark's real kernel/cmdline |
| Re-attestation when the gateway restarts with a new key | |

> **Say this in the demo:** anyone with `demo-keys/demo_root_private.pem` can forge passing evidence.
> On real hardware, Intel's and NVIDIA's keys take its place, and the private half never leaves their chips.

## 1. On the Spark: model server + gateway

```bash
# a model server (Ollama example; vLLM on :8000 works too)
ollama pull qwen2.5:3b

cd spark-demo
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt

# LAN demo with https + token (recommended when DSS runs on another machine)
python3 demo_gateway.py --tls-host 192.168.1.50 --token "$(openssl rand -hex 16)" --show-plaintext
# local-only demo:  python3 demo_gateway.py --host 127.0.0.1
```

On start it prints the **demo trust root**, the **measurements to pin**, and (with `--tls-host`) where the
self-signed cert is. The demo key stays the same across restarts. The HPKE key changes on every start.

## 2. Quick check without DSS

```bash
python3 demo_check.py --url https://192.168.1.50:8443 --token <token> \
    --ca-file demo-keys/gateway_cert.pem --print-pins > pins.json
python3 demo_check.py --url https://192.168.1.50:8443 --token <token> \
    --ca-file demo-keys/gateway_cert.pem --pins pins.json --model qwen2.5:3b
```

## 3. In Dataiku DSS

1. Install this fork (`make dev` builds `dist/…zip`). Its plugin id is `confidential-llm-demo`, so it doesn't
   overwrite the original.
2. Create a **Confidential endpoint** preset: the gateway URL, the token, the certificate PEM, and the demo key in
   **DEMO trust root**.
3. Run the **Attest confidential endpoint** macro. It shows an orange *DEMO attestation verified* banner and
   the measurements. Paste them into **Pinned TDX measurements**.
4. Create the Custom LLM connection (model `qwen2.5:3b`) and use it from Prompt Studio.
   Traces carry `attestation_demo_mode: true`.

Leave **DEMO trust root** empty and the plugin behaves exactly like the original: it will refuse this gateway.

## 4. Showing the plugin refuse

Restart the gateway with one of these flags and send a prompt (or run the macro):

| Flag | Simulates | Plugin says |
|---|---|---|
| `--tamper measurements` | booted an unreviewed kernel | identity doesn't match pins (differs in: rtmr1) |
| `--tamper binding` | evidence for a different key / man-in-the-middle | quote doesn't commit to this nonce and key |
| `--tamper gpu` | GPU in debug mode | GPU claim dbgstat='enabled' |
| `--tcb-status OutOfDate` | unpatched CPU microcode | TCB status not allowed |
| `--firmware-label v2` | firmware update | differs in: mr_td, rtmr0 |
| *(real)* reboot the Spark into a new kernel | kernel update | differs in: rtmr1 |
| *(real)* restart the gateway | new key | transparent re-attestation, then works |

`--show-plaintext` logs decrypted prompts on the gateway, to show that only the gateway (the "enclave") reads them.
Compare with `tcpdump -A port 8443`, where only ciphertext appears.

## Files

- `demo_gateway.py`: the gateway (EHBP server, simulated evidence, forwarding to the model server).
- `demo_check.py`: drives the plugin's own library from the command line.
- `mock_model_server.py`: fake OpenAI endpoint for testing without a model.
- Plugin changes: `python-lib/confidentialllm/demo.py` (new), small hooks in `config.py`, `attest.py`,
  `gpu.py`, the preset, the macro and the LLM trace. Tests: `tests/python/unit/test_demo.py`.
