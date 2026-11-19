# Confidential LLM (DSS plugin)

> **Status: experimental.** The attestation and encryption logic is unit-tested (and was exercised against a real gateway),
> but the plugin components have not yet been validated inside a running DSS instance. Treat it as a proof of concept, not a
> security product, and read "What is and is not protected" before relying on it.

Use an LLM that runs inside a hardware-attested confidential VM (Intel TDX + NVIDIA H100) as a model in the
Dataiku LLM Mesh. The plugin refuses to send anything until the endpoint proves what it is, then encrypts every
prompt to a key that the proof shows lives inside the verified enclave.

```
Prompt Studio / recipe / agent
   └─ (optional) a PII-masking guardrail
        └─ Confidential LLM (this plugin)
             ├─ verify: Intel TDX quote + NVIDIA GPU tokens, bound to a fresh nonce and the gateway's key
             ├─ check:  TDX measurements against the pins an administrator reviewed
             └─ send:   body-encrypted (EHBP/HPKE) request ──▶ gateway ──▶ model server inside the VM
```

## Components

| Component | What it does |
|---|---|
| `parameter-sets/confidential-endpoint` | Gateway URL, access token, pinned measurements, allowed Intel TCB statuses, attestation lifetime. |
| `python-llms/confidential-llm` | The LLM Mesh connection (completion and streaming). Thin adapter over the library. |
| `python-runnables/attest-endpoint` | Macro: attests the endpoint and shows the measurements to review and pin. |
| `python-lib/confidentialllm` | All logic: binding, TDX/GPU verification, session cache, message mapping, EHBP client. |

## Setting it up

1. Install the plugin and build its code environment.
2. Create a preset of **Confidential endpoint** with the gateway URL (and token, if the gateway requires one).
3. Run the macro **Attest confidential endpoint** on that preset. Review the report, then paste the shown
   `mr_td` / `rtmr0` / `rtmr1` / `rtmr2` into the preset's *Pinned TDX measurements*.
4. Create a Custom LLM connection using this plugin, choose the preset, and enter the model name the server exposes.

Without pins the connection refuses to start. The only way around that is the explicit *Allow unpinned endpoint
(testing only)* option, which proves "a genuine TDX machine", not "this machine".

## What is and is not protected

Protected: prompts and answers are unreadable to the network and to the cloud host, and are only decrypted by a
gateway whose key is bound to a verified TDX guest with a verified GPU.

Not protected, by design of the building blocks:

- **Which software runs.** TDX measurements identify the boot environment. The gateway, the model server and the model
  weights are not measured. Pins prove "the machine I reviewed", not "this exact model".
- **The administrator of the VM.** Anyone with root inside the VM can read model-server memory.
- **Headers and sizes.** HTTP metadata (including the gateway token) and message sizes are visible. Use `https` or a
  private network.
- **Forward secrecy and replay.** The body-encryption protocol has neither.
- **Anything before the plugin.** Prompts exist in plaintext inside DSS. Put a PII-masking guardrail in front (for example one built on the open-source Kiji Proxy) to remove PII first.

Attestation depends on Intel's and NVIDIA's public services being reachable from DSS (collateral and signing keys), and
the gateway calls NVIDIA's attestation service itself.

## The gateway

The plugin talks to a small gateway running **inside** the confidential VM, in front of an OpenAI-compatible model server
(for example Ollama). The gateway is not included here. It must provide:

- `GET /.well-known/hpke-keys`: the HPKE public key (`application/ohttp-keys`) used to encrypt request bodies (EHBP).
- `GET /attestation?nonce=<64 hex>`: JSON with `tdx_quote_b64` (an Intel TDX quote whose `report_data` starts with
  `SHA256("ehbp-attest-v1\0" || nonce || key_config)`, rest zero) and `gpu` (`nras_overall_jwt` and `nras_device_jwts`: NVIDIA
  NRAS tokens whose `eat_nonce` is that same digest, produced by `nvattest attest --verifier remote --nonce ...`).
- `POST /v1/chat/completions` with an EHBP-encrypted body, forwarded to the model server. Plaintext requests must be refused.
- A bearer-token check and TLS if it is reachable over a network you do not fully control.
- When it forwards to the model server, it must **rewrite the `Host` header to the server's own address**. Servers such as
  Ollama answer `403` (empty body) to any non-local `Host`, and the client's `Host` is the gateway's public address. Testing
  through a local tunnel hides this, because the host there is `localhost`.

The binding construction in `python-lib/confidentialllm/binding.py` must match the gateway byte for byte.

## Deploying a preset

- Preset values: *Gateway URL*, *Gateway access token*, *Trusted gateway certificate* (when self-signed), and *Pinned TDX
  measurements* (from the macro).
- DSS needs outbound internet to Intel's collateral service and NVIDIA's key service (attestation checks), and a code env built
  from `code-env/python/spec/requirements.txt`.
- Restrict network access to the gateway to the DSS host (cloud firewall), in addition to the token.
- Check a deployment without DSS: `GATEWAY_URL=https://host:8443 GATEWAY_TOKEN=... CA_FILE=gw.crt PINS_FILE=pins.json python3 scripts/live_check.py`.
  The token is never printed.
- A self-signed gateway certificate is tied to the host's IP; if the address changes, regenerate it and update the preset.

## Operating it: pins, reboots and kernel updates

Pins fingerprint how the VM booted. If the VM boots a different kernel or firmware, its measurements change and the plugin
**refuses to send anything** (DSS users see failing calls). That is the intended behavior, so plan for it:

- **A running kernel does not change without a reboot.** Ubuntu's unattended upgrades install new kernels but do not reboot by
  default. A reboot into a new kernel (or a host maintenance event that restarts the VM) is what changes the boot-chain
  measurements (`rtmr1`, `rtmr2`); firmware identity (`mr_td`, `rtmr0`) changes only when the cloud provider updates its firmware.
  This is expected rather than verified on a given machine: always check with the macro.
- **To avoid surprise changes:** keep automatic reboots off and hold the kernel packages on the VM
  (`apt-mark hold` the `linux-image-*` and matching NVIDIA module packages together; they must stay in step).
- **To rotate planned changes without downtime:** boot or update the VM, run the macro, review the new values, then ADD them to the
  preset as a second pin set (the setting accepts a list, any-of), and remove the old one once nothing uses it.
- **If calls start failing after a restart:** run the macro. It shows the new measurements even when they do not match the pins,
  and tells you which field differs, so you can judge whether the change is expected before re-pinning.
- The gateway and the model server must also be started again after any reboot (they are not services yet).

## Current limits

Text completion and streaming only. Tool calling, images and embeddings are rejected with an explicit error.
`topK` is ignored. The gateway this plugin talks to is **not part of this repository** (see "The gateway" below).

## Development

```
make unit-tests   # 110+ tests; no network or DSS needed
make plugin       # build dist/dss-plugin-confidential-llm-<version>.zip (needs a git commit)
make dev          # build from the working tree
```

`python-llms/confidential-llm/llm.py` and the macro need a DSS instance (`dataiku` package) and are not unit-tested.
