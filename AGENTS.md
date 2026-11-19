# AGENTS.md

## Purpose

DSS plugin that exposes a model running in an attested Intel TDX + NVIDIA GPU confidential VM as an LLM Mesh model.
The security property is: no prompt leaves DSS unless attestation succeeded, and prompts are encrypted to the exact
key that was attested.

## Structure rules

- Components stay thin; logic lives in `python-lib/confidentialllm/` and must be unit-testable without DSS or network.
  Keep pure checks (`tdx.check_*`, `gpu.check_*`, `config`) separate from I/O (`attest.fetch_evidence`, `tdx.verify_quote`).
- `binding.compute_digest` must stay byte-identical to `binding()` in the gateway's `main.go`. Changing one means changing both,
  and bumping the label (`ehbp-attest-v1`).
- Fail closed. Never add a code path that sends a request when attestation, pinning or token verification failed, and never
  downgrade to plaintext. `allow_unpinned` is an explicit opt-in; the macro uses `discovery=True` only to display measurements.
- Encrypt to `AttestationResult.key_config` only. Never re-fetch the key for a request (that would reopen a time-of-check gap).
- Do not put prompts, answers or the gateway token in trace attributes or logs.

## Testing

- `make unit-tests`. Every trust check needs a negative test (wrong nonce, wrong key, wrong pin, wrong signature, expired token, bad claim).
- GPU tests sign real ES384 tokens with a throwaway key and inject the key resolver; do not stub `jwt.decode`.
- `llm.py` and `runnable.py` import `dataiku` and are only exercised inside DSS.

## Known unverified points

- The `BaseLLM` streaming contract (`aprocess_stream` yielding `{"chunk": {"text": ...}}`) follows Dataiku's agent docs; confirm on a real DSS.
- DSS-side behavior of the `PASSWORD` param type in a parameter set and of `PRESET` access from a macro has not been run on a live instance.
