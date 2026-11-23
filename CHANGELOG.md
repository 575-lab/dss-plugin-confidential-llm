# Changelog

## 0.1.4-demo.1 (demo fork)

- DEMO mode: a "DEMO trust root" preset field accepts simulated attestation from `spark-demo/demo_gateway.py`, for machines without TDX/H100 (e.g. a DGX Spark). Production behavior is unchanged when the field is empty.
- `spark-demo/`: demo gateway (real EHBP encryption, simulated evidence, tamper switches), CLI check, mock model server.

## Version 0.1.0 - Unreleased
- Confidential LLM custom LLM connection (completion and streaming), attestation-gated and body-encrypted.
- "Attest confidential endpoint" macro for reviewing and pinning TDX measurements.
