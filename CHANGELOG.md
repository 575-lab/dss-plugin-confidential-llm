# Changelog

## Version 0.2.0 - Unreleased
- Support GCP Confidential VMs with AMD SEV (for example G4 with an NVIDIA RTX PRO 6000), attested through Google's vTPM:
  AK certificate chain to Google's EK/AK CA, TPM2 quote signature and nonce binding, event log replay, Confidential VM
  technology and Secure Boot checks, and PCR pins (pcr0, pcr4, pcr7, pcr8, pcr9, optional instance_id).
- New preset field "Confidential VM type" (Intel TDX by default, so existing presets are unchanged). The TEE type comes
  from the preset, never from the gateway, so a gateway cannot downgrade the check.
- New optional preset field "Allowed GPU models" (prefixes of NVIDIA's hwmodel claim).
- The macro shows the measurements, platform details and trust caveats for the selected TEE type.

## Version 0.1.0 - Unreleased
- Confidential LLM custom LLM connection (completion and streaming), attestation-gated and body-encrypted.
- "Attest confidential endpoint" macro for reviewing and pinning TDX measurements.
