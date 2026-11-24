"""Cross-check: evidence produced by the real gateway code (go-tpm-tools against the TPM 2.0 reference simulator)
verifies with the plugin's Python verifier. Regenerate the fixture from confidential-gateway/gateway with
VTPM_FIXTURE_OUT=<this dir>/data/vtpm_simulator_evidence.json go test -run TestVTPMQuoteProducesVerifiableEvidence ."""

import copy
import datetime
import json
import os

import pytest

from confidentialllm import vtpm
from confidentialllm.errors import AttestationError

FIXTURE = json.load(open(os.path.join(os.path.dirname(__file__), "data", "vtpm_simulator_evidence.json")))
ROOT = vtpm._load_pem(FIXTURE["root_pem"])
DIGEST = bytes.fromhex(FIXTURE["digest_hex"])
NOW = datetime.datetime.fromisoformat(FIXTURE["not_after"].replace("Z", "+00:00")) - datetime.timedelta(hours=2)


def run(evidence=None, digest=DIGEST):
    return vtpm.verify_evidence(evidence or FIXTURE["vtpm"], digest, roots=[ROOT], intermediates=[], now=NOW)


def test_gateway_evidence_verifies_end_to_end():
    measurements, platform = run()
    assert platform == {"technology": "AMD_SEV", "secure_boot": True, "instance": None}
    assert measurements["pcr4"] == "00" * 32  # untouched in the simulator
    assert measurements["pcr0"] != "00" * 32 and measurements["pcr7"] != "00" * 32


def test_gateway_quote_is_a_multi_pcr_sha256_quote():
    import base64
    quote = vtpm.parse_quote(base64.b64decode(FIXTURE["vtpm"]["quote_b64"]))
    assert quote.extra_data == DIGEST
    assert quote.selection[0][0] == vtpm.TPM_ALG_SHA256 and set(range(16)) <= set(quote.selection[0][1])


def test_gateway_evidence_rejected_for_another_digest():
    with pytest.raises(AttestationError, match="commit"):
        run(digest=bytes(32))


def test_gateway_evidence_with_a_changed_pcr_is_rejected():
    evidence = copy.deepcopy(FIXTURE["vtpm"])
    evidence["pcrs"]["4"] = "11" * 32
    with pytest.raises(AttestationError, match="PCR digest"):
        run(evidence)
