import hashlib

import pytest

from confidentialllm import binding, tdx
from confidentialllm.errors import AttestationError

H = "ab" * 48
H2 = "cd" * 48
MEAS = {"mr_td": H, "rtmr0": H2, "rtmr1": H, "rtmr2": H2, "rtmr3": "00" * 48}


def test_digest_matches_documented_construction():
    nonce, cfg = b"\x01" * 32, b"key-config"
    expected = hashlib.sha256(b"ehbp-attest-v1\x00" + nonce + cfg).digest()
    assert binding.compute_digest(nonce, cfg) == expected


def test_digest_depends_on_nonce_and_key():
    n, k = b"\x01" * 32, b"k"
    assert binding.compute_digest(n, k) != binding.compute_digest(b"\x02" * 32, k)
    assert binding.compute_digest(n, k) != binding.compute_digest(n, b"other")


def test_digest_rejects_wrong_nonce_size():
    with pytest.raises(ValueError):
        binding.compute_digest(b"short", b"k")


def test_report_data_must_commit_to_digest_with_zero_tail():
    digest = binding.compute_digest(b"\x01" * 32, b"k")
    tdx.check_report_data(digest + b"\x00" * 32, digest)
    with pytest.raises(AttestationError):
        tdx.check_report_data(b"\x00" * 64, digest)  # replay of a different nonce/key
    with pytest.raises(AttestationError):
        tdx.check_report_data(digest + b"\x00" * 31 + b"\x01", digest)  # junk in the tail
    with pytest.raises(AttestationError):
        tdx.check_report_data(digest, digest)  # wrong length


def test_pins_match_any_of():
    pins = ({"mr_td": H2}, {"mr_td": H, "rtmr0": H2})
    assert tdx.check_pins(MEAS, pins, False) is True


def test_pins_only_compare_fields_that_are_pinned():
    assert tdx.check_pins(MEAS, ({"mr_td": H},), False) is True


def test_pin_mismatch_is_rejected_and_names_the_field():
    with pytest.raises(AttestationError, match="rtmr0"):
        tdx.check_pins(MEAS, ({"mr_td": H, "rtmr0": H},), False)


def test_no_pins_rejected_unless_unpinned_allowed():
    with pytest.raises(AttestationError):
        tdx.check_pins(MEAS, (), False)
    assert tdx.check_pins(MEAS, (), True) is False
