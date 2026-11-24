"""Build synthetic GCE vTPM evidence (CA chain, AK certificate, event log, TPM2 quote) for tests.

Everything is generated locally: a fake "Google" root and intermediate, an ECDSA P-256 AK, and a quote over
SHA-256 PCRs 0-15. The layouts follow the TPM 2.0 spec and the TCG PC Client firmware profile, so the same
code paths run as for a real GCE quote.
"""

import base64
import datetime
import hashlib
import struct

from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives.asymmetric.utils import decode_dss_signature
from cryptography.x509.oid import NameOID

from confidentialllm import vtpm

QUOTED_PCRS = tuple(range(16))
NOW = datetime.datetime(2026, 11, 23, tzinfo=datetime.timezone.utc)


def _name(cn):
    return x509.Name([x509.NameAttribute(NameOID.ORGANIZATION_NAME, "Test Google"), x509.NameAttribute(NameOID.COMMON_NAME, cn)])


def _cert(subject, issuer, public_key, signing_key, ca, extensions=(), days=3650, not_before=None):
    not_before = not_before or NOW - datetime.timedelta(days=1)
    builder = (x509.CertificateBuilder().subject_name(subject).issuer_name(issuer).public_key(public_key)
               .serial_number(x509.random_serial_number()).not_valid_before(not_before)
               .not_valid_after(not_before + datetime.timedelta(days=days))
               .add_extension(x509.BasicConstraints(ca=ca, path_length=None), critical=True))
    for ext in extensions:
        builder = builder.add_extension(ext, critical=False)
    return builder.sign(signing_key, hashes.SHA256())


def _der(tag, value):
    n = len(value)
    length = bytes([n]) if n < 0x80 else bytes([0x80 | ((n.bit_length() + 7) // 8)]) + n.to_bytes((n.bit_length() + 7) // 8, "big")
    return bytes([tag]) + length + value


def _der_int(v):
    return _der(0x02, v.to_bytes(max(1, (v.bit_length() + 8) // 8), "big", signed=True))


def instance_info_der(zone="us-central1-b", project_number=1234567890, project_id="test-project",
                      instance_id=987654321012345678, instance_name="g4-test", is_production=True):
    props = _der(0x30, _der(0xA0, _der_int(1)) + _der(0xA1, _der(0x01, b"\xff" if is_production else b"\x00")))
    return _der(0x30, _der(0x0C, zone.encode()) + _der_int(project_number) + _der(0x0C, project_id.encode())
                + _der_int(instance_id) + _der(0x0C, instance_name.encode()) + props)


class CA:
    def __init__(self):
        self.root_key = ec.generate_private_key(ec.SECP384R1())
        self.root = _cert(_name("EK/AK CA Root"), _name("EK/AK CA Root"), self.root_key.public_key(), self.root_key, True)
        self.int_key = ec.generate_private_key(ec.SECP384R1())
        self.intermediate = _cert(_name("EK/AK CA Intermediate"), _name("EK/AK CA Root"),
                                  self.int_key.public_key(), self.root_key, True)

    def ak_cert(self, ak_public, instance=None, **kw):
        ext = [x509.UnrecognizedExtension(x509.ObjectIdentifier(vtpm.GCE_INSTANCE_INFO_OID), instance)] if instance else []
        return _cert(x509.Name([]), _name("EK/AK CA Intermediate"), ak_public, self.int_key, False, ext, **kw)


# --- event log ---
def _spec_id_event():
    body = vtpm.SPEC_ID_SIGNATURE + struct.pack("<IBBBB", 0, 0, 2, 0, 2) + struct.pack("<I", 2)
    body += struct.pack("<HH", vtpm.TPM_ALG_SHA1, 20) + struct.pack("<HH", vtpm.TPM_ALG_SHA256, 32) + b"\x00"
    return struct.pack("<II", 0, vtpm.EV_NO_ACTION) + bytes(20) + struct.pack("<I", len(body)) + body


def _event2(pcr, event_type, data, digest=None):
    digest = hashlib.sha256(data).digest() if digest is None else digest
    out = struct.pack("<III", pcr, event_type, 2)
    out += struct.pack("<H", vtpm.TPM_ALG_SHA1) + hashlib.sha1(data).digest()
    out += struct.pack("<H", vtpm.TPM_ALG_SHA256) + digest
    return out + struct.pack("<I", len(data)) + data


def secure_boot_data(enabled=True):
    name = "SecureBoot".encode("utf-16-le")
    value = b"\x01" if enabled else b"\x00"
    return vtpm.EFI_GLOBAL_VARIABLE_GUID + struct.pack("<QQ", len(name) // 2, len(value)) + name + value


def default_events(technology=1, secure_boot=True):
    """(pcr, type, data, digest_override) tuples for a minimal GCE-like boot."""
    sep = b"\x00\x00\x00\x00"
    return [
        (0, vtpm.EV_NO_ACTION, vtpm.STARTUP_LOCALITY_SIGNATURE + b"\x00", None),
        (0, 0x00000008, "GCE Virtual Firmware v2\x00".encode("utf-16-le"), None),  # EV_S_CRTM_VERSION
        (0, vtpm.EV_NONHOST_INFO, vtpm.GCE_NONHOST_SIGNATURE + bytes([technology]) + bytes(15), None),
        (7, vtpm.EV_EFI_VARIABLE_DRIVER_CONFIG, secure_boot_data(secure_boot), None),
        (0, vtpm.EV_SEPARATOR, sep, None),
        (7, vtpm.EV_SEPARATOR, sep, None),
        (4, 0x80000003, b"shim+grub+kernel", None),  # EV_EFI_BOOT_SERVICES_APPLICATION (digest is of the image)
        (8, 0x0000000D, b"grub_cmd linux /vmlinuz", None),
        (9, 0x0000000D, b"/boot/vmlinuz", None),
    ]


def build_event_log(events):
    return _spec_id_event() + b"".join(_event2(p, t, d, g) for p, t, d, g in events)


def replay_all(events):
    pcrs = {i: bytes(32) for i in QUOTED_PCRS}
    for pcr, event_type, data, digest in events:
        if event_type == vtpm.EV_NO_ACTION:
            continue
        digest = hashlib.sha256(data).digest() if digest is None else digest
        pcrs[pcr] = hashlib.sha256(pcrs[pcr] + digest).digest()
    return pcrs


# --- quote ---
def build_quote(extra_data, pcrs, indices=QUOTED_PCRS, alg=vtpm.TPM_ALG_SHA256):
    bitmap = bytearray(3)
    for i in indices:
        bitmap[i // 8] |= 1 << (i % 8)
    pcr_digest = hashlib.sha256(b"".join(pcrs[i] for i in sorted(indices))).digest()
    out = struct.pack(">IH", vtpm.TPM_GENERATED_VALUE, vtpm.TPM_ST_ATTEST_QUOTE)
    out += struct.pack(">H", 34) + b"\x00\x0b" + bytes(32)  # qualifiedSigner
    out += struct.pack(">H", len(extra_data)) + extra_data
    out += struct.pack(">QIIB", 1, 2, 3, 1) + struct.pack(">Q", 0x2025)
    out += struct.pack(">IHB", 1, alg, 3) + bytes(bitmap)
    return out + struct.pack(">H", 32) + pcr_digest


def sign_quote(ak_key, quote):
    r, s = decode_dss_signature(ak_key.sign(quote, ec.ECDSA(hashes.SHA256())))
    rb, sb = r.to_bytes(32, "big"), s.to_bytes(32, "big")
    return struct.pack(">HH", vtpm.TPM_ALG_ECDSA, vtpm.TPM_ALG_SHA256) + struct.pack(">H", 32) + rb + struct.pack(">H", 32) + sb


def b64(data):
    return base64.b64encode(data).decode()


def make_evidence(digest, ca=None, technology=1, secure_boot=True, events=None, instance=b"default",
                  send_intermediate=True, ak_key=None, pcr_tweak=None, quote_extra=None):
    """Return (evidence, ca). ``pcr_tweak`` = {index: bytes} changes the PCR values *after* the quote is built."""
    ca = ca or CA()
    ak_key = ak_key or ec.generate_private_key(ec.SECP256R1())
    events = events if events is not None else default_events(technology, secure_boot)
    pcrs = replay_all(events)
    quote = build_quote(digest if quote_extra is None else quote_extra, pcrs)
    if pcr_tweak:
        pcrs.update(pcr_tweak)
    instance_der = instance_info_der() if instance == b"default" else instance
    ak_cert = ca.ak_cert(ak_key.public_key(), instance_der)
    evidence = {
        "ak_cert_der_b64": b64(ak_cert.public_bytes(serialization.Encoding.DER)),
        "intermediates_der_b64": [b64(ca.intermediate.public_bytes(serialization.Encoding.DER))] if send_intermediate else [],
        "quote_b64": b64(quote),
        "signature_b64": b64(sign_quote(ak_key, quote)),
        "pcr_bank": "sha256",
        "pcrs": {str(i): v.hex() for i, v in pcrs.items()},
        "event_log_b64": b64(build_event_log(events)),
    }
    return evidence, ca
