import datetime
import http.server
import ipaddress
import ssl
import threading

import httpx
import pytest
from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.x509.oid import NameOID

from confidentialllm.config import EndpointConfig, tls_verify
from confidentialllm.errors import ConfigError

H = "ab" * 48


def make_cert(tmp_path, name):
    """A self-signed certificate usable as its own trust anchor, for 127.0.0.1 and localhost."""
    key = ec.generate_private_key(ec.SECP256R1())
    subject = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, name)])
    now = datetime.datetime.now(datetime.timezone.utc)
    cert = (
        x509.CertificateBuilder().subject_name(subject).issuer_name(subject).public_key(key.public_key())
        .serial_number(x509.random_serial_number()).not_valid_before(now - datetime.timedelta(minutes=1))
        .not_valid_after(now + datetime.timedelta(days=1))
        .add_extension(x509.BasicConstraints(ca=True, path_length=None), critical=True)
        .add_extension(x509.KeyUsage(True, False, False, False, False, True, False, False, False), critical=True)
        .add_extension(x509.SubjectAlternativeName([x509.IPAddress(ipaddress.ip_address("127.0.0.1")), x509.DNSName("localhost")]), critical=False)
        .add_extension(x509.SubjectKeyIdentifier.from_public_key(key.public_key()), critical=False)
        .sign(key, hashes.SHA256())
    )
    cert_path, key_path = tmp_path / (name + ".pem"), tmp_path / (name + ".key")
    cert_path.write_bytes(cert.public_bytes(serialization.Encoding.PEM))
    key_path.write_bytes(key.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8, serialization.NoEncryption()))
    return cert_path, key_path


@pytest.fixture
def tls_server(tmp_path):
    cert_path, key_path = make_cert(tmp_path, "gateway")

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200); self.end_headers(); self.wfile.write(b"ok")

        def log_message(self, *args):
            pass

    server = http.server.HTTPServer(("127.0.0.1", 0), Handler)
    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    ctx.load_cert_chain(cert_path, key_path)
    server.socket = ctx.wrap_socket(server.socket, server_side=True)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield "https://127.0.0.1:%d" % server.server_address[1], cert_path.read_text(), tmp_path
    server.shutdown()


def cfg(url, **extra):
    return EndpointConfig.from_preset({"gateway_url": url, "pins": '{"mr_td": "%s"}' % H, **extra})


def test_pinned_certificate_is_accepted(tls_server):
    url, pem, _ = tls_server
    assert httpx.get(url, verify=tls_verify(cfg(url, ca_certificate=pem))).text == "ok"


def test_a_different_certificate_is_rejected(tls_server):
    url, _, tmp_path = tls_server
    other_pem = make_cert(tmp_path, "attacker")[0].read_text()
    with pytest.raises(httpx.ConnectError):
        httpx.get(url, verify=tls_verify(cfg(url, ca_certificate=other_pem)))


def test_self_signed_is_rejected_without_a_configured_certificate(tls_server):
    url, _, _ = tls_server
    with pytest.raises(httpx.ConnectError):
        httpx.get(url, verify=tls_verify(cfg(url)))


def test_only_the_configured_certificate_is_trusted_not_the_system_store():
    ctx = tls_verify(cfg("https://gw.example.com", ca_certificate=_throwaway_pem()))
    assert len(ctx.get_ca_certs()) == 1


def _throwaway_pem():
    import pathlib, tempfile
    return make_cert(pathlib.Path(tempfile.mkdtemp()), "t")[0].read_text()


def test_invalid_certificate_text_is_rejected_at_config_time():
    with pytest.raises(ConfigError, match="valid PEM"):
        cfg("https://gw", ca_certificate="-----BEGIN CERTIFICATE-----\nnot a cert\n-----END CERTIFICATE-----")


def test_certificate_requires_https():
    with pytest.raises(ConfigError, match="https"):
        cfg("http://127.0.0.1:8443", ca_certificate=_throwaway_pem())


@pytest.mark.parametrize("url", ["http://203.0.113.10:8443", "http://gw.example.com", "http://10.0.0.5:8443"])
def test_token_is_never_sent_over_plain_http_to_a_remote_host(url):
    with pytest.raises(ConfigError, match="plain http"):
        cfg(url, gateway_token="s3cret")


@pytest.mark.parametrize("url", ["http://127.0.0.1:8443", "http://localhost:8443", "http://[::1]:8443"])
def test_token_over_http_is_allowed_to_loopback_only(url):
    assert cfg(url, gateway_token="s3cret").gateway_token == "s3cret"


def test_token_over_https_is_allowed():
    assert cfg("https://203.0.113.10:8443", gateway_token="s3cret", ca_certificate=_throwaway_pem()).gateway_token == "s3cret"


def test_http_without_token_to_a_remote_host_is_still_allowed_for_testing():
    assert cfg("http://10.0.0.5:8443").gateway_token is None
