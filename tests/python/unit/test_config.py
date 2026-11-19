import json

import pytest

from confidentialllm.config import EndpointConfig, parse_pins
from confidentialllm.errors import ConfigError

H = "ab" * 48  # 96 hex chars
H2 = "cd" * 48


def preset(**overrides):
    base = {"gateway_url": "https://gw.example.com:8443", "pins": json.dumps({"mr_td": H})}
    base.update(overrides)
    return base


def test_valid_preset():
    cfg = EndpointConfig.from_preset(preset())
    assert cfg.base_url == "https://gw.example.com:8443"
    assert cfg.pins == ({"mr_td": H},)
    assert cfg.allowed_tcb == frozenset({"UpToDate"})
    assert cfg.ttl_seconds == 600
    assert cfg.timeout_seconds == 300
    assert not cfg.allow_unpinned


def test_pins_are_required_by_default():
    with pytest.raises(ConfigError, match="No pinned measurements"):
        EndpointConfig.from_preset({"gateway_url": "https://gw"})


def test_unpinned_requires_explicit_opt_in():
    cfg = EndpointConfig.from_preset({"gateway_url": "https://gw", "allow_unpinned": True})
    assert cfg.pins == () and cfg.allow_unpinned


def test_discovery_mode_allows_no_pins():
    assert EndpointConfig.from_preset({"gateway_url": "https://gw"}, discovery=True).allow_unpinned


def test_pins_accept_object_or_list_and_normalize_case():
    assert parse_pins(json.dumps({"mr_td": H.upper()})) == ({"mr_td": H},)
    assert parse_pins(json.dumps([{"mr_td": H}, {"mr_td": H2, "rtmr0": H}])) == ({"mr_td": H}, {"mr_td": H2, "rtmr0": H})


@pytest.mark.parametrize("raw", [
    "not json", "[]", "42", json.dumps({"rtmr0": H}), json.dumps({"mr_td": "abc"}),
    json.dumps({"mr_td": "zz" * 48}), json.dumps({"mr_td": H, "extra": H}), json.dumps([1]),
])
def test_bad_pins_rejected(raw):
    with pytest.raises(ConfigError):
        parse_pins(raw)


@pytest.mark.parametrize("url", [
    "", None, "gw.example.com", "ftp://gw", "https://", "https://user:pw@gw", "https://gw?x=1", "https://gw#frag",
])
def test_bad_urls_rejected(url):
    with pytest.raises(ConfigError):
        EndpointConfig.from_preset(preset(gateway_url=url))


def test_url_trailing_slash_stripped():
    assert EndpointConfig.from_preset(preset(gateway_url="http://10.0.0.5:8443/")).base_url == "http://10.0.0.5:8443"


def test_revoked_tcb_can_never_be_allowed():
    with pytest.raises(ConfigError, match="revoked"):
        EndpointConfig.from_preset(preset(allowed_tcb=["UpToDate", "Revoked"]))


def test_tcb_accepts_comma_string_and_rejects_empty():
    assert EndpointConfig.from_preset(preset(allowed_tcb="UpToDate, SWHardeningNeeded")).allowed_tcb == {"UpToDate", "SWHardeningNeeded"}


@pytest.mark.parametrize("field", ["attestation_ttl_minutes", "request_timeout_seconds"])
@pytest.mark.parametrize("value", [0, -5, "abc"])
def test_bad_numbers_rejected(field, value):
    with pytest.raises(ConfigError):
        EndpointConfig.from_preset(preset(**{field: value}))


def test_cache_key_separates_trust_settings():
    a = EndpointConfig.from_preset(preset())
    b = EndpointConfig.from_preset(preset(pins=json.dumps({"mr_td": H2})))
    c = EndpointConfig.from_preset(preset())
    assert a.cache_key != b.cache_key and a.cache_key == c.cache_key
