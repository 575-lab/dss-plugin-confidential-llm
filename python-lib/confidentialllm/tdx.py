# Copyright 2026 Dataiku SAS
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Intel TDX quote checks. The pure checks are separate from the quote verification (network)."""

import hmac

from . import pins as pins_mod
from .errors import AttestationError

TDX_TEE_TYPE = 0x81


def check_report_data(report_data, digest):
    """The quote must commit to this request's nonce AND the server's HPKE key (and nothing else)."""
    report_data = bytes(report_data)
    if len(report_data) != 64:
        raise AttestationError("Unexpected TDX report_data length: {}".format(len(report_data)))
    if not hmac.compare_digest(report_data[:32], digest) or any(report_data[32:]):
        raise AttestationError(
            "The TDX quote does not commit to this request's nonce and the gateway's encryption key. "
            "It is stale, replayed, or belongs to a different key."
        )


def check_pins(measurements, pins, allow_unpinned):
    """Return True if measurements were enforced against pins, False if unpinned mode let them through."""
    return pins_mod.check_pins(measurements, pins, allow_unpinned, "TDX")


def read_measurements(report):
    return {
        "mr_td": bytes(report.mr_td).hex(),
        "rtmr0": bytes(report.rt_mr0).hex(),
        "rtmr1": bytes(report.rt_mr1).hex(),
        "rtmr2": bytes(report.rt_mr2).hex(),
        "rtmr3": bytes(report.rt_mr3).hex(),
    }


async def verify_quote(quote, allowed_tcb):
    """Verify the quote's signature chain against Intel's collateral and return (parsed_quote, tcb_status)."""
    import dcap_qvl  # imported lazily so the pure checks above can be tested without the native wheel

    try:
        result = await dcap_qvl.get_collateral_and_verify(quote)
    except Exception as exc:  # the library raises ValueError/RuntimeError for bad quotes or network failures
        raise AttestationError("The TDX quote did not verify against Intel's collateral: {}".format(exc)) from exc

    status = str(result.status)
    if status not in allowed_tcb:
        raise AttestationError(
            "Intel TCB status {!r} is not allowed (allowed: {}; advisories: {}).".format(
                status, sorted(allowed_tcb), list(result.advisory_ids))
        )

    parsed = dcap_qvl.parse_quote(quote)
    if not parsed.is_tdx() or parsed.header.tee_type != TDX_TEE_TYPE:
        raise AttestationError("The evidence is not an Intel TDX quote.")
    return parsed, status
