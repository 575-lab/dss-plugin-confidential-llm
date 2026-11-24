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

"""Match measurements against pinned values. Shared by every TEE type."""

import hmac

from .errors import AttestationError


def check_pins(measurements, pins, allow_unpinned, label):
    """Return True if measurements were enforced against pins, False if unpinned mode let them through.

    ``pins`` is an any-of list of validated pin sets: the machine passes if it matches at least one set
    exactly, on every field that set pins. ``label`` names the identity in error messages ("TDX", "vTPM").
    """
    if not pins:
        if allow_unpinned:
            return False
        raise AttestationError("No pinned measurements are configured; refusing to trust an unpinned endpoint.")

    mismatches = []
    for pin in pins:
        wrong = sorted(key for key, value in pin.items() if not hmac.compare_digest(
            str(value).encode(), str(measurements.get(key) or "").encode()))
        if not wrong:
            return True
        mismatches.append(wrong)
    best = min(mismatches, key=len)
    raise AttestationError(
        "The endpoint's {} identity does not match any pinned measurement (differs in: {}). "
        "The software or firmware may have changed; review and re-pin if that is expected.".format(label, ", ".join(best))
    )
