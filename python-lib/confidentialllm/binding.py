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

"""The value that ties attestation evidence to this request's nonce and the server's HPKE key.

The gateway computes the same digest. It goes into the TDX quote's ``report_data`` (first 32 bytes,
rest zero) and into the GPU token's ``eat_nonce``. It must stay byte-for-byte identical to the
gateway implementation (``binding()`` in the gateway's ``main.go``).
"""

import hashlib

LABEL = b"ehbp-attest-v1\x00"
NONCE_SIZE = 32


def compute_digest(nonce, key_config):
    if len(nonce) != NONCE_SIZE:
        raise ValueError("nonce must be {} bytes".format(NONCE_SIZE))
    return hashlib.sha256(LABEL + nonce + key_config).digest()
