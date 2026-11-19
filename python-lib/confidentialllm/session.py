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

import logging
import threading
import time

from . import attest

LOGGER = logging.getLogger(__name__)

_SESSIONS = {}
_SESSIONS_LOCK = threading.Lock()


class AttestedSession:
    """Caches a successful attestation for ``ttl_seconds``.

    Attesting takes seconds and calls Intel and NVIDIA services, so it is done once per
    process and reused. Requests are always encrypted to the exact key that was attested; if the
    gateway's key changes, decryption fails and the caller invalidates the session to force a new
    attestation. Concurrent callers share one in-flight attestation.
    """

    def __init__(self, config, attest_fn=None, clock=time.monotonic):
        self._config = config
        self._attest_fn = attest_fn or attest.attest_endpoint
        self._clock = clock
        self._lock = threading.Lock()
        self._result = None
        self._expires_at = 0.0

    @property
    def config(self):
        return self._config

    def current(self):
        """Return a still-valid AttestationResult, attesting first if there is none."""
        with self._lock:
            if self._result is None or self._clock() >= self._expires_at:
                LOGGER.info("Attesting confidential endpoint %s", self._config.base_url)
                self._result = self._attest_fn(self._config)
                self._expires_at = self._clock() + self._config.ttl_seconds
                LOGGER.info("Attestation succeeded (key %s)", self._result.info.get("key_fingerprint"))
            return self._result

    def invalidate(self, failed_key_config=None):
        """Drop the cached attestation.

        When ``failed_key_config`` is given, only drop it if it is still the cached one, so a late
        failure of an old key cannot discard a newer, good attestation made by another thread.
        """
        with self._lock:
            if failed_key_config is None or (self._result and self._result.key_config == failed_key_config):
                self._result = None


def get_session(config):
    with _SESSIONS_LOCK:
        session = _SESSIONS.get(config.cache_key)
        if session is None:
            session = _SESSIONS[config.cache_key] = AttestedSession(config)
        return session
