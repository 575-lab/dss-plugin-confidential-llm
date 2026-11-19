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


class ConfidentialLLMError(Exception):
    """Base class for all errors raised by this plugin."""


class ConfigError(ConfidentialLLMError, ValueError):
    """The endpoint settings are invalid or unsafe."""


class AttestationError(ConfidentialLLMError):
    """Attestation evidence is missing, invalid, stale, or does not match the pinned identity.

    No prompt is ever sent when this is raised.
    """


class GatewayError(ConfidentialLLMError):
    """The gateway answered with an error after attestation succeeded."""

    def __init__(self, message, status_code=None):
        super().__init__(message)
        self.status_code = status_code


class UnsupportedRequestError(ConfidentialLLMError, ValueError):
    """The DSS request uses a feature this connection does not implement."""
