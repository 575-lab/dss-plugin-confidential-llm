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

import json
import logging

import httpx
from ehbp import Client as EhbpClient
from ehbp import EHBPError

from .config import tls_verify
from .errors import GatewayError

LOGGER = logging.getLogger(__name__)
CHAT_PATH = "/v1/chat/completions"


def iter_sse_json(chunks):
    """Parse a server-sent-events byte stream into JSON events, stopping at ``[DONE]``.

    Network chunks do not align with event lines, so bytes are buffered until a full line arrives.
    """
    buffer = b""

    def parse(line):
        line = line.rstrip(b"\r")
        if not line.startswith(b"data:"):
            return None
        data = line[5:].strip()
        if not data:
            return None
        if data == b"[DONE]":
            return StopIteration
        return json.loads(data)

    for chunk in chunks:
        buffer += chunk
        while b"\n" in buffer:
            line, buffer = buffer.split(b"\n", 1)
            event = parse(line)
            if event is StopIteration:
                return
            if event is not None:
                yield event
    if buffer:
        event = parse(buffer)
        if event is not None and event is not StopIteration:
            yield event


def _ehbp_client(session_result, config):
    # Encrypt to EXACTLY the key bytes that were attested; never re-fetch the key.
    http = httpx.Client(timeout=httpx.Timeout(config.timeout_seconds, connect=15), verify=tls_verify(config))
    return EhbpClient.with_config(config.base_url, session_result.key_config, http_client=http)


def _headers(config):
    return {"Authorization": "Bearer {}".format(config.gateway_token)} if config.gateway_token else {}


def _raise_for_status(status_code, body_bytes):
    if status_code // 100 == 2:
        return
    detail = body_bytes[:300].decode("utf-8", "replace") if body_bytes else ""
    raise GatewayError("The confidential gateway returned HTTP {}: {}".format(status_code, detail), status_code)


def complete(session, body):
    """POST a (non-streamed) chat completion. Re-attests once if the gateway's key changed."""
    config = session.config
    for attempt in (1, 2):
        result = session.current()
        client = _ehbp_client(result, config)
        try:
            response = client.post(CHAT_PATH, json_body=body, headers=_headers(config))
        except EHBPError as exc:
            # A new gateway key (restart) makes decryption fail. Drop only the attestation we used, retry once.
            LOGGER.warning("Encrypted exchange failed (%s); re-attesting (attempt %d).", exc, attempt)
            session.invalidate(result.key_config)
            if attempt == 2:
                raise GatewayError("The encrypted exchange with the gateway failed: {}".format(exc)) from exc
            continue
        finally:
            client.close()
        _raise_for_status(response.status_code, response.content)
        return response.json()


def stream(session, body):
    """POST a streamed chat completion and yield parsed events. Re-attests once before any data is yielded."""
    config = session.config
    for attempt in (1, 2):
        result = session.current()
        client = _ehbp_client(result, config)
        started = False
        try:
            with client.stream("POST", CHAT_PATH, json_body=body, headers=_headers(config)) as response:
                if response.status_code // 100 != 2:
                    _raise_for_status(response.status_code, b"".join(response.iter_bytes()))
                for event in iter_sse_json(response.iter_bytes()):
                    started = True
                    yield event
            return
        except EHBPError as exc:
            session.invalidate(result.key_config)
            if started or attempt == 2:
                raise GatewayError("The encrypted exchange with the gateway failed: {}".format(exc)) from exc
            LOGGER.warning("Encrypted stream failed before any data (%s); re-attesting.", exc)
        finally:
            client.close()
