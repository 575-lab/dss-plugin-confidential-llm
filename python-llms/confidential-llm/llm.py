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

import asyncio
import logging
import time

from dataiku.llm.python import BaseLLM

from confidentialllm import client, messages
from confidentialllm.config import EndpointConfig
from confidentialllm.session import get_session

LOGGER = logging.getLogger(__name__)


class ConfidentialLLM(BaseLLM):
    """Chat completion served inside an attested confidential VM.

    Thin adapter: all logic lives in python-lib/confidentialllm. No prompt is sent before attestation
    succeeds; prompts and answers travel encrypted to the attested key.
    """

    def set_config(self, config, plugin_config):
        self.model = config.get("model")
        if not self.model:
            raise ValueError("The model name is required.")
        # Fails here (not at first request) if pins are missing or the URL is unsafe.
        self.endpoint_config = EndpointConfig.from_preset(config.get("endpoint"))
        self.session = get_session(self.endpoint_config)

    def _record_attestation(self, span, result):
        info = result.info
        span.attributes["attestation_pinned"] = result.pinned
        span.attributes["attestation_tcb_status"] = info.get("tcb_status")
        span.attributes["attestation_key_fingerprint"] = info.get("key_fingerprint")
        span.attributes["attestation_mr_td"] = info.get("measurements", {}).get("mr_td")
        span.attributes["attestation_age_seconds"] = round(time.time() - result.attested_at, 1)
        span.attributes["attestation_gpus"] = info.get("gpus")

    def process(self, query, settings, trace):
        with trace.subspan("Confidential LLM request") as span:
            body = messages.build_request(self.model, query, settings, stream=False)
            self._record_attestation(span, self.session.current())
            response = client.complete(self.session, body)
            answer = messages.to_dss_response(response)
            span.attributes["completion_tokens"] = answer.get("completionTokens")
            return answer

    async def aprocess_stream(self, query, settings, trace):
        loop = asyncio.get_running_loop()
        with trace.subspan("Confidential LLM streamed request") as span:
            body = messages.build_request(self.model, query, settings, stream=True)
            # Attestation blocks for seconds: keep it off the event loop.
            self._record_attestation(span, await loop.run_in_executor(None, self.session.current))

            events = client.stream(self.session, body)
            done = object()
            finish_reason = None
            saw_tool_calls = False
            try:
                while True:
                    event = await loop.run_in_executor(None, next, events, done)
                    if event is done:
                        break
                    chunk, finish = messages.stream_event_to_dss(event)
                    if chunk:
                        saw_tool_calls = saw_tool_calls or bool(chunk.get("toolCalls"))
                        yield {"chunk": chunk}
                    if finish:
                        finish_reason = finish
            finally:
                await loop.run_in_executor(None, events.close)
            if saw_tool_calls and finish_reason in (None, "stop", "unknown"):
                finish_reason = "tool_calls"
            yield {"footer": {"finishReason": finish_reason or "stop"}}
