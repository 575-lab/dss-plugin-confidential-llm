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

import dataclasses
import html
import json

from dataiku.runnables import Runnable

from confidentialllm import attest, tdx
from confidentialllm.config import EndpointConfig
from confidentialllm.errors import ConfidentialLLMError


def _row(label, value):
    return "<tr><th style='text-align:left;padding:2px 12px 2px 0'>{}</th><td>{}</td></tr>".format(
        html.escape(label), html.escape(str(value)))


class MyRunnable(Runnable):
    def __init__(self, project_key, config, plugin_config):
        self.config = config

    def get_progress_target(self):
        return None

    def run(self, progress_callback):
        cfg = EndpointConfig.from_preset(self.config.get("endpoint"), discovery=True)
        # Attest WITHOUT the configured pins so the measurements are shown even when they changed.
        discovery_cfg = dataclasses.replace(cfg, pins=(), allow_unpinned=True)

        try:
            result = attest.attest_endpoint(discovery_cfg)
        except ConfidentialLLMError as exc:
            return "<h3 style='color:#b00020'>Attestation failed</h3><p>{}</p><p>Nothing should be sent to this endpoint.</p>".format(
                html.escape(str(exc)))

        info = result.info
        measurements = info["measurements"]
        proposed = {key: measurements[key] for key in ("mr_td", "rtmr0", "rtmr1", "rtmr2")}

        if cfg.pins:
            try:
                tdx.check_pins(measurements, cfg.pins, False)
                pin_status = "<b style='color:#1b7f3b'>Matches the pinned measurements.</b>"
            except ConfidentialLLMError as exc:
                pin_status = "<b style='color:#b00020'>Does NOT match the pinned measurements:</b> {}".format(html.escape(str(exc)))
        else:
            pin_status = "<b style='color:#b45f06'>No pins configured.</b> Review the values below, then paste them into the preset."

        rows = [
            _row("Intel TCB status", info["tcb_status"]),
            _row("Gateway key fingerprint", info["key_fingerprint"]),
        ]
        for gpu_info in info["gpus"]:
            rows.append(_row(gpu_info["gpu"], "{} - driver {} - vbios {}".format(
                gpu_info["model"], gpu_info["driver"], gpu_info["vbios"])))

        return (
            "<h3 style='color:#1b7f3b'>Attestation verified</h3>"
            "<p>The TDX quote chains to Intel and commits to this request's nonce and the gateway's key; "
            "the GPU tokens are signed by NVIDIA and bound to the same nonce.</p>"
            "<table>{rows}</table><h4>Pin status</h4><p>{pin_status}</p>"
            "<h4>Measurements to pin (review before trusting)</h4><pre>{pins}</pre>"
            "<p><i>These prove a genuine TDX guest with a valid GPU and a specific boot identity. They do not "
            "cover the model server or weights, and anyone with administrative access to the VM can read its memory.</i></p>"
        ).format(rows="".join(rows), pin_status=pin_status, pins=html.escape(json.dumps(proposed, indent=2)))
