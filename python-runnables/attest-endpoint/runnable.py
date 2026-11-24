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

from confidentialllm import attest, pins
from confidentialllm.config import PIN_KEYS, TEE_GCE_VTPM, TEE_LABELS, EndpointConfig
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
        is_vtpm = cfg.tee_type == TEE_GCE_VTPM
        label = "vTPM" if is_vtpm else "TDX"
        proposed = {key: measurements[key] for key in (("pcr0", "pcr4", "pcr7", "pcr8", "pcr9") if is_vtpm else PIN_KEYS)}

        if cfg.pins:
            try:
                pins.check_pins(measurements, cfg.pins, False, label)
                pin_status = "<b style='color:#1b7f3b'>Matches the pinned measurements.</b>"
            except ConfidentialLLMError as exc:
                pin_status = "<b style='color:#b00020'>Does NOT match the pinned measurements:</b> {}".format(html.escape(str(exc)))
        else:
            pin_status = "<b style='color:#b45f06'>No pins configured.</b> Review the values below, then paste them into the preset."

        rows = [_row("TEE type", TEE_LABELS[cfg.tee_type])]
        if is_vtpm:
            platform = info["platform"]
            rows.append(_row("Confidential VM technology", platform["technology"]))
            rows.append(_row("Secure Boot", "on" if platform["secure_boot"] else "off"))
            instance = platform.get("instance")
            if instance:
                rows.append(_row("GCE instance", "{} (id {}), zone {}, project {}".format(
                    instance["instance_name"], instance["instance_id"], instance["zone"], instance["project_id"])))
        else:
            rows.append(_row("Intel TCB status", info["tcb_status"]))
        rows.append(_row("Gateway key fingerprint", info["key_fingerprint"]))
        for gpu_info in info["gpus"]:
            rows.append(_row(gpu_info["gpu"], "{} - driver {} - vbios {}".format(
                gpu_info["model"], gpu_info["driver"], gpu_info["vbios"])))

        if is_vtpm:
            proof = ("The vTPM quote is signed by an attestation key certified by Google's EK/AK CA, commits to this "
                     "request's nonce and the gateway's key, and its event log shows a Confidential VM with Secure Boot; "
                     "the GPU tokens are signed by NVIDIA and bound to the same nonce.")
            extra = ""
            if "instance_id" in measurements:
                extra = ("<p>Optional: add <code>\"instance_id\": \"{}\"</code> to bind trust to this one VM "
                         "(re-pin after recreating it).</p>").format(html.escape(measurements["instance_id"]))
            caveat = ("These prove a Google Confidential VM (AMD SEV) with a valid GPU and a specific boot identity, as "
                      "reported by Google's vTPM. The root of trust is Google's platform, not the CPU vendor: plain SEV has "
                      "no hardware attestation report and no memory integrity protection. They do not cover the model "
                      "server or weights, and anyone with administrative access to the VM can read its memory.")
        else:
            proof = ("The TDX quote chains to Intel and commits to this request's nonce and the gateway's key; "
                     "the GPU tokens are signed by NVIDIA and bound to the same nonce.")
            extra = ""
            caveat = ("These prove a genuine TDX guest with a valid GPU and a specific boot identity. They do not "
                      "cover the model server or weights, and anyone with administrative access to the VM can read its memory.")

        return (
            "<h3 style='color:#1b7f3b'>Attestation verified</h3>"
            "<p>{proof}</p>"
            "<table>{rows}</table><h4>Pin status</h4><p>{pin_status}</p>"
            "<h4>Measurements to pin (review before trusting)</h4><pre>{pins}</pre>{extra}"
            "<p><i>{caveat}</i></p>"
        ).format(proof=html.escape(proof), rows="".join(rows), pin_status=pin_status,
                 pins=html.escape(json.dumps(proposed, indent=2)), extra=extra, caveat=html.escape(caveat))
