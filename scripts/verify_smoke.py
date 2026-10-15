#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Validate the optional one-update Ravens/Transporter training check.
"""Validate finite one-step losses and reload both generated network weights."""
import json
import math
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]


def main():
    import tensorflow as tf
    from ravens.agents.transporter import OriginalTransporterAgent

    lines = re.findall(r"Train Iter: (\d+) Loss: (\S+) (\S+)",
                       (ROOT / "logs/train_smoke.log").read_text(encoding="utf-8"))
    if len(lines) != 1 or lines[0][0] != "1":
        raise ValueError("Expected exactly one logged optimizer step per network")
    losses = [float(value) for value in lines[0][1:]]
    if not all(math.isfinite(value) for value in losses):
        raise ValueError("Non-finite training loss")
    tf.keras.backend.clear_session()
    agent = OriginalTransporterAgent("block-insertion-transporter-10-0",
                                     "block-insertion", str(ROOT / "smoke_run"))
    agent.load(1)
    result = {"optimizer_steps_per_network": 1, "attention_loss_rounded": losses[0],
              "transport_loss_rounded": losses[1], "saved_weights_reload": "passed",
              "device": "CPU", "purpose": "Pipeline check, not a trained reproduction model"}
    (ROOT / "results/train_smoke.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
