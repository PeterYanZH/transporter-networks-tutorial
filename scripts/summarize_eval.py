#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Summarize a locally generated Ravens evaluation; never open untrusted pickle.
"""Convert the current upstream evaluation result into a readable JSON summary."""
import json
import math
from pathlib import Path
import pickle

ROOT = Path(__file__).resolve().parents[1]


def main():
    assets = json.loads((ROOT / "results/assets.json").read_text(encoding="utf-8"))
    path = ROOT / "block-insertion-transporter-10-0-10000.pkl"
    # This path is written by the immediately preceding official test.py call.
    with path.open("rb") as stream:
        episodes = pickle.load(stream)
    rewards = [float(item[0]) for item in episodes]
    if len(rewards) != 3 or not all(math.isfinite(value) for value in rewards):
        raise ValueError("Expected exactly three finite episode rewards")
    result = {"model": "block-insertion-transporter-10-0", "checkpoint_step": 10000,
              "episodes": len(rewards), "seeds": assets["test_seeds"], "rewards": rewards,
              "mean_reward": sum(rewards) / len(rewards),
              "successes": sum(value >= 0.99 for value in rewards),
              "device": "CPU", "scope": "3-episode pipeline check, not full paper benchmark"}
    (ROOT / "results/eval_pretrained.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
