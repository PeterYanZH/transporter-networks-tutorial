#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Educational environment check for Google Research Ravens / Transporter Networks.
"""Check the pinned source, CPU dependencies, and one headless scene reset."""
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
COMMIT = "4752ffab487ebd387781ad7890aa540ae2780970"


def main():
    if sys.version_info[:2] != (3, 10):
        raise RuntimeError("This tutorial expects Python 3.10")
    commit = subprocess.check_output(
        ["git", "-C", str(ROOT / "upstream"), "rev-parse", "HEAD"], text=True).strip()
    if commit != COMMIT:
        raise RuntimeError("Upstream source differs from the pinned commit")
    import tensorflow as tf
    from ravens.environments.environment import Environment
    from ravens.tasks.block_insertion import BlockInsertion

    if tf.config.list_physical_devices("GPU"):
        raise RuntimeError("Expected CPU mode; run through check_environment.sh")
    env = Environment(str(ROOT / "upstream/ravens/environments/assets"), disp=False)
    try:
        task = BlockInsertion()
        task.mode = "test"
        env.set_task(task)
        env.seed(1)
        observation = env.reset()
        shapes = {key: [list(value.shape) for value in values]
                  for key, values in observation.items()}
    finally:
        env.close()
    packages = ("tensorflow-cpu", "keras", "tensorflow-addons", "numpy",
                "scipy", "pybullet", "gym", "opencv-python-headless")
    result = {"python": sys.version, "upstream_commit": commit,
              "versions": {name: importlib.metadata.version(name) for name in packages},
              "visible_gpus": 0, "observation_shapes": shapes,
              "headless_scene_reset": "passed"}
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results/environment.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
