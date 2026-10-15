#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Educational asset verification for Google Research Ravens / Transporter Networks.
"""Verify SHA256/CRC and the exact 10-train / 3-test subset without loading pickle."""
import json
from pathlib import Path

from fetch_subset import EPISODE_NAME, FIELDS, destination, hash_file

ROOT = Path(__file__).resolve().parents[1]


def verify(root=ROOT):
    root = root.resolve()
    config = json.loads((root / "config/minimal-assets.json").read_text(encoding="utf-8"))
    manifests = sorted((root / "results").glob("fetch-manifest-*.json"), reverse=True)
    manifest = None
    for path in manifests:
        candidate = json.loads(path.read_text(encoding="utf-8"))
        if candidate.get("status") == "complete":
            manifest = candidate
            break
    if manifest is None:
        raise RuntimeError("No completed asset manifest; run bash scripts/fetch_assets.sh")
    for key in ("model_prefix", "step", "train_episodes", "test_episodes"):
        if manifest.get(key) != config[key]:
            raise ValueError("Completed manifest has the wrong asset selection: " + key)
    records = manifest["files"]
    if len(records) != config["expected_files"]:
        raise ValueError("Expected exactly 67 downloaded files")
    names = [item["destination"] for item in records]
    if len(set(names)) != len(names):
        raise ValueError("Duplicate file paths in completed manifest")
    expected = {f'{config["model_prefix"]}/{kind}-ckpt-{config["step"]}.h5'
                for kind in ("attention", "transport")}
    seeds = {}
    for split in ("train", "test"):
        count = config[split + "_episodes"]
        split_root = root / "data" / ("block-insertion-" + split)
        # Dataset.load counts files; extra episodes silently enlarge evaluation.
        fields_present = {path.name for path in split_root.iterdir() if path.is_dir()}
        if fields_present != set(FIELDS):
            raise ValueError("Unexpected or incomplete dataset fields: " + split)
        canonical = None
        for field in FIELDS:
            filenames = sorted(path.name for path in (split_root / field).iterdir())
            parsed = [EPISODE_NAME.fullmatch(name) for name in filenames]
            if (len(filenames) != count or not all(parsed) or
                    [int(item.group(1)) for item in parsed] != list(range(count))):
                raise ValueError("Expected contiguous episode IDs of exact subset size: " + split)
            if canonical is None:
                canonical = filenames
                seeds[split] = [int(item.group(2)) for item in parsed]
            elif filenames != canonical:
                raise ValueError("Dataset field filenames disagree: " + split)
            expected.update(f"data/block-insertion-{split}/{field}/{name}" for name in filenames)
    if seeds["test"] != config["test_seeds"]:
        raise ValueError("Unexpected test scene seeds")
    if set(names) != expected:
        raise ValueError("Completed manifest does not match the exact tutorial subset")
    for record in records:
        relative = Path(record["destination"])
        target = destination(root, relative)
        size, crc, sha256 = hash_file(target)
        if (size != record["bytes"] or "%08x" % crc != record["crc32"] or
                sha256 != record["sha256"]):
            raise ValueError("Asset checksum mismatch: " + relative.as_posix())
    result = {"status": "complete", "files_verified": len(records),
              "model_prefix": config["model_prefix"], "checkpoint_step": config["step"],
              "train_episodes": config["train_episodes"], "test_episodes": config["test_episodes"],
              "test_seeds": seeds["test"], "sha256_and_crc32": "passed",
              "verification_note": "Integrity checked against the range-download manifest; pickle not deserialized."}
    (root / "results/assets.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


if __name__ == "__main__":
    print(json.dumps(verify(), indent=2))
