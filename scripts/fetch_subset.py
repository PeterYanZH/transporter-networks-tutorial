#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Educational download tooling for Google Research Ravens / Transporter Networks.
"""Read Ravens ZIP indexes and fetch a small verified subset using HTTP Range.

Python 3.10+ standard library only. No archive is downloaded in full. Examples:
  python fetch_subset.py --list
  python fetch_subset.py --fetch --project-root . \
    --model-prefix checkpoints/block-insertion-transporter-10-0 --step 10000 \
    --train-episodes 10 --test-episodes 3

Use the exact model prefix and step reported by --list; the example is not a
promise that this checkpoint exists. Pickle files are copied without loading.
"""

import argparse
import contextlib
import datetime
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
import struct
import sys
import tempfile
import urllib.request
import zipfile
import zlib


BASE_URL = "https://storage.googleapis.com/ravens-assets/"
FIELDS = ("action", "color", "depth", "reward", "info")
CHUNK_SIZE = 1024 * 1024
MAX_READ = 64 * 1024 * 1024
MAX_MEMBER = 1024 * 1024 * 1024
EPISODE_NAME = re.compile(r"^(\d{6})-(-?\d+)\.pkl$")


class RangeReader(io.RawIOBase):
    """Seekable remote file; refuse anything except the exact requested range."""

    def __init__(self, url, timeout=60):
        super().__init__()
        self.url = url
        self.timeout = timeout
        self.position = 0
        self.transferred_bytes = 0
        self.range_requests = 0
        self._cache_start = 0
        self._cache = b""
        request = urllib.request.Request(
            url, method="HEAD", headers={"Accept-Encoding": "identity"})
        with urllib.request.urlopen(request, timeout=timeout) as response:
            if response.status != 200:
                raise RuntimeError("HEAD failed for %s: %s" % (url, response.status))
            length = response.headers.get("Content-Length", "")
            if not length.isdigit() or int(length) <= 0:
                raise RuntimeError("Missing or invalid archive Content-Length")
            self.size = int(length)
            self.etag = response.headers.get("ETag")
            self.last_modified = response.headers.get("Last-Modified")

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.position

    def seek(self, offset, whence=io.SEEK_SET):
        if whence == io.SEEK_CUR:
            offset += self.position
        elif whence == io.SEEK_END:
            offset += self.size
        elif whence != io.SEEK_SET:
            raise ValueError("Invalid seek mode")
        if offset < 0 or offset > self.size:
            raise ValueError("Seek outside archive")
        self.position = offset
        return offset

    def iter_range(self, start, size):
        """Stream a whole member through one validated HTTP Range connection."""
        if size < 0 or start < 0 or start + size > self.size:
            raise ValueError("Range outside archive")
        if size == 0:
            return
        end = start + size - 1
        expected = end - start + 1
        headers = {"Range": "bytes=%d-%d" % (start, end),
                   "Accept-Encoding": "identity"}
        if self.etag and not self.etag.startswith("W/"):
            headers["If-Match"] = self.etag
        request = urllib.request.Request(self.url, headers=headers)
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            # Check headers BEFORE reading a body. In particular, a server that
            # ignores Range must never make us download the full archive.
            if response.status != 206:
                raise RuntimeError("Refusing non-206 Range response: %s" % response.status)
            wanted = "bytes %d-%d/%d" % (start, end, self.size)
            if response.headers.get("Content-Range") != wanted:
                raise RuntimeError("Unexpected Content-Range: %r, wanted %r" % (
                    response.headers.get("Content-Range"), wanted))
            content_length = response.headers.get("Content-Length")
            if content_length is not None and content_length != str(expected):
                raise RuntimeError("Range Content-Length mismatch")
            if response.headers.get("Content-Encoding", "identity") != "identity":
                raise RuntimeError("Refusing encoded Range response")
            if self.etag and response.headers.get("ETag") != self.etag:
                raise RuntimeError("Archive ETag changed while reading")
            self.range_requests += 1
            remaining = expected
            while remaining:
                body = response.read(min(CHUNK_SIZE, remaining))
                self.transferred_bytes += len(body)
                if not body or len(body) > remaining:
                    raise RuntimeError("Range response body length mismatch")
                remaining -= len(body)
                yield body
            extra = response.read(1)
            self.transferred_bytes += len(extra)
            if extra:
                raise RuntimeError("Range response body exceeds requested length")

    def _range(self, start, end):
        return b"".join(self.iter_range(start, end - start + 1))

    def read(self, size=-1):
        if self.closed:
            raise ValueError("Read from closed archive")
        if size is None or size < 0:
            size = self.size - self.position
        size = min(size, self.size - self.position)
        if size > MAX_READ:
            raise RuntimeError("Refusing unexpectedly large single ZIP read")
        if size == 0:
            return b""
        chunks = []
        remaining = size
        while remaining:
            cache_offset = self.position - self._cache_start
            if 0 <= cache_offset < len(self._cache):
                part = self._cache[cache_offset:cache_offset + remaining]
            else:
                # Small read-ahead combines local ZIP headers and filenames.
                count = min(max(remaining, 65536), CHUNK_SIZE,
                            self.size - self.position)
                self._cache_start = self.position
                self._cache = self._range(self.position, self.position + count - 1)
                part = self._cache[:remaining]
            chunks.append(part)
            self.position += len(part)
            remaining -= len(part)
        return b"".join(chunks)

    def summary(self):
        return {"url": self.url, "archive_bytes": self.size, "etag": self.etag,
                "last_modified": self.last_modified,
                "transferred_bytes": self.transferred_bytes,
                "range_requests": self.range_requests}


def safe_name(name):
    """Reject ambiguous ZIP names, absolute paths, traversal, and symlinks later."""
    if not name or "\\" in name or "\x00" in name or ":" in name:
        raise ValueError("Unsafe ZIP member name: %r" % name)
    parts = name.rstrip("/").split("/")
    if any(part in ("", ".", "..") for part in parts):
        raise ValueError("Unsafe ZIP member path: %r" % name)
    path = PurePosixPath(name)
    if path.is_absolute():
        raise ValueError("Absolute ZIP member path: %r" % name)
    return path


def index_archive(archive):
    index = {}
    for info in archive.infolist():
        safe_name(info.filename)
        if info.filename in index:
            raise ValueError("Duplicate ZIP member: %s" % info.filename)
        if stat.S_ISLNK(info.external_attr >> 16):
            raise ValueError("ZIP symlink is not allowed: %s" % info.filename)
        if info.flag_bits & 1:
            raise ValueError("Encrypted ZIP members are not supported")
        index[info.filename] = info
    return index


def data_groups(index):
    """Find task split directories even if the ZIP wraps them in another dir."""
    groups = {"train": {}, "test": {}}
    for name, info in index.items():
        if info.is_dir():
            continue
        parts = safe_name(name).parts
        for split in groups:
            marker = "block-insertion-" + split
            if len(parts) >= 3 and parts[-3] == marker and parts[-2] in FIELDS:
                match = EPISODE_NAME.fullmatch(parts[-1])
                if not match:
                    raise ValueError("Unrecognized episode filename: %s" % name)
                episode_id = int(match.group(1))
                seed = int(match.group(2))
                key = (episode_id, seed, parts[-1])
                fields = groups[split].setdefault(key, {})
                if parts[-2] in fields:
                    raise ValueError("Duplicate episode field in ZIP: %s" % name)
                fields[parts[-2]] = info
    return groups


def model_listing(index):
    groups = {}
    for name, info in index.items():
        path = safe_name(name)
        if "block-insertion" not in name or info.is_dir():
            continue
        match = re.fullmatch(r"(attention|transport)-ckpt-(\d+)\.h5", path.name)
        if match:
            prefix = str(path.parent)
            step = match.group(2)
            groups.setdefault(prefix, {}).setdefault(step, []).append({
                "member": name, "bytes": info.file_size,
                "compressed_bytes": info.compress_size,
                "crc32": "%08x" % info.CRC})
    return groups


def print_listing(readers, indexes):
    models = model_listing(indexes[0])
    datasets = {}
    for split, episodes in data_groups(indexes[1]).items():
        keys = sorted(episodes)
        datasets[split] = {
            "episode_count": len(keys),
            "complete_episode_count": sum(set(v) == set(FIELDS) for v in episodes.values()),
            "compressed_bytes": sum(i.compress_size for v in episodes.values() for i in v.values()),
            "uncompressed_bytes": sum(i.file_size for v in episodes.values() for i in v.values()),
            "sample_episodes": [{"id": key[0], "seed": key[1],
                                 "fields": {field: info.filename for field, info in episodes[key].items()}}
                                for key in keys[:3]]}
    print(json.dumps({"archives": [r.summary() for r in readers],
                      "total_archive_bytes": sum(r.size for r in readers),
                      "block_insertion_models": models,
                      "datasets": datasets}, indent=2))


def select_files(indexes, prefix, step, train_count, test_count):
    prefix_path = safe_name(prefix)
    dirname = prefix_path.name
    if not re.fullmatch(r"block-insertion-transporter-\d+-\d+", dirname):
        raise ValueError("Choose an original block-insertion-transporter-N-R model prefix from --list")
    selected = []
    for kind in ("attention", "transport"):
        filename = "%s-ckpt-%d.h5" % (kind, step)
        source = str(prefix_path / filename)
        if source not in indexes[0]:
            raise ValueError("Requested model member not found: %s (run --list)" % source)
        selected.append((0, indexes[0][source], Path("checkpoints") / dirname / filename))
    groups = data_groups(indexes[1])
    for split, count in (("train", train_count), ("test", test_count)):
        by_id = {}
        for key, fields in groups[split].items():
            if key[0] in by_id:
                raise ValueError("Multiple seeds for episode ID %d in %s" % (key[0], split))
            by_id[key[0]] = (key, fields)
        # The upstream Dataset.load uses IDs 0..n_episodes-1; keep that convention.
        for episode_id in range(count):
            if episode_id not in by_id:
                raise ValueError("Missing %s episode %d" % (split, episode_id))
            key, fields = by_id[episode_id]
            if set(fields) != set(FIELDS):
                raise ValueError("Incomplete %s episode %d" % (split, episode_id))
            for field in FIELDS:
                target = Path("data") / ("block-insertion-" + split) / field / key[2]
                selected.append((1, fields[field], target))
    return selected


def destination(root, relative):
    target = (root / relative).resolve()
    try:
        target.relative_to(root)
    except ValueError:
        raise ValueError("Destination escapes project root: %s" % relative)
    return target


def hash_file(path):
    digest = hashlib.sha256()
    crc = 0
    count = 0
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(CHUNK_SIZE), b""):
            count += len(block)
            digest.update(block)
            crc = zlib.crc32(block, crc)
    return count, crc & 0xffffffff, digest.hexdigest()


def member_data_offset(archive, reader, info):
    """Validate the local header; ZIP64 sizes come from the central directory."""
    reader.seek(info.header_offset)
    header = reader.read(30)
    if len(header) != 30:
        raise ValueError("Truncated local ZIP header")
    fields = struct.unpack("<4s5H3L2H", header)
    signature, flags, method = fields[0], fields[2], fields[3]
    filename_size, extra_size = fields[-2:]
    if signature != b"PK\x03\x04":
        raise ValueError("Invalid local ZIP header signature")
    if flags != info.flag_bits or method != info.compress_type:
        raise ValueError("Local ZIP header disagrees with central directory")
    if flags & (1 | 0x20 | 0x40):
        raise ValueError("Encrypted or patched ZIP entries are not supported")
    local_fields = reader.read(filename_size + extra_size)
    if len(local_fields) != filename_size + extra_size:
        raise ValueError("Truncated local ZIP filename/extra fields")
    name = local_fields[:filename_size].decode("utf-8" if flags & 0x800 else "cp437")
    safe_name(name)
    if name != info.filename:
        raise ValueError("Local ZIP filename disagrees with central directory")
    offset = info.header_offset + 30 + filename_size + extra_size
    limit = min(reader.size, archive.start_dir,
                getattr(info, "_end_offset", None) or archive.start_dir)
    if info.compress_size < 0 or offset + info.compress_size > limit:
        raise ValueError("ZIP member payload overlaps another entry or index")
    return offset


def extract_one(archive, reader, info, target, relative):
    if info.is_dir() or info.file_size > MAX_MEMBER:
        raise ValueError("Unexpected member type or size: %s" % info.filename)
    if info.compress_type not in (zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED):
        raise ValueError("Unsupported ZIP compression: %s" % info.filename)
    before = reader.transferred_bytes
    existing = target.exists()
    if existing:
        size, crc, sha256 = hash_file(target)
        if size != info.file_size or crc != info.CRC:
            raise ValueError("Existing file differs; refusing overwrite: %s" % target)
    else:
        target.parent.mkdir(parents=True, exist_ok=True)
        temp_name = None
        try:
            with tempfile.NamedTemporaryFile(prefix=".ravens-part-", dir=target.parent,
                                             delete=False) as output:
                temp_name = output.name
                digest = hashlib.sha256()
                size = 0
                crc = 0

                def write_block(block):
                    nonlocal size, crc
                    if block:
                        size += len(block)
                        if size > info.file_size or size > MAX_MEMBER:
                            raise ValueError("Expanded ZIP member exceeds declared size")
                        digest.update(block)
                        crc = zlib.crc32(block, crc)
                        output.write(block)

                offset = member_data_offset(archive, reader, info)
                decoder = (zlib.decompressobj(-zlib.MAX_WBITS)
                           if info.compress_type == zipfile.ZIP_DEFLATED else None)
                with contextlib.closing(reader.iter_range(offset, info.compress_size)) as source:
                    for compressed in source:
                        if decoder is None:
                            write_block(compressed)
                        else:
                            pending = compressed
                            while pending:
                                write_block(decoder.decompress(pending, CHUNK_SIZE))
                                if decoder.unused_data:
                                    raise ValueError("Extra bytes after the DEFLATE stream")
                                pending = decoder.unconsumed_tail
                if decoder is not None:
                    # Drain bounded output buffers, then require a complete raw
                    # DEFLATE stream. No full member is held in memory.
                    while True:
                        block = decoder.decompress(b"", CHUNK_SIZE)
                        if not block:
                            break
                        write_block(block)
                    if not decoder.eof or decoder.unused_data or decoder.unconsumed_tail:
                        raise ValueError("Truncated or malformed DEFLATE stream")
                crc &= 0xffffffff
                sha256 = digest.hexdigest()
                if size != info.file_size or crc != info.CRC:
                    raise ValueError("ZIP size/CRC mismatch: %s" % info.filename)
            os.replace(temp_name, target)
            temp_name = None
        finally:
            if temp_name is not None:
                os.unlink(temp_name)
    return {"source_url": reader.url, "source_member": info.filename,
            "destination": relative.as_posix(), "bytes": size,
            "compressed_bytes": info.compress_size, "sha256": sha256,
            "crc32": "%08x" % crc, "crc_verified": True,
            "reused_existing": existing,
            "transferred_bytes_this_file": reader.transferred_bytes - before}


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--list", action="store_true", help="Only fetch ZIP indexes and print JSON")
    mode.add_argument("--fetch", action="store_true", help="Fetch only selected members")
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--config", type=Path, help="Pinned asset profile; disallows selection overrides")
    parser.add_argument("--model-prefix", help="Exact directory prefix reported by --list")
    parser.add_argument("--step", type=int)
    parser.add_argument("--train-episodes", type=int, default=10)
    parser.add_argument("--test-episodes", type=int, default=3)
    parser.add_argument("--timeout", type=int, default=60)
    args = parser.parse_args()
    config = None
    if args.config:
        if (args.model_prefix or args.step is not None or
                args.train_episodes != 10 or args.test_episodes != 3):
            parser.error("--config cannot be combined with selection overrides")
        config = json.loads(args.config.read_text(encoding="utf-8"))
        args.model_prefix = config["model_prefix"]
        args.step = config["step"]
        args.train_episodes = config["train_episodes"]
        args.test_episodes = config["test_episodes"]
    if args.fetch and (not args.model_prefix or args.step is None or args.step < 0):
        parser.error("--fetch requires --model-prefix and a nonnegative --step")
    if not 0 <= args.train_episodes <= 1000 or not 0 <= args.test_episodes <= 1000:
        parser.error("Episode counts must be between 0 and 1000")
    if args.timeout <= 0:
        parser.error("--timeout must be positive")

    readers = []
    archives = []
    manifest = None
    manifest_path = None
    try:
        for filename in ("checkpoints.zip", "block-insertion.zip"):
            reader = RangeReader(BASE_URL + filename, args.timeout)
            readers.append(reader)
            if config:
                expected = next(item for item in config["archives"] if item["url"] == reader.url)
                if reader.size != expected["archive_bytes"] or reader.etag != expected["etag"]:
                    raise ValueError("Official archive metadata changed; inspect before updating the pinned profile")
            archives.append(zipfile.ZipFile(reader, "r"))
        indexes = [index_archive(archive) for archive in archives]
        if args.list:
            print_listing(readers, indexes)
            return
        selected = select_files(indexes, args.model_prefix, args.step,
                                args.train_episodes, args.test_episodes)
        if config:
            if (len(selected) != config["expected_files"] or
                    sum(i.compress_size for _, i, _ in selected) != config["selected_compressed_bytes"] or
                    sum(i.file_size for _, i, _ in selected) != config["selected_uncompressed_bytes"]):
                raise ValueError("Selected files disagree with the pinned asset profile")
        root = args.project_root.resolve()
        # Preflight every destination and member before writing any output.
        destinations = []
        for archive_id, info, relative in selected:
            if info.file_size > MAX_MEMBER or info.is_dir():
                raise ValueError("Unexpected selected member: %s" % info.filename)
            destinations.append(destination(root, relative))
        if len(set(destinations)) != len(destinations):
            raise ValueError("Duplicate output paths")
        root.mkdir(parents=True, exist_ok=True)
        stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
        manifest_path = destination(root, Path("results") / ("fetch-manifest-" + stamp + ".json"))
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest = {"created_utc": stamp, "status": "in_progress",
                    "model_prefix": args.model_prefix, "step": args.step,
                    "train_episodes": args.train_episodes,
                    "test_episodes": args.test_episodes,
                    "selected_compressed_bytes": sum(i.compress_size for _, i, _ in selected),
                    "selected_uncompressed_bytes": sum(i.file_size for _, i, _ in selected),
                    "files": [],
                    "transfer_note": "Payload bytes of actual HTTP Range responses, including ZIP metadata and small read-ahead; excludes HTTP headers. No pickle was deserialized."}
        for (archive_id, info, relative), target in zip(selected, destinations):
            print("Fetching %s" % info.filename, file=sys.stderr, flush=True)
            record = extract_one(archives[archive_id], readers[archive_id],
                                 info, target, relative)
            manifest["files"].append(record)
        manifest["status"] = "complete"
    except (Exception, KeyboardInterrupt) as exc:
        if manifest is not None:
            manifest["status"] = "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed"
            manifest["error"] = "%s: %s" % (type(exc).__name__, exc)
        raise
    finally:
        if manifest is not None:
            manifest["archives"] = [r.summary() for r in readers]
            manifest["actual_transferred_bytes"] = sum(r.transferred_bytes for r in readers)
            manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
            print("Manifest: %s" % manifest_path, file=sys.stderr, flush=True)
        for archive in archives:
            archive.close()
        for reader in readers:
            reader.close()


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, RuntimeError, zipfile.BadZipFile) as error:
        print("ERROR: %s" % error, file=sys.stderr)
        sys.exit(1)
