# SPDX-License-Identifier: Apache-2.0
# Offline safety tests for the tutorial's Ravens asset range downloader.
import io
from pathlib import Path
import stat
import sys
import tempfile
import unittest
from unittest.mock import patch
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from fetch_subset import (RangeReader, destination, extract_one, index_archive,
                          safe_name, select_files)


class Response(io.BytesIO):
    def __init__(self, body=b"", status=200, headers=None):
        super().__init__(body)
        self.status = status
        self.headers = headers or {}
        self.bytes_read = 0

    def read(self, size=-1):
        result = super().read(size)
        self.bytes_read += len(result)
        return result


class MemoryRangeReader(io.BytesIO):
    """In-memory transport exercising the production ZIP validation/extraction."""
    def __init__(self, content):
        super().__init__(content)
        self.content = content
        self.size = len(content)
        self.transferred_bytes = 0
        self.url = "https://example.invalid/test.zip"

    def iter_range(self, start, size):
        data = self.content[start:start + size]
        self.transferred_bytes += len(data)
        yield data


def make_zip(name="safe/file.pkl", payload=b"example data", compression=zipfile.ZIP_DEFLATED):
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=compression) as archive:
        archive.writestr(name, payload)
    return buffer.getvalue()


class RangeTests(unittest.TestCase):
    def request(self, response):
        head = Response(headers={"Content-Length": "10", "ETag": '"fixed"'})
        mock = patch("urllib.request.urlopen", side_effect=[head, response])
        return mock

    def test_correct_range_and_if_match(self):
        response = Response(b"123", 206, {"Content-Range": "bytes 1-3/10",
                           "Content-Length": "3", "ETag": '"fixed"'})
        with self.request(response) as opener:
            reader = RangeReader("https://example.invalid/archive.zip")
            self.assertEqual(b"".join(reader.iter_range(1, 3)), b"123")
            self.assertEqual(reader.transferred_bytes, 3)
            self.assertEqual(opener.call_args.args[0].headers["If-match"], '"fixed"')

    def test_server_ignoring_range_is_rejected_before_body(self):
        response = Response(b"entire huge archive", 200)
        with self.request(response):
            reader = RangeReader("https://example.invalid/archive.zip")
            with self.assertRaisesRegex(RuntimeError, "non-206"):
                list(reader.iter_range(1, 3))
        self.assertEqual(response.bytes_read, 0)

    def test_invalid_headers_are_rejected_before_body(self):
        cases = ({"Content-Range": "bytes 0-2/10"}, {"Content-Length": "4"},
                 {"ETag": '"changed"'}, {"Content-Encoding": "gzip"})
        for change in cases:
            with self.subTest(change=change):
                headers = {"Content-Range": "bytes 1-3/10", "Content-Length": "3",
                           "ETag": '"fixed"'}
                headers.update(change)
                response = Response(b"123", 206, headers)
                with self.request(response):
                    reader = RangeReader("https://example.invalid/archive.zip")
                    with self.assertRaises(RuntimeError):
                        list(reader.iter_range(1, 3))
                self.assertEqual(response.bytes_read, 0)

    def test_truncated_or_extra_response_body(self):
        for body in (b"12", b"1234"):
            with self.subTest(body=body):
                response = Response(body, 206, {"Content-Range": "bytes 1-3/10",
                                    "Content-Length": "3", "ETag": '"fixed"'})
                with self.request(response):
                    reader = RangeReader("https://example.invalid/archive.zip")
                    with self.assertRaises(RuntimeError):
                        list(reader.iter_range(1, 3))


class ZipTests(unittest.TestCase):
    def test_reject_unsafe_member_names(self):
        for name in ("../escape", "/absolute", "a/../b", "a\\b", "C:/evil",
                     "a//b", "./a", "a/./b", "a\x00b", ""):
            with self.subTest(name=name), self.assertRaises(ValueError):
                safe_name(name)

    def test_reject_path_outside_project(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            with self.assertRaises(ValueError):
                destination(root, Path("..") / "outside")

    def test_reject_duplicate_names(self):
        import warnings
        buffer = io.BytesIO()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            with zipfile.ZipFile(buffer, "w") as archive:
                archive.writestr("a.pkl", b"one")
                archive.writestr("a.pkl", b"two")
        with zipfile.ZipFile(io.BytesIO(buffer.getvalue())) as archive:
            with self.assertRaisesRegex(ValueError, "Duplicate"):
                index_archive(archive)

    def test_reject_symlink(self):
        buffer = io.BytesIO()
        member = zipfile.ZipInfo("link")
        member.create_system = 3
        member.external_attr = (stat.S_IFLNK | 0o777) << 16
        with zipfile.ZipFile(buffer, "w") as archive:
            archive.writestr(member, "target")
        with zipfile.ZipFile(io.BytesIO(buffer.getvalue())) as archive:
            with self.assertRaisesRegex(ValueError, "symlink"):
                index_archive(archive)

    def test_extract_and_reuse_stored_or_deflated(self):
        for compression in (zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED):
            with self.subTest(compression=compression), tempfile.TemporaryDirectory() as directory:
                payload = b"example data" * 1000
                reader = MemoryRangeReader(make_zip(payload=payload, compression=compression))
                with zipfile.ZipFile(reader) as archive:
                    info = archive.getinfo("safe/file.pkl")
                    target = Path(directory) / "safe/file.pkl"
                    record = extract_one(archive, reader, info, target, Path("safe/file.pkl"))
                    self.assertEqual(target.read_bytes(), payload)
                    self.assertTrue(record["crc_verified"])
                    record = extract_one(archive, reader, info, target, Path("safe/file.pkl"))
                    self.assertTrue(record["reused_existing"])
                    self.assertEqual(record["transferred_bytes_this_file"], 0)

    def test_crc_failure_removes_partial_output(self):
        with tempfile.TemporaryDirectory() as directory:
            reader = MemoryRangeReader(make_zip())
            with zipfile.ZipFile(reader) as archive:
                info = archive.getinfo("safe/file.pkl")
                info.CRC ^= 1
                target = Path(directory) / "file.pkl"
                with self.assertRaisesRegex(ValueError, "CRC mismatch"):
                    extract_one(archive, reader, info, target, Path("file.pkl"))
                self.assertFalse(target.exists())
                self.assertEqual(list(Path(directory).iterdir()), [])

    def test_existing_different_file_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            reader = MemoryRangeReader(make_zip())
            with zipfile.ZipFile(reader) as archive:
                target = Path(directory) / "file.pkl"
                target.write_bytes(b"keep this content")
                with self.assertRaisesRegex(ValueError, "refusing overwrite"):
                    extract_one(archive, reader, archive.getinfo("safe/file.pkl"), target, Path("file.pkl"))
                self.assertEqual(target.read_bytes(), b"keep this content")

    def test_incomplete_episode_rejected(self):
        model_prefix = "checkpoints/block-insertion-transporter-10-0"
        model_index = {f"{model_prefix}/{kind}-ckpt-10000.h5": zipfile.ZipInfo(
                       f"{model_prefix}/{kind}-ckpt-10000.h5") for kind in ("attention", "transport")}
        name = "block-insertion-train/action/000000-0.pkl"
        with self.assertRaisesRegex(ValueError, "Incomplete"):
            select_files([model_index, {name: zipfile.ZipInfo(name)}], model_prefix, 10000, 1, 0)


if __name__ == "__main__":
    unittest.main()
