"""Offline regression tests for ZIP/copy export and path policy."""

import tempfile
import unittest
import zipfile
from pathlib import Path

from app.export_utils import (
    ONE_GB,
    build_zip_bytes,
    copy_without_compression,
    make_export_plan,
)


class ExportUtilsTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.data = self.base / "data"
        self.data.mkdir()
        self.first = self.data / "A" / "room" / "rir.wav"
        self.second = self.data / "B" / "rir.wav"
        self.first.parent.mkdir(parents=True)
        self.second.parent.mkdir(parents=True)
        self.first.write_bytes(b"ABCD" * 100)
        self.second.write_bytes(b"EFGH" * 200)

    def rows(self):
        return [
            {"dataset": "A", "file_path": str(self.first)},
            {"dataset": "B", "file_path": str(self.second)},
        ]

    def plan(self, rows=None):
        return make_export_plan(
            self.rows() if rows is None else rows,
            self.data,
            lambda row: Path(row["file_path"]),
        )

    def test_plan_counts_sizes_and_root_restriction(self):
        outside = self.base / "outside.wav"
        outside.write_bytes(b"private")
        rows = self.rows() + [
            {"dataset": "O", "file_path": str(outside)},
            {"dataset": "Missing", "file_path": str(self.data / "absent.wav")},
        ]
        plan = self.plan(rows)
        self.assertEqual(plan.candidates, 4)
        self.assertEqual(len(plan.entries), 2)
        self.assertEqual(plan.skipped_missing, 1)
        self.assertEqual(plan.skipped_outside, 1)
        self.assertEqual(plan.total_bytes, 1200)
        self.assertEqual(ONE_GB, 1_000_000_000)

    def test_zip_contents_and_byte_progress(self):
        plan = self.plan()
        updates = []
        archive_data = build_zip_bytes(
            plan, b"dataset,file_path\n", lambda processed, total, name: updates.append((processed, total, name))
        )
        self.assertTrue(updates)
        self.assertEqual(updates[-1][:2], (1200, 1200))
        with tempfile.TemporaryFile() as zip_file:
            zip_file.write(archive_data)
            zip_file.seek(0)
            with zipfile.ZipFile(zip_file) as zf:
                self.assertIsNone(zf.testzip())
                self.assertEqual(zf.read("files/A/room/rir.wav"), self.first.read_bytes())
                self.assertEqual(zf.read("files/B/rir.wav"), self.second.read_bytes())
                self.assertIn("manifest.csv", zf.namelist())
                self.assertIn("zip_diagnostics.txt", zf.namelist())

    def test_copy_is_uncompressed_and_keeps_source_tree(self):
        root = self.base / "allowed_copies"
        output = copy_without_compression(
            self.plan(), b"manifest contents", root / "project", root
        )
        self.assertEqual((output / "files/A/room/rir.wav").read_bytes(), self.first.read_bytes())
        self.assertEqual((output / "files/B/rir.wav").read_bytes(), self.second.read_bytes())
        self.assertEqual((output / "manifest.csv").read_bytes(), b"manifest contents")
        self.assertTrue((output / "zip_diagnostics.txt").is_file())
        self.assertTrue(self.first.is_file())  # source is never moved

    def test_copy_rejects_destination_outside_allowlist(self):
        with self.assertRaisesRegex(ValueError, "allowed copy root"):
            copy_without_compression(
                self.plan(), b"manifest", self.base / "unauthorized", self.base / "allowed"
            )

    def test_name_collision_is_handled_without_overwrite(self):
        rows = self.rows() + [{"dataset": "A", "file_path": str(self.first)}]
        names = [entry.relative_name for entry in self.plan(rows).entries]
        self.assertEqual(len(set(names)), 3)


if __name__ == "__main__":
    unittest.main()
