import csv
import gzip
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from baseline_io import read_baseline, write_baseline


class TestBaselineIO(unittest.TestCase):
    def test_write_and_read_provenance(self):
        for status, expected_dirty in (("", "false"), ("?? new_file\n", "true")):
            with self.subTest(status=status), tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "baseline.csv.gz"
                responses = [
                    subprocess.CompletedProcess([], 0, stdout="abc123\n"),
                    subprocess.CompletedProcess([], 0, stdout=status),
                ]
                with patch("baseline_io.subprocess.run", side_effect=responses):
                    write_baseline(path, {"energy": 0.25})
                with gzip.open(path, "rt", newline="", encoding="utf-8") as file:
                    rows = list(csv.reader(file))
                self.assertEqual(rows[:3], [
                    ["result", "value"],
                    ["__commit__", "abc123"],
                    ["__worktree_dirty__", expected_dirty],
                ])
                self.assertEqual(read_baseline(path), {"energy": 0.25})

    def test_git_failure_does_not_replace_baseline(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "baseline.csv.gz"
            path.write_text("original", encoding="utf-8")
            with patch("baseline_io.subprocess.run", side_effect=FileNotFoundError("git")):
                with self.assertRaises(FileNotFoundError):
                    write_baseline(path, {"energy": 0.25})
            self.assertEqual(path.read_text(encoding="utf-8"), "original")

    def test_duplicate_rows_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "baseline.csv"
            path.write_text(
                "result,value\n__commit__,unknown\n__worktree_dirty__,unknown\n"
                "energy,1\nenergy,2\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "Duplicate"):
                read_baseline(path)

    def test_reads_plain_csv_for_compatibility(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "baseline.csv"
            path.write_text(
                "result,value\n__commit__,abc123\n__worktree_dirty__,false\nenergy,0.25\n",
                encoding="utf-8",
            )
            self.assertEqual(read_baseline(path), {"energy": 0.25})


if __name__ == "__main__":
    unittest.main()
