"""Read and update numerical CSV baselines with Git provenance."""

import csv
import gzip
import math
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
METADATA_KEYS = ("__commit__", "__worktree_dirty__")


def open_baseline(path, mode):
    if path.suffix == ".gz":
        return gzip.open(path, f"{mode}t", newline="", encoding="utf-8")
    return path.open(mode, newline="", encoding="utf-8")


def write_baseline(path, results):
    if not all(math.isfinite(value) for value in results.values()):
        raise ValueError("Cannot update the baseline with non-finite results")
    if set(results) & set(METADATA_KEYS):
        raise ValueError("Result name conflicts with baseline metadata")

    def git(*args):
        return subprocess.run(
            ["git", "-C", str(ROOT), *args],
            check=True, capture_output=True, text=True,
        ).stdout.strip()

    commit = git("rev-parse", "HEAD")
    dirty = bool(git("status", "--porcelain", "--untracked-files=normal"))
    with open_baseline(path, "w") as file:
        writer = csv.writer(file)
        writer.writerow(("result", "value"))
        writer.writerow((METADATA_KEYS[0], commit))
        writer.writerow((METADATA_KEYS[1], str(dirty).lower()))
        writer.writerows(results.items())


def read_baseline(path):
    with open_baseline(path, "r") as file:
        reader = csv.reader(file)
        if next(reader, None) != ["result", "value"]:
            raise ValueError(f"Invalid baseline header in {path}")
        rows = list(reader)
    if any(len(row) != 2 for row in rows):
        raise ValueError(f"Invalid baseline row in {path}")
    values = dict(rows)
    if len(values) != len(rows):
        raise ValueError(f"Duplicate baseline result names in {path}")
    for key in METADATA_KEYS:
        if key not in values:
            raise ValueError(f"Missing {key} metadata in {path}")
        del values[key]
    return {name: float(value) for name, value in values.items()}
