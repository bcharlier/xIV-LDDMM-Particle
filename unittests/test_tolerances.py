"""Verify the shared numerical comparison tolerances."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch

from utils import assert_close, isclose


class TestTolerances(unittest.TestCase):
    def test_absolute_tolerance(self):
        self.assertTrue(isclose(0.0, 0.9e-5))
        self.assertFalse(isclose(0.0, 1.1e-5))
        assert_close(torch.tensor(0.0), torch.tensor(0.9e-5))
        with self.assertRaises(AssertionError):
            assert_close(torch.tensor(0.0), torch.tensor(1.1e-5))

    def test_relative_tolerance(self):
        self.assertTrue(isclose(100.0, 100.009))
        self.assertFalse(isclose(100.0, 100.011))
        assert_close(torch.tensor(100.0), torch.tensor(100.009))
        with self.assertRaises(AssertionError):
            assert_close(torch.tensor(100.0), torch.tensor(100.011))
