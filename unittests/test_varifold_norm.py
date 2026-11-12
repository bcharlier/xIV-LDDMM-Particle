import math
import os
import sys
import unittest
from pathlib import Path
from sys import path as sys_path

sys_path.append(os.path.join(os.path.dirname(__file__), ".."))
sys_path.insert(0, os.path.dirname(__file__))

import torch

from xmodmap.distance import LossVarifoldNorm
from utils import isclose, read_baseline, write_baseline


BASELINE = Path(__file__).with_name("baseline_varifold_norm.csv.gz")


def calculate_results():
    # Keep baseline generation independent of whether the machine has a GPU.
    with torch.device("cpu"):
        source = torch.tensor(
            [[0.1, -0.2], [0.8, 0.4], [1.3, -0.1]],
            dtype=torch.float64, requires_grad=True,
        )
        weights = torch.tensor(
            [[0.5], [0.9], [0.6]], dtype=torch.float64, requires_grad=True
        )
        features = torch.tensor(
            [[1.0, 0.0], [0.3, 0.7], [0.0, 1.0]],
            dtype=torch.float64, requires_grad=True,
        )
        mapping = torch.tensor(
            [[0.8, 0.2], [0.3, 0.7]], dtype=torch.float64, requires_grad=True
        )
        target = torch.tensor([[0.0, 0.0], [1.0, 0.5]], dtype=torch.float64)
        target_weights = torch.tensor([[0.7], [1.1]], dtype=torch.float64)
        target_features = torch.tensor([[1.0, 0.0], [0.2, 0.8]], dtype=torch.float64)

        loss = LossVarifoldNorm([0.4, 1.2], target_weights, target_features, target)
        loss.beta = [0.6, 1.4]
        value = loss(source, weights, features, mapping)
        gradients = torch.autograd.grad(value, (source, weights, features, mapping))

        results = {"loss": value.item()}
        for name, gradient in zip(("source", "weights", "features", "mapping"), gradients):
            for index, value in enumerate(gradient.detach().reshape(-1).tolist()):
                results[f"gradient_{name}_{index}"] = value
        return results


def updateTest():
    results = calculate_results()
    write_baseline(BASELINE, results)
    print(f"Updated {BASELINE}")


class TestVarifoldNorm(unittest.TestCase):
    def test_matches_baseline(self):
        actual = calculate_results()
        expected = read_baseline(BASELINE)
        self.assertEqual(actual.keys(), expected.keys())
        for name, value in actual.items():
            with self.subTest(result=name):
                self.assertTrue(math.isfinite(value))
                self.assertTrue(math.isfinite(expected[name]))
                self.assertTrue(
                    isclose(value, expected[name]),
                    f"{name}: actual {value}, baseline {expected[name]}",
                )


if __name__ == "__main__":
    if sys.argv[1:] == ["--updateTest"]:
        updateTest()
    else:
        unittest.main()
