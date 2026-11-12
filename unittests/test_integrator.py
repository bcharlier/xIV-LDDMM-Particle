"""Test Ralston integration; regenerate the baseline with --updateTest."""

import math
import os
import sys
import unittest
from pathlib import Path
from sys import path as sys_path

sys_path.append(os.path.join(os.path.dirname(__file__), ".."))
sys_path.insert(0, os.path.dirname(__file__))

import torch

from xmodmap.deformation.Shooting import Abstract_Shooting
from utils import assert_close, isclose, read_baseline, write_baseline


BASELINE = Path(__file__).with_name("baseline_integrator.csv.gz")


def calculate_results():
    with torch.device("cpu"):
        position = torch.tensor([1.0, -0.5], dtype=torch.float64, requires_grad=True)
        velocity = torch.tensor([0.25, 0.75], dtype=torch.float64, requires_grad=True)

        def oscillator(x, v):
            return v, -x

        trajectory = Abstract_Shooting.ralstonIntegrator(
            oscillator, (position, velocity), nt=5, deltat=1.5
        )
        results = {}
        for step, state in enumerate(trajectory):
            for name, tensor in zip(("position", "velocity"), state):
                for index, value in enumerate(tensor.detach().tolist()):
                    results[f"step_{step}_{name}_{index}"] = value

        objective = trajectory[-1][0].square().sum() + trajectory[-1][1].square().sum()
        results["final_energy"] = objective.item()
        gradients = torch.autograd.grad(objective, (position, velocity))
        for name, gradient in zip(("position", "velocity"), gradients):
            for index, value in enumerate(gradient.detach().tolist()):
                results[f"gradient_{name}_{index}"] = value
        return results


def updateTest():
    results = calculate_results()
    write_baseline(BASELINE, results)
    print(f"Updated {BASELINE}")


class TestIntegrator(unittest.TestCase):
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

    def test_constant_derivative(self):
        with torch.device("cpu"):
            initial = torch.tensor([1.0, -2.0], dtype=torch.float64)
            derivative = torch.tensor([0.5, 2.0], dtype=torch.float64)
            trajectory = Abstract_Shooting.ralstonIntegrator(
                lambda x: (derivative,), (initial,), nt=4, deltat=2.0
            )
            self.assertEqual(len(trajectory), 5)
            for step, state in enumerate(trajectory):
                self.assertEqual(len(state), 1)
                assert_close(state[0], initial + step * 0.5 * derivative)
            assert_close(initial, torch.tensor([1.0, -2.0], dtype=torch.float64))


if __name__ == "__main__":
    if sys.argv[1:] == ["--updateTest"]:
        updateTest()
    else:
        unittest.main()
