"""Test rotation-only Hamiltonian; regenerate the baseline with --updateTest."""

import math
import os
import sys
import unittest
from pathlib import Path
from sys import path as sys_path

sys_path.append(os.path.join(os.path.dirname(__file__), ".."))
sys_path.insert(0, os.path.dirname(__file__))

import torch

from xmodmap.deformation import Hamiltonian
from utils import assert_close, isclose, read_baseline, write_baseline


BASELINE = Path(__file__).with_name("baseline_hamiltonian_rotation.csv.gz")


def calculate_results():
    with torch.device("cpu"):
        px = torch.tensor([[0.5, -0.2], [0.3, 0.6]], dtype=torch.float64, requires_grad=True)
        pw = torch.tensor([[0.4], [-0.1]], dtype=torch.float64, requires_grad=True)
        qx = torch.tensor([[0.0, 0.2], [1.0, -0.5]], dtype=torch.float64, requires_grad=True)
        qw = torch.tensor([[0.7], [1.2]], dtype=torch.float64, requires_grad=True)
        points = torch.tensor([[0.5, 0.7], [-0.2, 0.9], [1.5, 0.1]], dtype=torch.float64)
        weights = torch.tensor([[0.3], [0.9], [1.1]], dtype=torch.float64)
        hamiltonian = Hamiltonian([0.2], qx, cA=2.0, dimEff=2, transform="rotation")
        energy = hamiltonian(px, pw, qx, qw)
        system = hamiltonian.forward_system(px, pw, qx, qw)
        point_system = hamiltonian.system_with_points(px, pw, qx, qw, points, weights)
        point_velocity = hamiltonian.velocity(points, weights, px, pw, qx, qw)

        results = {"energy": energy.item()}
        for prefix, tensors in (
            ("system", system), ("point_system", point_system), ("velocity", point_velocity)
        ):
            for component, tensor in enumerate(tensors):
                for index, value in enumerate(tensor.detach().reshape(-1).tolist()):
                    results[f"{prefix}_{component}_{index}"] = value
        return results


def updateTest():
    results = calculate_results()
    write_baseline(BASELINE, results)
    print(f"Updated {BASELINE}")


class TestHamiltonianRotation(unittest.TestCase):
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

    def test_rotation_properties(self):
        with torch.device("cpu"):
            px = torch.tensor([[0.0, -1.0], [0.0, 1.0]], dtype=torch.float64, requires_grad=True)
            pw = torch.tensor([[0.4], [-0.1]], dtype=torch.float64, requires_grad=True)
            qx = torch.tensor([[-1.0, 0.0], [1.0, 0.0]], dtype=torch.float64, requires_grad=True)
            qw = torch.ones((2, 1), dtype=torch.float64, requires_grad=True)
            hamiltonian = Hamiltonian([0.2], qx, cA=2.0, dimEff=2, transform="rotation")
            energy = hamiltonian(px, pw, qx, qw)
            _, _, position_rate, mass_rate = hamiltonian.forward_system(px, pw, qx, qw)

            # The centered points rotate with angular velocity 0.5; H = 0.5.
            self.assertTrue(isclose(energy.item(), 0.5))
            assert_close(position_rate, torch.tensor(
                [[0.0, -0.5], [0.0, 0.5]], dtype=torch.float64
            ))
            assert_close(mass_rate, torch.zeros_like(mass_rate))
            point_velocity, weight_rate = hamiltonian.velocity(qx, qw, px, pw, qx, qw)
            assert_close(point_velocity, position_rate)
            assert_close(weight_rate, torch.zeros_like(weight_rate))
            # A rotational velocity is perpendicular to the radius.
            assert_close(
                (qx * position_rate).sum(dim=1), torch.zeros(2, dtype=torch.float64)
            )


if __name__ == "__main__":
    if sys.argv[1:] == ["--updateTest"]:
        updateTest()
    else:
        unittest.main()
