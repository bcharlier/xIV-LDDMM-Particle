"""Test non-rigid Hamiltonian; regenerate the baseline with --updateTest."""

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
from baseline_io import read_baseline, write_baseline


BASELINE = Path(__file__).with_name("baseline_hamiltonian_nonrigid.csv.gz")


def calculate_results():
    with torch.device("cpu"):
        px = torch.tensor([[0.5, -0.2], [0.3, 0.6]], dtype=torch.float64, requires_grad=True)
        pw = torch.tensor([[0.4], [-0.1]], dtype=torch.float64, requires_grad=True)
        qx = torch.tensor([[0.0, 0.2], [1.0, -0.5]], dtype=torch.float64, requires_grad=True)
        qw = torch.tensor([[0.7], [1.2]], dtype=torch.float64, requires_grad=True)
        points = torch.tensor([[0.5, 0.7], [-0.2, 0.9], [1.5, 0.1]], dtype=torch.float64)
        weights = torch.tensor([[0.3], [0.9], [1.1]], dtype=torch.float64)
        hamiltonian = Hamiltonian([0.6, 1.2], qx.detach(), cS=2.0, dimEff=2, transform="diffeo")
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


class TestHamiltonianNonRigid(unittest.TestCase):
    def test_matches_baseline(self):
        actual = calculate_results()
        expected = read_baseline(BASELINE)
        self.assertEqual(actual.keys(), expected.keys())
        for name, value in actual.items():
            with self.subTest(result=name):
                self.assertTrue(math.isfinite(value))
                self.assertTrue(math.isfinite(expected[name]))
                self.assertTrue(
                    math.isclose(value, expected[name], rel_tol=1e-8, abs_tol=1e-10),
                    f"{name}: actual {value}, baseline {expected[name]}",
                )

    def test_single_particle_properties(self):
        with torch.device("cpu"):
            px = torch.tensor([[1.0, 0.0]], dtype=torch.float64, requires_grad=True)
            pw = torch.zeros((1, 1), dtype=torch.float64, requires_grad=True)
            qx = torch.tensor([[0.3, -0.2]], dtype=torch.float64, requires_grad=True)
            qw = torch.ones((1, 1), dtype=torch.float64, requires_grad=True)
            hamiltonian = Hamiltonian([1.0], qx.detach(), cS=2.0, dimEff=2, transform="diffeo")
            energy = hamiltonian(px, pw, qx, qw)
            momentum_rate, _, position_rate, mass_rate = hamiltonian.forward_system(px, pw, qx, qw)

            # At the particle center, the Gaussian is 1 and its normalization is cS = 2.
            self.assertAlmostEqual(energy.item(), 0.25)
            torch.testing.assert_close(position_rate, torch.tensor([[0.5, 0.0]], dtype=torch.float64))
            torch.testing.assert_close(momentum_rate, torch.zeros_like(momentum_rate))
            torch.testing.assert_close(mass_rate, torch.zeros_like(mass_rate))
            point_velocity, weight_rate = hamiltonian.velocity(qx, qw, px, pw, qx, qw)
            torch.testing.assert_close(point_velocity, position_rate)
            torch.testing.assert_close(weight_rate, mass_rate)


if __name__ == "__main__":
    if sys.argv[1:] == ["--updateTest"]:
        updateTest()
    else:
        unittest.main()
