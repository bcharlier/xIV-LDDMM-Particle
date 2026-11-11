"""Regression test for the 2D rotation example; update with --updateTest."""

import math
import os
import sys
import tempfile
import unittest
from pathlib import Path
from sys import path as sys_path

sys_path.append(os.path.join(os.path.dirname(__file__), ".."))
sys_path.insert(0, os.path.dirname(__file__))

import torch
import xmodmap
import xmodmap.preprocess

from baseline_io import read_baseline, write_baseline


BASELINE = Path(__file__).with_name("baseline_2d_rotation_registration.csv")
DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "2D_amygdala"


def calculate_results():
    with torch.device("cpu"), tempfile.TemporaryDirectory() as savedir:
        S, w_S, zeta_S = torch.load(DATA_DIR / "onehot.pt", map_location="cpu").values()
        T, w_T, zeta_T = torch.load(
            DATA_DIR / "onehot_rotation.pt", map_location="cpu"
        ).values()
        Stilde, Ttilde, _, _ = xmodmap.preprocess.rescaleData(S, T)

        dataloss = xmodmap.distance.LossVarifoldNorm([0.2], w_T, zeta_T, Ttilde)
        dataloss.weight = 1.0
        hamiltonian = xmodmap.deformation.Hamiltonian(
            [0.2, 0.1, 0.05], Stilde, cA=1.0, cS=1.0, cT=1.0,
            dimEff=2, single=False, transform="rigid",
        )
        hamiltonian.weight = 0.1
        shooting = xmodmap.deformation.Shooting(hamiltonian, nt=1)

        variables = {
            "px": torch.zeros_like(Stilde).requires_grad_(True),
            "pw": torch.zeros_like(w_S).requires_grad_(True),
            "qx": Stilde.clone().detach().requires_grad_(True),
            "qw": w_S.clone().detach().requires_grad_(True),
            "zeta_S": zeta_S,
        }
        model = xmodmap.model.SingleModality(hamiltonian, shooting, dataloss)
        model.init(variables, ["px", "pw"], precond={"px": 1.0, "pw": 1.0 / 2 / w_S},
                   savedir=savedir)

        def loss_components():
            return tuple(value.detach().item() for value in model.loss(
                **model.get_variables_optimized()
            ))

        initial_h, initial_data = loss_components()
        model.optimize(10)
        final_h, final_data = loss_components()
        optimum = model.get_variables_optimized()
        results = {
            "initial_hamiltonian_loss": initial_h,
            "initial_data_loss": initial_data,
            "final_hamiltonian_loss": final_h,
            "final_data_loss": final_data,
        }
        for name in ("px", "pw", "qx", "qw"):
            tensor = optimum[name].detach().cpu()
            for axis, size in enumerate(tensor.shape):
                results[f"{name}_shape_{axis}"] = size
            for index, value in enumerate(tensor.reshape(-1).tolist()):
                results[f"{name}_{index}"] = value
        return results


def updateTest():
    write_baseline(BASELINE, calculate_results())
    print(f"Updated {BASELINE}")


class Test2DRotationRegistration(unittest.TestCase):
    def test_matches_baseline(self):
        actual = calculate_results()
        expected = read_baseline(BASELINE)
        self.assertEqual(actual.keys(), expected.keys())
        initial = actual["initial_hamiltonian_loss"] + actual["initial_data_loss"]
        final = actual["final_hamiltonian_loss"] + actual["final_data_loss"]
        self.assertLess(final, initial)
        for name, value in actual.items():
            with self.subTest(result=name):
                self.assertTrue(math.isfinite(value))
                self.assertTrue(math.isfinite(expected[name]))
                self.assertTrue(
                    math.isclose(value, expected[name], rel_tol=1e-4, abs_tol=1e-6),
                    f"{name}: actual {value}, baseline {expected[name]}",
                )


if __name__ == "__main__":
    if sys.argv[1:] == ["--updateTest"]:
        updateTest()
    else:
        unittest.main()
