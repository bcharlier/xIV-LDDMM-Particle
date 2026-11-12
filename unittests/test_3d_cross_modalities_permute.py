"""Regression test for the 3D cross-modalities permutation example."""

import math
import os
import sys
import tempfile
import unittest
from contextlib import contextmanager
from pathlib import Path
from sys import path as sys_path

sys_path.append(os.path.join(os.path.dirname(__file__), ".."))
sys_path.insert(0, os.path.dirname(__file__))

import torch
import xmodmap
import xmodmap.preprocess

from utils import isclose, read_baseline, write_baseline

BASELINE = Path(__file__).with_name("baseline_3d_cross_modalities_permute.csv.gz")
DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "3D_MTL"


@contextmanager
def single_cpu_thread():
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield
    finally:
        torch.set_num_threads(previous_threads)


def calculate_results():
    with single_cpu_thread(), torch.device("cpu"), tempfile.TemporaryDirectory() as savedir:
        S, w_S, zeta_S = torch.load(DATA_DIR / "MTL_lf_123.pt", map_location="cpu").values()
        T, w_T, zeta_T = torch.load(DATA_DIR / "MTL_lf_231.pt", map_location="cpu").values()
        w_S = w_S.reshape(-1, 1)
        w_T = w_T.reshape(-1, 1)
        S, T, _, _ = xmodmap.preprocess.centerData(S, T, w_S, w_T)
        Stilde, Ttilde, _, _ = xmodmap.preprocess.rescaleData(S, T)

        hamiltonian = xmodmap.deformation.Hamiltonian(
            [0.1, 0.025], Stilde, cA=1.0, cS=10.0, cT=1.0,
            dimEff=3, single=True, transform="affine",
        )
        hamiltonian.weight = 0.1
        shooting = xmodmap.deformation.Shooting(hamiltonian, nt=5)

        dataloss = xmodmap.distance.LossVarifoldNorm([0.1], w_T, zeta_T, Ttilde)
        dataloss.weight = 1.0
        pi_loss = xmodmap.distance.PiRegularizationSystem(zeta_S, zeta_T, w_T, norm=True)
        pi_loss.weight = 0.0

        variables = {
            "px": torch.zeros_like(Stilde).requires_grad_(True),
            "pw": torch.zeros_like(w_S).requires_grad_(True),
            "qx": Stilde.clone().detach().requires_grad_(True),
            "qw": w_S.clone().detach().requires_grad_(True),
            "zeta_S": zeta_S,
            "pi_ST": pi_loss.Pi_ST_init(w_S).clone().detach().requires_grad_(True),
        }
        model = xmodmap.model.CrossModality(hamiltonian, shooting, dataloss, pi_loss)
        model.init(
            variables,
            ["px", "pw", "pi_ST"],
            precond={"px": 1.0, "pw": 1.0 / 3 / w_S, "pi_ST": 1.0},
            savedir=savedir,
        )

        def loss_components():
            return tuple(
                value.detach().item()
                for value in model.loss(**model.get_variables_optimized())
            )

        initial_h, initial_data, initial_pi = loss_components()
        model.optimize(10)
        final_h, final_data, final_pi = loss_components()

        optimized = model.get_variables_optimized()
        coupling = optimized["pi_ST"].detach().square().cpu()
        results = {
            "initial_hamiltonian_loss": initial_h,
            "initial_data_loss": initial_data,
            "initial_pi_loss": initial_pi,
            "final_hamiltonian_loss": final_h,
            "final_data_loss": final_data,
            "final_pi_loss": final_pi,
        }
        for axis, size in enumerate(coupling.shape):
            results[f"pi_ST_shape_{axis}"] = size
        for index, value in enumerate(coupling.reshape(-1).tolist()):
            results[f"pi_ST_{index}"] = value
        return results


def updateTest():
    write_baseline(BASELINE, calculate_results())
    print(f"Updated {BASELINE}")


class Test3DCrossModalitiesPermute(unittest.TestCase):
    def test_matches_baseline(self):
        actual = calculate_results()
        expected = read_baseline(BASELINE)
        self.assertEqual(actual.keys(), expected.keys())
        initial = sum(actual[f"initial_{name}_loss"] for name in ("hamiltonian", "data", "pi"))
        final = sum(actual[f"final_{name}_loss"] for name in ("hamiltonian", "data", "pi"))
        self.assertLess(final, initial)
        self.assertEqual(actual["pi_ST_shape_0"], expected["pi_ST_shape_0"])
        self.assertEqual(actual["pi_ST_shape_1"], expected["pi_ST_shape_1"])
        differences = [
            f"{name}: actual {value}, baseline {expected[name]}"
            for name, value in actual.items()
            if not math.isfinite(value) or not math.isfinite(expected[name])
            or not isclose(value, expected[name])
        ]
        self.assertFalse(differences, f"{len(differences)} mismatches: {differences[:10]}")


if __name__ == "__main__":
    if sys.argv[1:] == ["--updateTest"]:
        updateTest()
    else:
        unittest.main()
