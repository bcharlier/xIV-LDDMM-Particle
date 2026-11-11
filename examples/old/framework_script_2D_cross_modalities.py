import os
from sys import path as sys_path
sys_path.append(os.path.join(os.path.dirname(__file__), ".."))

import torch
import xmodmap
import xmodmap.preprocess

# set random seed
device = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_default_device(device)

data_dir = os.path.join(os.path.dirname(__file__), "..", "data", "2D_NisslToTau")

source_file = os.path.join(data_dir, "nissl_32bin.pt")
target_file = os.path.join(data_dir, "tau_32bin.pt")
S, w_S, zeta_S = torch.load(source_file, map_location=device).values()
T, w_T, zeta_T = torch.load(target_file, map_location=device).values()

Stilde, Ttilde, s, m = xmodmap.preprocess.rescaleData(S, T)

d = 2
dimEff = 2
sigmaRKHS = [0.1]
sigmaVar = [0.2] 
cA, cT, cS = 1., 1., 1.
single = False
gamma = 0.1

## Varifold distance without boundaries
dataloss = xmodmap.distance.LossVarifoldNorm(sigmaVar, w_T, zeta_T, Ttilde)   
dataloss.weight = 1.

## Cross-Modality
piLoss = xmodmap.distance.PiRegularizationSystem(zeta_S, zeta_T, w_T)
piLoss.weight = 0. * gamma / torch.log(torch.tensor(zeta_T.shape[-1]))

## non-rigid and affine deformations
hamiltonian = xmodmap.deformation.Hamiltonian(sigmaRKHS, Stilde, cA=cA, cS=cS,  cT=cT, dimEff=dimEff, single=single, transform="diffeo")
hamiltonian.weight = gamma
shooting = xmodmap.deformation.Shooting(hamiltonian, nt=10)


# Optimization

variable_init = {
    "px": torch.zeros_like(Stilde).requires_grad_(True),
    "pw": torch.zeros_like(w_S).requires_grad_(True),
    "qx": Stilde.clone().detach().requires_grad_(True),
    "qw": w_S.clone().detach().requires_grad_(True),
    "zeta_S": zeta_S,
    "pi_ST":piLoss.Pi_ST_init(w_S).clone().detach().requires_grad_(True), 
}

variable_to_optimize = ["px", "pw", "pi_ST"]

precond = {
    "px": 1.,
    "pw": 1. / dimEff / w_S,
    "pi_ST": 1.,
}

loss = xmodmap.model.CrossModality(hamiltonian, shooting, dataloss, piLoss)
loss.init(variable_init, variable_to_optimize, precond=precond)
loss.optimize(10)

# Plot source, target and deformed source (with time slider)
import xmodmap.plot

variables_optimum = loss.get_variables_optimized()
trajectory = shooting(*(variables_optimum[k] for k in ["px", "pw", "qx", "qw"]))

xmodmap.plot.plot_registration_3d(Stilde, w_S, zeta_S, Ttilde, w_T, zeta_T, trajectory, Pi_ST)
