import os, math
from sys import path as sys_path

sys_path.append(os.path.join("..", ".."))

import torch
import xmodmap
import xmodmap.preprocess

# set random seed
torch.manual_seed(0)
torch.set_printoptions(precision=6)
device = "cuda" if torch.cuda.is_available() * 0 else "cpu"
torch.set_default_device(device)
# Data Loading: 8-876; 9-856; 10-836; 11-816; 12-796
savedir = os.path.join("output", "2D_amygdala")


source_file = os.path.join("..", "..", "data", "2D_amygdala","hat_run77_10000points_real_source.pt")
target_file = os.path.join("..", "..", "data", "2D_amygdala","hat_run77_10000points_real_target.pt")
S, w_S, zeta_S = torch.load(source_file, map_location=device).values()
T, w_T, zeta_T, _, _, _ = torch.load(target_file, map_location=device).values()

w_S = w_S[:, None]
w_T = w_T[:, None]

Stilde, Ttilde, s, m = xmodmap.preprocess.rescaleData(S, T)

d = 2
dimEff = 2
sigmaRKHS = [0.2,0.1, 0.05]
sigmaVar = [0.2, 0.1, 0.05, 0.02] #[0.5, 0.2, 0.05, 0.02]  # as of 3/16, should be fraction of total domain of S+T #10.0
steps = 30
beta = None
res = 1.0
kScale = torch.tensor(1.)
extra = ""
cA = 1.0
cT = 1.0  # original is 0.5
cS = 10.0 #10 vs. 10000 for rigid
Csqpi = 10000.0
Csqlamb = 100.0
eta0 = torch.tensor(0.2).sqrt()
lambInit = -1
single = False
gamma = 0.1

## Varifold distance without boundaries
dataloss = xmodmap.distance.LossVarifoldNorm(sigmaVar, w_T, zeta_T, Ttilde)   # Dream of : (sigmaVar, T, nu_T)
dataloss.weight = 1.

## non-rigid and affine deformations
hamiltonian = xmodmap.deformation.Hamiltonian(sigmaRKHS, Stilde, cA=cA, cS=cS,  cT=cT, dimEff=dimEff, single=single)
hamiltonian.weight = gamma
shooting = xmodmap.deformation.Shooting(sigmaRKHS, Stilde, cA=cA, cS=cS,  cT=cT, dimEff=dimEff, single=single)


# Optimization

variable_init = {
    "px": torch.zeros_like(Stilde).requires_grad_(True),
    "pw": torch.zeros_like(w_S).requires_grad_(True),
    "qx": Stilde.clone().detach().requires_grad_(True),
    "qw": w_S.clone().detach().requires_grad_(True),
    "zeta_S": zeta_S,
}

variable_to_optimize = ["px", "pw"]

precond = {
    "px": torch.rsqrt(kScale),
    "pw": torch.rsqrt(kScale) / dimEff / w_S,
}

loss = xmodmap.model.SingleModality(hamiltonian, shooting, dataloss)
loss.init(variable_init, variable_to_optimize, precond=precond, savedir=savedir)
#loss.resume(variable_init, os.path.join(savedir, 'checkpoint.pt'))
loss.optimize(steps)

# Saving
precondVar = loss.get_variables_optimized()
torch.save(precondVar,os.path.join(savedir,"precondVar.pt"))

print("shooting forward")
px1,pw1,qx1,qw1 = shooting(precondVar['px'], precondVar['pw'], precondVar['qx'], precondVar['qw'])[-1] # get Deformed Atlas

#print("shooting back")
#shootingBack = xmodmap.deformation.ShootingBackwards(sigmaRKHS,Stilde,cA=cA,cS=cS,cT=cT, dimEff=dimEff, single=False)
#_,_,_,_,Td,wTd = shootingBack(px1, pw1, qx1, qw1, Ttilde, w_T)[-1] # get Deformed Target

#f = loss.print_log()
#f.savefig(os.path.join(savedir,"loss.png"),dpi=300)

#f = loss.print_log(logScale=True)
#f.savefig(os.path.join(savedir,"logloss.png"),dpi=300)


