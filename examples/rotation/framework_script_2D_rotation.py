import os, math
from sys import path as sys_path
sys_path.append(os.path.join("..", ".."))

import torch
import xmodmap

# set random seed
torch.manual_seed(0)
torch.set_printoptions(precision=6)


# Data Loading: 8-876; 9-856; 10-836; 11-816; 12-796
savedir = os.path.join("output", "2D_amygdala")


source_file = os.path.join("..", "..", "data", "2D_amygdala","hat_run77_10000points_real.pt")
S, nu_S = torch.load(source_file)

angle, translation = .6, torch.randn(1, 2, device=S.device, dtype=S.dtype)
T, nu_T = (
    S.clone()
    @ torch.tensor(
        [[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]],
        device=S.device,
        dtype=S.dtype,
    ).T
    + translation,
    nu_S.clone(),
)

d = 2
dimEff = 2
labs = nu_T.shape[-1]  # in target
labS = nu_S.shape[-1]  # template
sigmaRKHS = [0.2,0.1, 0.05] #[0.2,0.1,0.05] as of 3/16, should be fraction of total domain of S+T #[10.0]
sigmaVar = [0.2, 0.1, 0.05, 0.02] #[0.5, 0.2, 0.05, 0.02]  # as of 3/16, should be fraction of total domain of S+T #10.0
steps = 300
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
gamma = 0.1 #0.01 #10.0

# make them both in z = 0 plane
if dimEff == 2:
    S[:,-1] = 0
    T[:,-1] = 0
if d == 2:
    S = S[:,0:2]
    T = T[:,0:2]

print("S type: ", S.dtype)
print("nu_S type: ", nu_S.dtype)
print("T type: ", T.dtype)
print("nu_T type: ", nu_T.dtype)
'''
A = torch.zeros((2,2))
A[0,1] = 1
A[1,0] = -1
S = S @ A
'''

from xmodmap.preprocess.makePQ_legacy import makePQ
(
    w_S,
    w_T,
    zeta_S,
    zeta_T,
    q0,
    p0,
    numS,
    Stilde,
    Ttilde,
    s,
    m,
    pi_STinit,
    lamb0,
) = makePQ(S, nu_S, T, nu_T,
           Csqpi=Csqpi,
           lambInit=lambInit,
           Csqlamb=Csqlamb)

sm = {
    "s": s,
    "m": m,
    "cA": cA,
    "cT": cT,
    "cS": cS,
    "sigmaRKHS": sigmaRKHS,
    "d": d,
    "dimEff": dimEff,
    "single": single
}

torch.save(sm,os.path.join(savedir,"sm.pt"))


# Model Setup

## Varifold distance without boundaries
dataloss = xmodmap.distance.LossVarifoldNorm(sigmaVar, w_T, zeta_T, Ttilde)   # Dream of : (sigmaVar, T, nu_T)
dataloss.normalize_across_scale(Stilde, w_S, zeta_S, pi_STinit)
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


