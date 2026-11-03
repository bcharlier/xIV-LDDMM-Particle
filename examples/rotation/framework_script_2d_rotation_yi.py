"""
2D rotation example with xmodmap2.

The target is an exact rigid copy of the source (rotation by `angle` and a random
translation), so the recovered transform can be checked against the ground truth.

Usage (from any folder):
    python framework_script_2d_rotation_yi.py                      # mode "all-mixed" (= xmodmap)
    python framework_script_2d_rotation_yi.py --mode rigid         # closed-form rigid flow
    python framework_script_2d_rotation_yi.py --mode "non-rigid+rigid" --steps 50 --npoints 2000

Differences with framework_script_2D_rotation.py (xmodmap):
    * no Hamiltonian / Shooting objects: a single xmodmap2.deformation.Deformation whose
      mode is selected by a string; the shooting is a method of its Hamiltonian;
    * cS is renamed cNR (cost of the non-rigid mode) and scaling has its own cost cScale
      (it was hard-coded to 1 in xmodmap);
    * the data file stores a dict of mesh fields (3D points, no features): points are
      reduced to 2D BEFORE the rotation is applied and unit masses are used.
"""

import argparse
import math
import os
from sys import path as sys_path

HERE = os.path.dirname(os.path.abspath(__file__))
sys_path.append(os.path.join(HERE, "..", ".."))

import torch
import xmodmap2

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--mode", default="all-mixed", help=xmodmap2.deformation.available_modes())
parser.add_argument("--steps", type=int, default=300, help="number of optimizer steps")
parser.add_argument("--npoints", type=int, default=None, help="random subsample of the source (default: all)")
args = parser.parse_args()

# set random seed
torch.manual_seed(0)
torch.set_printoptions(precision=6)


# Data Loading
mode = args.mode
savedir = os.path.join(HERE, "output", "2D_amygdala_yi", mode.replace("+", "_"))
os.makedirs(savedir, exist_ok=True)

source_file = os.path.join(HERE, "..", "..", "data", "2D_amygdala", "hat_run77_10000points_real.pt")
data = torch.load(source_file)
if isinstance(data, dict):  # mesh fields saved from the .vtk file: 'points' (N x 3) and no point_data
    S = data["points"].clone()
    point_data = data.get("point_data") or {}
    nu_S = point_data["nu_X"].clone() if "nu_X" in point_data else torch.ones(S.shape[0], 1)
else:
    S, nu_S = data

d = 2
dimEff = 2

angle, translation = 0.6, torch.randn(1, 2, device=S.device, dtype=S.dtype)

if args.npoints is not None and args.npoints < S.shape[0]:
    keep = torch.randperm(S.shape[0])[: args.npoints]
    S, nu_S = S[keep], nu_S[keep]

# make them both in the z = 0 plane / drop z BEFORE the (2D) rotation
if d == 2:
    S = S[:, 0:2].contiguous()
elif dimEff == 2:
    S[:, -1] = 0

R = torch.eye(d, device=S.device, dtype=S.dtype)
R[:2, :2] = torch.tensor(
    [[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]],
    device=S.device,
    dtype=S.dtype,
)
t_true = torch.zeros(1, d, device=S.device, dtype=S.dtype)
t_true[:, :2] = translation
T, nu_T = S.clone() @ R.T + t_true, nu_S.clone()

labs = nu_T.shape[-1]  # in target
labS = nu_S.shape[-1]  # template
sigmaRKHS = [0.2, 0.1, 0.05]  # should be fraction of total domain of S+T
sigmaVar = [0.2, 0.1, 0.05, 0.02]  # should be fraction of total domain of S+T
steps = args.steps
kScale = torch.tensor(1.)
cA = 1.0      # rotation
cT = 1.0      # translation
cScale = 1.0  # scaling (implicitly 1 in xmodmap)
cNR = 10.0    # non-rigid (cS in xmodmap); 10 vs. 10000 for "almost rigid"
Csqpi = 10000.0
Csqlamb = 100.0
lambInit = -1
single = False
gamma = 0.1  # weight of the Hamiltonian in the loss

print("mode: ", mode)
print("S type: ", S.dtype, S.shape)
print("nu_S type: ", nu_S.dtype, nu_S.shape)

from xmodmap2.preprocess.makePQ_legacy import makePQ
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
    "mode": mode,
    "cA": cA,
    "cT": cT,
    "cScale": cScale,
    "cNR": cNR,
    "sigmaRKHS": sigmaRKHS,
    "d": d,
    "dimEff": dimEff,
    "single": single
}

torch.save(sm, os.path.join(savedir, "sm.pt"))


# Model Setup

## Varifold distance without boundaries
dataloss = xmodmap2.distance.LossVarifoldNorm(sigmaVar, w_T, zeta_T, Ttilde)
dataloss.normalize_across_scale(Stilde, w_S, zeta_S, pi_STinit)
dataloss.weight = 1.

## Deformation: the mode string selects the Hamiltonian (and its shooting)
deformation = xmodmap2.deformation.Deformation(
    mode,
    Stilde,
    sigma=sigmaRKHS,
    cA=cA,
    cT=cT,
    cScale=cScale,
    cNR=cNR,
    dimEff=dimEff,
    single=single,
)
deformation.weight = gamma
print(deformation)


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

loss = xmodmap2.model.SingleModality(deformation, dataloss)
loss.init(variable_init, variable_to_optimize, precond=precond, savedir=savedir)
#loss.resume(variable_init, os.path.join(savedir, 'checkpoint.pt'))
loss.optimize(steps)

# Saving
precondVar = loss.get_variables_optimized()
torch.save(precondVar, os.path.join(savedir, "precondVar.pt"))

print("shooting forward")
px1, pw1, qx1, qw1 = deformation(precondVar['px'], precondVar['pw'], precondVar['qx'], precondVar['qw'])  # get Deformed Atlas


# Report: energy of each mode and recovered transform versus the ground truth
with torch.no_grad():
    px0, pw0, qx0, qw0 = (precondVar[k].detach() for k in ("px", "pw", "qx", "qw"))
    print("energy of each mode (unweighted):",
          {k: round(v.item(), 6) for k, v in deformation.energies(px0, pw0, qx0, qw0).items()})

    # In rescaled coordinates: Ttilde = Stilde R^T + b_true
    b_true = ((m @ R.T + t_true - m) / s).flatten()
    if deformation.has_closed_form:
        Phi, b = deformation.flow_map(px0, pw0, qx0, qw0)
        print("closed-form map  Phi:", Phi.tolist(), " b:", b.tolist())
    else:  # best rigid fit (weighted Procrustes) of the deformed source
        w = qw0 / qw0.sum()
        xc, yc = (w * qx0).sum(0), (w * qx1.detach()).sum(0)
        U, _, Vt = torch.linalg.svd(((qx1.detach() - yc) * w).T @ (qx0 - xc))
        D = torch.eye(d)
        D[-1, -1] = torch.sign(torch.linalg.det(U @ Vt))
        Phi = U @ D @ Vt
        b = yc - Phi @ xc
        print("best rigid fit   Phi:", Phi.tolist(), " b:", b.tolist())
    print(f"rotation angle: recovered {math.atan2(Phi[1, 0].item(), Phi[0, 0].item()):.6f}, true {angle}")
    print("translation (rescaled units): recovered", b.tolist(), " true", b_true.tolist())
    rms = (qx1.detach() - Ttilde).square().sum(-1).mean().sqrt().item()
    print(f"RMS distance between deformed source and target (rescaled units, known correspondences): {rms:.3e}")

#print("shooting back")
#_,_,_,_,Td,wTd = deformation.hamiltonian.shoot_backward(px1, pw1, qx1, qw1, Ttilde, w_T)[-1] # get Deformed Target

#f = loss.print_log()
#f.savefig(os.path.join(savedir,"loss.png"),dpi=300)

#f = loss.print_log(logScale=True)
#f.savefig(os.path.join(savedir,"logloss.png"),dpi=300)
