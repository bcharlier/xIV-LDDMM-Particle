import numpy as np
import scipy as sp
import torch
from pykeops.torch import Vi,Vj

from xmodmap.io.getOutput import writeParticleVTK

def applyAffine(Z, nu_Z, A, tau, bc=False):
    """
    Makes a new set of particles based on an input set and applying the affine transformation given by matrix A and translation, tau
    """
    print("max before ", torch.max(Z, axis=0))
    R = torch.clone(Z)
    nu_R = torch.clone(nu_Z)
    if not bc:
        R = R @ A.T + tau
    else:
        # rotate around center of mass
        xc = torch.sum((torch.sum(nu_R, axis=-1) * Z), axis=0) / torch.sum(nu_R)
        Rp = R - xc
        R = Rp @ A.T + tau
    print("max ", torch.max(R, axis=0))
    return R, nu_R


def flip(Z):
    R = torch.clone(Z)
    R[:, 0] = -1.0 * R[:, 0]

    return R


def alignBaryCenters(S, nu_S, T, nu_T):
    """
    Translate S to be on barycenter of T with barycenters computed based on sum over features in nu_S and nu_T
    """
    wS = nu_S.sum(axis=-1)
    wT = nu_T.sum(axis=-1)

    xcS = (S * wS).sum(dim=0) / (wS.sum(dim=0))
    xcT = (T * wT).sum(dim=0) / (wT.sum(dim=0))

    tau = xcT - xcS
    Snew = S + tau

    return Snew


def get3DRotMatrix(thetaX, thetaY, thetaZ):
    """
    thetaX, thetaY, and thetaZ should all be torch tensors in radians
    """
    Ax = torch.zeros((3, 3))
    Ax[0, 0] = torch.tensor(1.0)
    Ax[1, 1] = torch.cos(thetaX)
    Ax[2, 2] = Ax[1, 1]
    Ax[1, 2] = -torch.sin(thetaX)
    Ax[2, 1] = torch.sin(thetaX)

    Ay = torch.zeros((3, 3))
    Ay[0, 0] = torch.cos(thetaY)
    Ay[0, 2] = torch.sin(thetaY)
    Ay[2, 0] = -torch.sin(thetaY)
    Ay[2, 2] = torch.cos(thetaY)
    Ay[1, 1] = torch.tensor(1.0)

    Az = torch.zeros((3, 3))
    Az[0, 0] = torch.cos(thetaZ)
    Az[0, 1] = -torch.sin(thetaZ)
    Az[1, 0] = torch.sin(thetaZ)
    Az[1, 1] = torch.cos(thetaZ)
    Az[2, 2] = torch.tensor(1.0)

    R = Ax @ Ay @ Az
    print("R: ", R)
    return R


def get3DRotMatrixAxis(theta, ax=None):
    # ax = vector on unit sphere (e.g.
    # theta is numpy number in radians
    if ax is None:
        ax = np.zeros((3, 1))
        ax[1] = 1.0
    K = np.zeros((3, 3))
    K[0, 1] = -ax[2]
    K[0, 2] = ax[1]
    K[1, 0] = ax[2]
    K[2, 0] = -ax[1]
    K[1, 2] = -ax[0]
    K[2, 1] = ax[0]
    # A = np.cross(-theta*ax,np.identity(3),axisa=0,axisb=0)
    R = sp.linalg.expm(theta * K)
    return torch.from_numpy(R)


def rescaleData(S, T):
    """
    Rescales Data to be in bounding box of [0,1]^d
    Returns rescaled data and rescaling coefficient (one coefficient for each direction)
    """

    X = torch.cat((S, T))
    m = torch.min(X, axis=0).values
    rang = torch.max(X, axis=0).values
    print("min and max originally")
    print(m.detach())
    print(rang.detach())
    rang = rang - m
    s = torch.max(rang)
    Stilde = (S - m) * (1.0 / s)
    Ttilde = (T - m) * (1.0 / s)

    return Stilde, Ttilde, s, m


def rescaleDataList(Slist):
    X = torch.cat((Slist[0], Slist[1]))
    for i in range(2, len(Slist)):
        X = torch.cat((X, Slist[i]))
    m = torch.min(X, axis=0).values
    rang = torch.max(X, axis=0).values - m
    s = torch.max(rang)
    Stilde = []
    for i in range(len(Slist)):
        Stilde.append((Slist[i] - m) * (1.0 / s))
    return Stilde, s, m


def resizeData(Xtilde, s, m):
    """
    Inverse of rescaleData. Takes coefficient of scaling s and rescales it appropriately.
    """
    X = (Xtilde * s) + m
    return X

def rescaleDataSM(X, s, m):
    '''
    Rescales data with given s, m found from rescaling based on S, T
    '''
    Xtilde = (X - m) * (1.0 / s)
    return Xtilde

