# def getATauAlpha(px, qx, pw, qw, cA=1.0, cT=1.0, dimEff=3, single=False):
#     """
#     Compute Controls (A,Tau, Alpha) from momentums (px, qx, pw, qw)

#     Args:
#         `dimEff`: effective dimension of the data (warning, everything is hardcoded for 3D... even if effective dimension is 2)
#         `qx`: is the source point cloud (N x 3)
#         `px`: is the momentums (N x 3)
#         `qw`: is the source features mass (N x 1) (total amount of mass across feature space)
#         `pw`: is the feature mass momentums (N x 1)
#         `single`: if True if homoscedastic, if False x,y are homoscadastic ans z coordinate is different

#     Return:
#         `A`: is skew symetric matrix (infinitesimal rotation)
#         `tau`: is the (infinitesimal) translation vector
#         `Alpha`: is a diagonal matrix
#     """
#     xc = (qw * qx).sum(dim=0) / (qw.sum(dim=0))  # moving barycenters; should be 1 x 3
#     A = ((1.0 / (2.0 * cA)) * (px.T @ (qx - xc) - (qx - xc).T @ px))  # 3 x N * N x 3
#     tau = ((1.0 / cT) * (px.sum(dim=0)))

#     if qx.shape[-1] == 2:
#         alpha = (px * (qx - xc)).sum() / 2. + (pw * qw).sum()
#         Alpha = torch.diag(alpha.repeat(2))

#     elif dimEff == 2:
#         alpha = ((px[:,:2] * (qx[:,:2] - xc[:,:2])).sum() / 2. + (pw * qw).sum())
#         Alpha = torch.diag(torch.cat((alpha.view(1), alpha.view(1), torch.zeros(1))))  # always scale Z by 0

#     elif dimEff == 3 and single:
#         alpha = ((px * (qx - xc)).sum() / 3. + (pw * qw).sum())
#         Alpha = torch.diag(alpha.repeat(3))

#     else:
#         alpha_xy = ((px[:, 0:2] * (qx - xc)[:, 0:2]).sum() / 2. + (pw * qw).sum())
#         alpha_z = ((px[:, -1] * (qx - xc)[:, -1]).sum() + (pw * qw).sum())
#         Alpha = torch.diag(torch.cat((alpha_xy.view(1), alpha_xy.view(1), alpha_z.view(1))))

#     return A, tau, Alpha

import torch


def getBarycenter(qx, qw):
    """Weighted barycenter of the source points. Returns a vector of length d."""
    return (qw * qx).sum(dim=0) / qw.sum(dim=0)


def getA(px, qx, qw, cA=1.0):
    """Skew-symmetric matrix (infinitesimal rotation), d x d."""
    xc = getBarycenter(qx, qw)
    return (1.0 / (2.0 * cA)) * (px.T @ (qx - xc) - (qx - xc).T @ px)


def getTau(px, cT=1.0):
    """Infinitesimal translation vector, length d."""
    return (1.0 / cT) * px.sum(dim=0)


def getAlpha(px, qx, pw, qw, dimEff=3, single=False):
    """
    Diagonal scale matrix, d x d.

    d == 2 or dimEff == 2: one scale for x and y (z is not scaled when d == 3)
    dimEff == 3, single:   one scale for x, y and z
    dimEff == 3, otherwise: one scale for x and y, and a separate one for z
    """
    d = qx.shape[-1]
    r = qx - getBarycenter(qx, qw)
    mass = (pw * qw).sum()

    if d == 2:
        alpha = (px * r).sum() / 2. + mass
        return torch.diag(alpha.repeat(2))

    if dimEff == 2:
        alpha = (px[:, :2] * r[:, :2]).sum() / 2. + mass
        return torch.diag(torch.cat((alpha.view(1), alpha.view(1),
                                     torch.zeros(1, device=qx.device, dtype=qx.dtype))))

    if dimEff == 3 and single:
        alpha = (px * r).sum() / 3. + mass
        return torch.diag(alpha.repeat(3))

    alpha_xy = (px[:, :2] * r[:, :2]).sum() / 2. + mass
    alpha_z = (px[:, -1] * r[:, -1]).sum() + mass
    return torch.diag(torch.cat((alpha_xy.view(1), alpha_xy.view(1), alpha_z.view(1))))


def getATauAlpha(px, qx, pw, qw, cA=1.0, cT=1.0, dimEff=3, single=False,
                 rotation=True, translation=True, scale=True):
    d = qx.shape[-1]
    A = getA(px, qx, qw, cA=cA) if rotation else qx.new_zeros(d, d)
    tau = getTau(px, cT=cT) if translation else qx.new_zeros(d)
    Alpha = getAlpha(px, qx, pw, qw, dimEff=dimEff, single=single) if scale else qx.new_zeros(d, d)
    return A, tau, Alpha
