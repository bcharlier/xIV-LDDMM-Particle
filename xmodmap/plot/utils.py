"""Dimension-agnostic helpers shared by the 2D and 3D plotting functions."""
import numpy as np
import torch


def to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def feature_to_scalar(zeta):
    """Scalar value per point: argmax for (n, k) features (e.g. one-hot), identity for (n,) / (n, 1)."""
    zeta = to_numpy(zeta)
    if zeta.ndim == 1:
        return zeta.astype(float)
    if zeta.shape[1] == 1:
        return zeta[:, 0].astype(float)
    return zeta.argmax(axis=1).astype(float)


def weight_to_radius(w, scale):
    """Radius ~ sqrt(weight), so that the disc area (2D) follows the weight."""
    return scale * np.sqrt(np.clip(to_numpy(w).reshape(-1), 0, None))


def default_radius_scale(positions, weights, fraction=0.015):
    """Scale such that the heaviest particle has a radius of `fraction` of the data extent."""
    pts = np.concatenate(positions)
    extent = (pts.max(axis=0) - pts.min(axis=0)).max()
    wmax = max(w.max() for w in weights)
    return fraction * extent / np.sqrt(wmax)


def unpack_trajectory(trajectory, dims):
    """From a list of (px, pw, qx, qw) tuples, return the lists of positions (restricted to `dims`) and weights."""
    Q = [to_numpy(step[2])[:, list(dims)] for step in trajectory]
    W = [to_numpy(step[3]).reshape(-1) for step in trajectory]
    return Q, W
