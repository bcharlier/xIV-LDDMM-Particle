"""
KeOps formulas for the non-rigid (RKHS) part of the deformation.

The non-rigid velocity field is a weighted sum over scales sigma,

    u(x) = sum_sigma  w_sigma  sum_j  K_sigma(x, q_j) (p_j + wp_j (x - q_j) / sigma^2)    (Gauss)

with wp_j = pw_j * qw_j the weight momentum and

    w_sigma = 1 / (cNR * N_sigma).

Here N_sigma is the data-dependent scale normalization (see
``kernel_normalization``) and cNR is the user coefficient of the non-rigid
mode.  In ``xmodmap`` both were merged into ``uCoeff = cS * N_sigma``; here
they are kept apart so that cNR is an explicit, data-independent knob.
"""

from pykeops.torch import Vi, Vj

KERNELS = ("gauss", "laplace")
LAPLACE_EPS = 1e-8  # regularization of |x - y| at x == y (same value as xmodmap)


def _check_kernel(kernel):
    if kernel not in KERNELS:
        raise ValueError(f"Unknown kernel '{kernel}'. Choose one of {KERNELS}.")


# ---------------------------------------------------------------------------
# Scale normalization  (old xmodmap.deformation.Ucoeff, WITHOUT cS)
# ---------------------------------------------------------------------------

def _kernel_sum_single(sig, d, kernel):
    x, y = Vi(0, d) / sig, Vj(1, d) / sig
    if kernel == "gauss":
        K = (-x.sqdist(y) * 0.5).exp()
    else:
        K = (-(x.sqdist(y) + LAPLACE_EPS).sqrt()).exp()
    return K.sum_reduction(axis=1)


def kernel_normalization(sigma, Stilde, kernel="gauss"):
    """
    Data-dependent normalization of each scale of the non-rigid kernel:

        N_sigma = sum_{i,j} K_sigma(S_i, S_j) / (N^2 sigma^2).

    It is computed once on the (rescaled) source and makes the kinetic
    energy of every scale of the same order of magnitude.  It does NOT
    contain the user coefficient cNR (formerly cS).

    Returns:
        list of python floats, one per scale.
    """
    _check_kernel(kernel)
    N, d = Stilde.shape
    out = []
    for sig in sigma:
        Ksum = _kernel_sum_single(sig, d, kernel)(Stilde, Stilde).sum()
        out.append(float(Ksum / (N * N * sig * sig)))
    return out


# ---------------------------------------------------------------------------
# Hamiltonian  H_NR(p, q) = sum_sigma w_sigma sum_{i,j} K_sigma(q_i, q_j) h_ij
# ---------------------------------------------------------------------------

def hamiltonian_kernel(kernel, sigma, d, weights):
    """
    KeOps reduction returning, for each i, sum_j sum_sigma w_sigma K h_ij.
    Call it as ``f(qx, qx, px, px, wpq, wpq)`` with wpq = pw * qw.
    (Identical to xmodmap's GaussKernelHamiltonian / LaplaceKernelHamiltonian
    with 1 / uCoeff replaced by the explicit weights.)
    """
    _check_kernel(kernel)
    qxO, qyO = Vi(0, d), Vj(1, d)
    px, py = Vi(2, d), Vj(3, d)
    wpxO, wpyO = Vi(4, 1), Vj(5, 1)

    retVal = None
    for sig, w in zip(sigma, weights):
        qx, qy, wpx, wpy = qxO / sig, qyO / sig, wpxO / sig, wpyO / sig
        if kernel == "gauss":
            D2 = qx.sqdist(qy)
            K = (-D2 * 0.5).exp()
            h = (
                0.5 * (px * py).sum()
                + wpy * ((qx - qy) * px).sum()
                - 0.5 * wpx * wpy * (D2 - d)
            )
        else:
            diff = qx - qy
            R = ((diff * diff).sum() + LAPLACE_EPS).sqrt()
            K = (-R).exp()
            h = (
                0.5 * (px * py).sum()
                + wpy * ((diff / R) * px).sum()
                - 0.5 * wpx * wpy * (1.0 - (d - 1.0) / R)
            )
        term = w * K * h
        retVal = term if retVal is None else retVal + term
    return retVal.sum_reduction(axis=1)


# ---------------------------------------------------------------------------
# Velocity field u(x) and its divergence (old getU / getUdiv)
# ---------------------------------------------------------------------------

def velocity_kernel(kernel, sigma, d, weights):
    """
    u(x): call as ``f(x, qx, px, wpq)``, returns G x d.
    """
    _check_kernel(kernel)
    xO, qyO, py, wpyO = Vi(0, d), Vj(1, d), Vj(2, d), Vj(3, 1)
    retVal = None
    for sig, w in zip(sigma, weights):
        x, qy, wpy = xO / sig, qyO / sig, wpyO / sig
        if kernel == "gauss":
            K = (-x.sqdist(qy) * 0.5).exp()
            h = py + wpy * (x - qy)
        else:
            diff = x - qy
            R = ((diff * diff).sum() + LAPLACE_EPS).sqrt()
            K = (-R).exp()
            h = py + wpy * diff / R
        term = w * K * h
        retVal = term if retVal is None else retVal + term
    return retVal.sum_reduction(axis=1)


def divergence_kernel(kernel, sigma, d, weights):
    """
    div u(x): call as ``f(x, qx, px, wpq)``, returns G x 1.

    Gauss:   div = sum (w/sigma) K [ wp~ (d - |z|^2) - z.p ]
    Laplace: div = sum (w/sigma) K [ wp~ ((d-1)/r - 1) - z.p / r ]
    with z = (x - q)/sigma, r = |z|, wp~ = wp/sigma.
    """
    _check_kernel(kernel)
    xO, qyO, py, wpyO = Vi(0, d), Vj(1, d), Vj(2, d), Vj(3, 1)
    retVal = None
    for sig, w in zip(sigma, weights):
        x, qy, wpy = xO / sig, qyO / sig, wpyO / sig
        if kernel == "gauss":
            D2 = x.sqdist(qy)
            K = (-D2 * 0.5).exp()
            h = wpy * (d - D2) - ((x - qy) * py).sum()
        else:
            diff = x - qy
            R = ((diff * diff).sum() + LAPLACE_EPS).sqrt()
            K = (-R).exp()
            h = wpy * ((d - 1.0) / R - 1.0) - (diff * py).sum() / R
        term = (w / sig) * K * h
        retVal = term if retVal is None else retVal + term
    return retVal.sum_reduction(axis=1)
