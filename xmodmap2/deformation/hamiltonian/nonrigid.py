"""
Non-rigid (diffeomorphic, LDDMM) deformation mode.
"""

from xmodmap2.deformation.hamiltonian.base import Hamiltonian
from xmodmap2.deformation.hamiltonian.kernels import (
    kernel_normalization,
    hamiltonian_kernel,
    velocity_kernel,
    divergence_kernel,
)


class NonRigidHamiltonian(Hamiltonian):
    """
    Multi-scale RKHS velocity field

        u(x) = sum_sigma w_sigma sum_j K_sigma(x, q_j) (p_j + wp_j grad_y K ...),

    with per-scale weight

        w_sigma = 1 / (cNR * N_sigma).

    * N_sigma (``self.normalization``) is computed once from the source and
      only rescales each scale to unit order (former ``Ucoeff`` without cS).
    * cNR (former ``cS``) is the explicit cost of the non-rigid mode:
      the velocity produced by a momentum p is proportional to 1/cNR and its
      kinetic energy cNR/2 |u|^2.  Increasing cNR makes the mapping stiffer;
      in a mixed mode it shifts the motion towards the other modes.

    Args:
        sigma: list of kernel scales (in the rescaled unit box).
        Stilde: N x d rescaled source positions (only used for N_sigma and d).
        cNR: non-rigid cost coefficient (former cS).
        kernel: "gauss" (default) or "laplace".
        normalize: if False, N_sigma = 1 for every scale.
    """

    name = "non-rigid"
    has_closed_form = False

    def __init__(self, sigma, Stilde, cNR=1.0, kernel="gauss", normalize=True, nt=10, integrator="ralston"):
        super().__init__(Stilde.shape[1], nt=nt, integrator=integrator)
        self.sigma = list(sigma)
        self.cNR = cNR
        self.kernel = kernel
        self.normalize = normalize
        self.normalization = (
            kernel_normalization(self.sigma, Stilde, kernel) if normalize else [1.0] * len(self.sigma)
        )
        self.scale_weights = [1.0 / (cNR * n) for n in self.normalization]

        self._H = hamiltonian_kernel(kernel, self.sigma, self.d, self.scale_weights)
        self._U = velocity_kernel(kernel, self.sigma, self.d, self.scale_weights)
        self._divU = divergence_kernel(kernel, self.sigma, self.d, self.scale_weights)

    def H(self, px, pw, qx, qw):
        wpq = pw * qw
        return self._H(qx, qx, px, px, wpq, wpq).sum()

    def velocity(self, px, pw, qx, qw, x):
        return self._U(x, qx, px, pw * qw)

    def divergence(self, px, pw, qx, qw, x):
        return self._divU(x, qx, px, pw * qw)

    def get_params(self):
        return {
            **super().get_params(),
            "sigma": self.sigma,
            "cNR": self.cNR,
            "kernel": self.kernel,
            "normalize": self.normalize,
            "normalization": self.normalization,
        }
