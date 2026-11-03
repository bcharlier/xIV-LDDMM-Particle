"""
Sum of several modes sharing the same momenta.

If the velocity field is v = v_1 + ... + v_K with independent controls and
separable costs, the reduced Hamiltonian is the sum of the reduced
Hamiltonians of the modes.  This is exactly how xmodmap couples the non-rigid,
rotation, translation and scaling modes (mode "all-mixed").

All modes see the same momentum p, so the share of the motion taken by mode k
scales like 1/c_k.  Only the ratios of the coefficients matter: multiplying
every c_k by lambda gives the same optimal deformation as multiplying the
Hamiltonian weight (gamma) by lambda.
"""

import torch

from xmodmap2.deformation.hamiltonian.base import Hamiltonian
from xmodmap2.deformation.hamiltonian.finitedim import (
    FiniteDimHamiltonian,
    ScalingHamiltonian,
    linear_flow_trajectory,
    explicit_affine_map,
)


def mixed_flow_kind(components):
    """
    Closed-form kind of a sum of modes, or None if it has to be integrated.

    Translation never interacts with the linear part.  The linear part keeps
    a constant L if it is a rotation, a (possibly anisotropic) scaling, or a
    rotation plus an isotropic scaling (they commute).  A general linear mode
    alone has the "gl" flow.  Anything else (non-rigid, rotation plus
    anisotropic scaling, linear plus another linear mode) has no closed form.
    """
    if not all(isinstance(c, FiniteDimHamiltonian) for c in components):
        return None
    kinds = {c.name for c in components} - {"translation"}
    if kinds <= {"rotation"} or kinds == {"scaling"}:
        return "constant"
    if kinds == {"rotation", "scaling"} and all(
        c.isotropic for c in components if isinstance(c, ScalingHamiltonian)
    ):
        return "constant"
    if kinds == {"linear"} and sum(c.name == "linear" for c in components) == 1:
        return "gl"
    return None


class MixedHamiltonian(Hamiltonian):
    name = "mixed"

    def __init__(self, components, nt=10, integrator="ralston"):
        if len(components) == 0:
            raise ValueError("MixedHamiltonian needs at least one component.")
        super().__init__(components[0].d, nt=nt, integrator=integrator)
        self.components = list(components)
        self.flow_kind = mixed_flow_kind(self.components)
        self.has_closed_form = self.flow_kind is not None
        self.name = "+".join(c.name for c in self.components)

    def H(self, px, pw, qx, qw):
        return sum(c.H(px, pw, qx, qw) for c in self.components)

    def energies(self, px, pw, qx, qw):
        return {c.name: c.H(px, pw, qx, qw) for c in self.components}

    def velocity(self, px, pw, qx, qw, x):
        return sum(c.velocity(px, pw, qx, qw, x) for c in self.components)

    def divergence(self, px, pw, qx, qw, x):
        return sum(c.divergence(px, pw, qx, qw, x) for c in self.components)

    # --- closed form (only when every component is finite-dimensional) ---
    def controls(self, px, pw, qx, qw):
        if not all(isinstance(c, FiniteDimHamiltonian) for c in self.components):
            raise NotImplementedError("controls() is only defined for finite-dimensional modes.")
        Ls, taus = zip(*(c.controls(px, pw, qx, qw) for c in self.components))
        return sum(Ls), sum(taus)

    def flow_trajectory(self, times, px, pw, qx, qw, X=None, wX=None):
        if not self.has_closed_form:
            return super().flow_trajectory(times, px, pw, qx, qw, X, wX)
        L, tau = self.controls(px, pw, qx, qw)
        return linear_flow_trajectory(times, self.flow_kind, L, tau, px, pw, qx, qw, X, wX)

    def flow_map(self, px, pw, qx, qw, t=1.0):
        if not self.has_closed_form:
            raise NotImplementedError(f"Mode '{self.name}' has no closed-form flow map.")
        L, tau = self.controls(px, pw, qx, qw)
        return explicit_affine_map(self.flow_kind, L, tau, qx, qw, t)

    def get_params(self):
        return {
            **super().get_params(),
            "closed_form": self.flow_kind,
            "components": [c.get_params() for c in self.components],
        }
