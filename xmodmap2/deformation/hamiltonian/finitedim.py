"""
Finite-dimensional (rigid / affine) deformation modes.

Every mode here produces a velocity field of the form

    v(x) = L (x - c) + tau,      c = sum_i qw_i qx_i / sum_i qw_i   (barycenter),

with div v = tr(L), so masses evolve as dqw_i/dt = tr(L) qw_i.

With M = sum_i px_i (qx_i - c)^T  (d x d),  P = sum_i px_i,  Pi = sum_i pw_i qw_i,
the pairing between momenta and velocity is  <L, M + Pi I>_F + <tau, P>.
Maximizing it minus the quadratic cost of each mode gives the controls:

    translation   tau = P / cT                                H = cT/2 |tau|^2
    rotation      A   = (M - M^T) / (2 cA)                     H = cA/2 |A|_F^2
    scaling       B   = sum_g a_g E_g,                         H = cScale/2 |B|_F^2
                  a_g = (<E_g, M> + k_g Pi) / (cScale k_g)
                  (E_g: 0/1 diagonal mask of the axis group g, k_g = |g|)
    linear        L   = (M + Pi I) / cL   (any d x d matrix)   H = cL/2 |L|_F^2

Closed-form flows (derivation in xmodmap2/REPORT.md).  Along the geodesic:
    * P is conserved, so tau is constant and the barycenter moves as c(t) = c0 + t tau;
    * y_i = qx_i - c obeys dy_i/dt = L(t) y_i, so y_i(t) = Phi(t) y_i(0) with dPhi/dt = L Phi;
    * L(t) satisfies the Euler-Arnold equation  dL/dt = [L, L^T]  restricted to the mode:
        - translation, rotation, scaling, rotation + isotropic scaling: L is constant,
          Phi(t) = expm(t L0)                                            ("constant")
        - general linear: L(t) = K0 + e^{2tK0} S0 e^{-2tK0}  (K0/S0: skew/sym part of L0),
          Phi(t) = expm(2 t K0) expm(t L0^T)                             ("gl")
Then, with W = sum_i qw_i(0) and lam_i = qw_i(0) / W,

    qx_i(t) = c0 + t tau + Phi(t) (qx_i(0) - c0)
    qw_i(t) = det Phi(t) qw_i(0)                       (det Phi(t) = exp(t tr L0))
    px_i(t) = lam_i P + Phi(t)^{-T} (px_i(0) - lam_i P)
    pw_i(t) = [pw_i(0) + P^T (Phi(t) - I)(qx_i(0) - c0) / W] / det Phi(t)
"""

import torch

from xmodmap2.deformation.hamiltonian.base import Hamiltonian


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def moments(px, pw, qx, qw):
    """Return (M, P, Pi, c) as defined in the module docstring."""
    c = (qw * qx).sum(dim=0) / qw.sum()
    M = px.T @ (qx - c)
    return M, px.sum(dim=0), (pw * qw).sum(), c


def flow_matrices(kind, L, t):
    """Phi(t) and Phi(t)^{-1} for the two solvable Euler-Arnold dynamics."""
    if kind == "constant":
        return torch.linalg.matrix_exp(t * L), torch.linalg.matrix_exp(-t * L)
    if kind == "gl":
        K = 0.5 * (L - L.T)
        Phi = torch.linalg.matrix_exp(2 * t * K) @ torch.linalg.matrix_exp(t * L.T)
        Phi_inv = torch.linalg.matrix_exp(-t * L.T) @ torch.linalg.matrix_exp(-2 * t * K)
        return Phi, Phi_inv
    raise ValueError(f"No closed-form flow of kind '{kind}'.")


def linear_flow_trajectory(times, kind, L, tau, px, pw, qx, qw, X=None, wX=None):
    """
    Exact solution of Hamilton's equations for v(x) = L(t)(x - c(t)) + tau,
    evaluated at each t in ``times`` (see module docstring).
    """
    eye = torch.eye(qx.shape[1], dtype=qx.dtype, device=qx.device)
    W = qw.sum()
    c = (qw * qx).sum(dim=0) / W
    y = qx - c
    P = px.sum(dim=0)
    lam = qw / W
    r = px - lam * P
    trL = torch.trace(L)

    traj = []
    for t in times:
        Phi, Phi_inv = flow_matrices(kind, L, t)
        det = torch.exp(t * trL)
        ct = c + t * tau
        state = (
            lam * P + r @ Phi_inv,                                    # px(t)
            (pw + ((y @ (Phi - eye).T) @ P)[:, None] / W) / det,      # pw(t)
            ct + y @ Phi.T,                                           # qx(t)
            qw * det,                                                 # qw(t)
        )
        if X is not None:
            state += (ct + (X - c) @ Phi.T, wX * det)
        traj.append(state)
    return traj


def scaling_groups(d, dimEff=None, single=False):
    """
    Groups of axes that share one scaling factor (same convention as
    xmodmap.deformation.control.affine.getATauAlpha):
        d = 2                       -> one isotropic factor        [[0, 1]]
        d = 3, dimEff = 2           -> xy factor, z not scaled     [[0, 1]]
        d = 3, dimEff = 3, single   -> one isotropic factor        [[0, 1, 2]]
        d = 3, dimEff = 3           -> xy factor and z factor      [[0, 1], [2]]
    """
    dimEff = d if dimEff is None else dimEff
    if d == 2:
        return [[0, 1]]
    if d == 3 and dimEff == 2:
        return [[0, 1]]
    if single or d != 3:
        return [list(range(d))]
    return [[0, 1], [2]]


# ---------------------------------------------------------------------------
# Mother class of the finite-dimensional modes
# ---------------------------------------------------------------------------

class FiniteDimHamiltonian(Hamiltonian):
    """
    Mode with velocity v(x) = L (x - c) + tau.  Children implement
    ``controls`` (the maximizer (L, tau) of the pairing minus the cost) and
    ``control_energy``.  The flow is known in closed form.
    """

    name = "finite-dim"
    has_closed_form = True
    flow_kind = "constant"  # how L evolves along the geodesic: "constant" or "gl"

    def controls(self, px, pw, qx, qw):
        """Optimal (L, tau): L is d x d, tau has shape (d,)."""
        raise NotImplementedError

    def control_energy(self, L, tau):
        raise NotImplementedError

    def _zeros_L(self, ref):
        return torch.zeros(self.d, self.d, dtype=ref.dtype, device=ref.device)

    def _zeros_tau(self, ref):
        return torch.zeros(self.d, dtype=ref.dtype, device=ref.device)

    def H(self, px, pw, qx, qw):
        return self.control_energy(*self.controls(px, pw, qx, qw))

    def velocity(self, px, pw, qx, qw, x):
        L, tau = self.controls(px, pw, qx, qw)
        c = (qw * qx).sum(dim=0) / qw.sum()
        return (x - c) @ L.T + tau

    def divergence(self, px, pw, qx, qw, x):
        L, _ = self.controls(px, pw, qx, qw)
        return torch.trace(L) * torch.ones(x.shape[0], 1, dtype=x.dtype, device=x.device)

    def flow_trajectory(self, times, px, pw, qx, qw, X=None, wX=None):
        L, tau = self.controls(px, pw, qx, qw)
        return linear_flow_trajectory(times, self.flow_kind, L, tau, px, pw, qx, qw, X, wX)

    def flow_map(self, px, pw, qx, qw, t=1.0):
        """
        The deformation at time t as an explicit affine map  x -> Phi x + b
        (returned as (Phi, b)).  Apply it to row-stacked points as X @ Phi.T + b.
        """
        L, tau = self.controls(px, pw, qx, qw)
        return explicit_affine_map(self.flow_kind, L, tau, qx, qw, t)


def explicit_affine_map(kind, L, tau, qx, qw, t=1.0):
    c = (qw * qx).sum(dim=0) / qw.sum()
    Phi, _ = flow_matrices(kind, L, t)
    return Phi, c + t * tau - Phi @ c


# ---------------------------------------------------------------------------
# The modes
# ---------------------------------------------------------------------------

class TranslationHamiltonian(FiniteDimHamiltonian):
    name = "translation"

    def __init__(self, d, cT=1.0, nt=10, integrator="ralston"):
        super().__init__(d, nt=nt, integrator=integrator)
        self.cT = cT

    def controls(self, px, pw, qx, qw):
        return self._zeros_L(px), px.sum(dim=0) / self.cT

    def control_energy(self, L, tau):
        return (self.cT / 2.0) * (tau * tau).sum()

    def get_params(self):
        return {**super().get_params(), "cT": self.cT}


class RotationHamiltonian(FiniteDimHamiltonian):
    """Infinitesimal rotation A (skew-symmetric) about the barycenter."""

    name = "rotation"

    def __init__(self, d, cA=1.0, nt=10, integrator="ralston"):
        super().__init__(d, nt=nt, integrator=integrator)
        self.cA = cA

    def controls(self, px, pw, qx, qw):
        M, _, _, _ = moments(px, pw, qx, qw)
        return (M - M.T) / (2.0 * self.cA), self._zeros_tau(px)

    def control_energy(self, L, tau):
        return (self.cA / 2.0) * (L * L).sum()

    def get_params(self):
        return {**super().get_params(), "cA": self.cA}


class ScalingHamiltonian(FiniteDimHamiltonian):
    """Diagonal scaling about the barycenter; one factor per group of axes."""

    name = "scaling"

    def __init__(self, d, cScale=1.0, dimEff=None, single=False, nt=10, integrator="ralston"):
        super().__init__(d, nt=nt, integrator=integrator)
        self.cScale = cScale
        self.dimEff = d if dimEff is None else dimEff
        self.single = single
        self.groups = scaling_groups(d, self.dimEff, single)

    @property
    def isotropic(self):
        """True if a single factor scales all d axes (then it commutes with any rotation)."""
        return len(self.groups) == 1 and len(self.groups[0]) == self.d

    def controls(self, px, pw, qx, qw):
        M, _, Pi, _ = moments(px, pw, qx, qw)
        diagM = torch.diagonal(M)
        b = torch.zeros(self.d, dtype=px.dtype, device=px.device)
        for g in self.groups:
            k = len(g)
            a = (diagM[g].sum() + k * Pi) / (self.cScale * k)
            b = b.index_add(0, torch.tensor(g, device=px.device), a.expand(k))
        return torch.diag(b), self._zeros_tau(px)

    def control_energy(self, L, tau):
        return (self.cScale / 2.0) * (L * L).sum()

    def get_params(self):
        return {**super().get_params(), "cScale": self.cScale, "dimEff": self.dimEff,
                "single": self.single, "groups": self.groups}


class GeneralLinearHamiltonian(FiniteDimHamiltonian):
    """Any d x d matrix L about the barycenter (rotation + scaling + shear)."""

    name = "linear"
    flow_kind = "gl"

    def __init__(self, d, cL=1.0, nt=10, integrator="ralston"):
        super().__init__(d, nt=nt, integrator=integrator)
        self.cL = cL

    def controls(self, px, pw, qx, qw):
        M, _, Pi, _ = moments(px, pw, qx, qw)
        eye = torch.eye(self.d, dtype=px.dtype, device=px.device)
        return (M + Pi * eye) / self.cL, self._zeros_tau(px)

    def control_energy(self, L, tau):
        return (self.cL / 2.0) * (L * L).sum()

    def get_params(self):
        return {**super().get_params(), "cL": self.cL}
