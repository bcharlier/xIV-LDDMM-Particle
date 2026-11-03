import torch

from pykeops.torch import Vi, Vj

from xmodmap.deformation.control.nonRigid import getU, getUdiv
from xmodmap.deformation.control.affine import getA, getTau, getAlpha, getBarycenter
import xmodmap.deformation.Ucoeff as uc
from xmodmap.deformation.transforms import TRANSFORMS

class Control:
    """One term of the Hamiltonian and its velocity. Subclasses are added together by Hamiltonian."""
    uses_pw = False  

    def energy(self, px, pw, qx, qw):
        raise NotImplementedError

    def velocity(self, x, w, px, pw, qx, qw):
        """Returns the point velocity (M x d) and the weight rate (M x 1)."""
        raise NotImplementedError


class GlobalControl(Control):
    """Parent of the translation, rotation and scale controls."""
    def __init__(self, cA, cT, dimEff, single):
        self.cA, self.cT, self.dimEff, self.single = cA, cT, dimEff, single


class TranslationControl(GlobalControl):
    def energy(self, px, pw, qx, qw):
        tau = getTau(px, cT=self.cT)
        return (self.cT / 2) * (tau * tau).sum()

    def velocity(self, x, w, px, pw, qx, qw):
        return getTau(px, cT=self.cT).expand_as(x), torch.zeros_like(w)


class RotationControl(GlobalControl):
    def energy(self, px, pw, qx, qw):
        A = getA(px, qx, qw, cA=self.cA)
        return (self.cA / 2) * (A * A).sum()

    def velocity(self, x, w, px, pw, qx, qw):
        A = getA(px, qx, qw, cA=self.cA)
        return (x - getBarycenter(qx, qw)) @ A.T, torch.zeros_like(w)


class ScaleControl(GlobalControl):
    uses_pw = True

    def energy(self, px, pw, qx, qw):
        Alpha = getAlpha(px, qx, pw, qw, dimEff=self.dimEff, single=self.single)
        return 0.5 * (Alpha * Alpha).sum()

    def velocity(self, x, w, px, pw, qx, qw):
        Alpha = getAlpha(px, qx, pw, qw, dimEff=self.dimEff, single=self.single)
        return (x - getBarycenter(qx, qw)) @ Alpha, Alpha.sum() * w


class NonRigidControl(Control):
    uses_pw = True

    def __init__(self, sigma, Stilde, cS):
        d = Stilde.shape[1]
        uCoeff = uc.Ucoeff(sigma, Stilde, cS).uCoeff
        self.K0 = Hamiltonian.GaussKernelHamiltonian(sigma, d, uCoeff)
        self.U = getU(sigma, d, uCoeff)
        self.Udiv = getUdiv(sigma, d, uCoeff)

    def energy(self, px, pw, qx, qw):
        wpq = pw * qw
        return self.K0(qx, qx, px, px, wpq, wpq).sum()

    def velocity(self, x, w, px, pw, qx, qw):
        wpq = pw * qw
        return self.U(x, qx, px, wpq), self.Udiv(x, qx, px, wpq) * w
    
class Hamiltonian:
    def __init__(self, sigma, Stilde, cA=1.0, cS=10.0, cT=1.0, dimEff=3, single=False,
        transform="affine+diffeo"):
        self.d = Stilde.shape[1]
        self.numS = Stilde.shape[0]
        self.sigma, self.cA, self.cS, self.cT = sigma, cA, cS, cT
        self.dimEff, self.single, self.transform = dimEff, single, transform
        self.weight = 1.0

        self.components = [self._build_component(name, Stilde) for name in TRANSFORMS[transform]]

    def _build_component(self, name, Stilde):
        global_args = (self.cA, self.cT, self.dimEff, self.single)
        if name == "translation":
            return TranslationControl(*global_args)
        if name == "rotation":
            return RotationControl(*global_args)
        if name == "scale":
            return ScaleControl(*global_args)
        if name == "diffeo":
            return NonRigidControl(self.sigma, Stilde, self.cS)
        raise ValueError(f"Unknown component: {name!r}")

    @property
    def uses_pw(self):
        return any(c.uses_pw for c in self.components)

    def get_params(self):
        return {
            "sigma": self.sigma, "cA": self.cA, "cS": self.cS, "cT": self.cT,
            "dimEff": self.dimEff, "single": self.single,
            "weight": self.weight, "transform": self.transform,
        }

    def H(self, px, pw, qx, qw):
        return sum(c.energy(px, pw, qx, qw) for c in self.components)

    def velocity(self, x, w, px, pw, qx, qw):
        vs = [c.velocity(x, w, px, pw, qx, qw) for c in self.components]
        return sum(v[0] for v in vs), sum(v[1] for v in vs)

    def _grad_H(self, px, pw, qx, qw):
        inputs = (px, pw, qx, qw)
        grads = torch.autograd.grad(self.H(*inputs), inputs, create_graph=True, allow_unused=True)
        return [g if g is not None else torch.zeros_like(x) for g, x in zip(grads, inputs)]

    def forward_system(self, px, pw, qx, qw):
        Gpx, Gpw, Gqx, Gqw = self._grad_H(px, pw, qx, qw)
        return -Gqx, -Gqw, Gpx, Gpw

    def system_with_points(self, px, pw, qx, qw, x, w):
        Gpx, Gpw, Gqx, Gqw = self._grad_H(px, pw, qx, qw)
        Gx, Gw = self.velocity(x, w, px, pw, qx, qw)
        return -Gqx, -Gqw, Gpx, Gpw, Gx, Gw

    def __call__(self, px, pw, qx, qw):
        return self.H(px, pw, qx, qw)

    @staticmethod
    def GaussKernelHamiltonian(sigma, d, uCoeff):
        # TODO: could be place in getU
        qxO = Vi(0, d)
        qyO = Vj(1, d)
        px = Vi(2, d)
        py = Vj(3, d)
        wpxO = Vi(4, 1)
        wpyO = Vj(5, 1)

        for sInd in range(len(sigma)):
            sig = sigma[sInd]
            qx, qy, wpx, wpy = qxO / sig, qyO / sig, wpxO / sig, wpyO / sig
            D2 = qx.sqdist(qy)
            K = (-D2 * 0.5).exp()
            h = (
                    0.5 * (px * py).sum()
                    + wpy * ((qx - qy) * px).sum()
                    - (0.5) * wpx * wpy * (D2 - d)
            )  # 1/2 factor included here
            if sInd == 0:
                retVal = (1.0 / uCoeff[sInd]) * K * h  # normalize by N_sigma/(N sigma**2)
            else:
                retVal += (1.0 / uCoeff[sInd]) * K * h

        return retVal.sum_reduction(
            axis=1
        )  # (K*h).sum_reduction(axis=1) #,  h2, h3.sum_reduction(axis=1)

    @staticmethod
    def LaplaceKernelHamiltonian(sigma, d, uCoeff):
        # TODO: could be place in getU

        qxO = Vi(0, d)
        qyO = Vj(1, d)
        px = Vi(2, d)
        py = Vj(3, d)
        wpxO = Vi(4, 1)
        wpyO = Vj(5, 1)

        for sInd in range(len(sigma)):
            sig = sigma[sInd]
            qx, qy, wpx, wpy = qxO / sig, qyO / sig, wpxO / sig, wpyO / sig
            diff = qx - qy
            R2 = (diff * diff).sum()
            R = (R2 + 1e-8).sqrt()
            K = (-R).exp()

            h = (
                    0.5 * (px * py).sum()
                    + wpy * ((diff / R) * px).sum()
                    - 0.5 * wpx * wpy * (1.0 - (d - 1.0) / R)
            )  # 1/2 factor included here
            if sInd == 0:
                retVal = (1.0 / uCoeff[sInd]) * K * h  # normalize by N_sigma/(N sigma**2)
            else:
                retVal += (1.0 / uCoeff[sInd]) * K * h

        return retVal.sum_reduction(
            axis=1
        )  # (K*h).sum_reduction(axis=1) #,  h2, h3.sum_reduction(axis=1)