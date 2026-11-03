"""
User-facing deformation model.

    deformation = Deformation("rigid", Stilde, cA=1.0, cT=1.0)
    deformation = Deformation("all-mixed", Stilde, sigma=[0.2, 0.1, 0.05], cNR=10.0)
    deformation = Deformation("non-rigid+translation", Stilde, sigma=[0.1])

``deformation.hamiltonian`` is the Hamiltonian of the selected mode; it owns
the shooting methods.  ``deformation.weight`` is the weight (gamma) of the
Hamiltonian in the total loss.
"""

from xmodmap2.deformation.hamiltonian import (
    NonRigidHamiltonian,
    TranslationHamiltonian,
    RotationHamiltonian,
    ScalingHamiltonian,
    GeneralLinearHamiltonian,
    MixedHamiltonian,
)

# Elementary modes, in the order they are summed.
COMPONENTS = ("non-rigid", "rotation", "translation", "scaling", "linear")

# Named combinations ("+" sums modes that share the same momenta).
ALIASES = {
    "rigid": ["rotation", "translation"],
    "similarity": ["rotation", "translation", "scaling"],   # = linear part of xmodmap
    "affine": ["linear", "translation"],                     # general d x d matrix + translation
    "all-mixed": ["non-rigid", "rotation", "translation", "scaling"],  # = xmodmap
}

_SYNONYMS = {
    "nonrigid": "non-rigid",
    "lddmm": "non-rigid",
    "diffeomorphic": "non-rigid",
    "rotations": "rotation",
    "translations": "translation",
    "scale": "scaling",
    "general-linear": "linear",
    "gl": "linear",
    "mixed": "all-mixed",
    "all": "all-mixed",
    "allmixed": "all-mixed",
}


def parse_mode(mode):
    """
    Turn a mode string into the ordered list of elementary modes.

    >>> parse_mode("rigid")
    ['rotation', 'translation']
    >>> parse_mode("non-rigid + rigid")
    ['non-rigid', 'rotation', 'translation']
    """
    names = []
    for token in str(mode).lower().replace("_", "-").replace(" ", "").split("+"):
        token = _SYNONYMS.get(token, token)
        if token in ALIASES:
            names += ALIASES[token]
        elif token in COMPONENTS:
            names.append(token)
        else:
            raise ValueError(f"Unknown deformation mode '{token}' in '{mode}'. {available_modes()}")
    unique = [n for n in COMPONENTS if n in names]  # canonical order, no duplicates
    if "linear" in unique and ({"rotation", "scaling"} & set(unique)):
        raise ValueError(
            "'linear' (or 'affine') already contains rotations and scalings; "
            "do not combine it with 'rotation', 'scaling', 'rigid' or 'similarity'."
        )
    return unique


def available_modes():
    return (
        f"Elementary modes: {', '.join(COMPONENTS)}. "
        f"Aliases: {', '.join(f'{k} = {chr(43).join(v)}' for k, v in ALIASES.items())}. "
        "Combine with '+', e.g. 'non-rigid+rigid'."
    )


class Deformation:
    """
    Args:
        mode: string selecting the deformation, see ``available_modes()``.
        Stilde: N x d rescaled source positions.
        sigma: kernel scales of the non-rigid mode (required if it is used).
        cA: cost of the rotation mode.
        cT: cost of the translation mode.
        cScale: cost of the scaling mode (hard-coded to 1 in xmodmap).
        cNR: cost of the non-rigid mode (named ``cS`` in xmodmap).
        cL: cost of the general linear mode (used by "linear" / "affine").
        dimEff: effective dimension of the data (default: d).
        single: in 3D with dimEff = 3, one isotropic scaling factor instead of xy + z.
        kernel: "gauss" or "laplace" (non-rigid mode).
        normalize_kernel: rescale each kernel scale by its data-dependent normalization.
        nt: number of time steps of the numerical integration (also the number of
            stored states of a closed-form trajectory).
        integrator: "ralston" (default, as xmodmap), "rk4" or "euler".
        weight: gamma, weight of the Hamiltonian in the total loss.

    Coefficients act as inverse gains: a mode with cost c moves like 1/c for a
    given momentum and costs c/2 |control|^2.  Only their ratios change the
    optimal deformation; a common factor is equivalent to rescaling ``weight``.
    """

    def __init__(
        self,
        mode,
        Stilde,
        sigma=None,
        cA=1.0,
        cT=1.0,
        cScale=1.0,
        cNR=10.0,
        cL=1.0,
        dimEff=None,
        single=False,
        kernel="gauss",
        normalize_kernel=True,
        nt=10,
        integrator="ralston",
        weight=1.0,
        **kwargs,
    ):
        if "cS" in kwargs:
            raise TypeError(
                "'cS' was the non-rigid coefficient of xmodmap; it is called 'cNR' in xmodmap2. "
                "(Scaling now has its own coefficient 'cScale', which xmodmap fixed to 1.)"
            )
        if kwargs:
            raise TypeError(f"Unexpected arguments: {sorted(kwargs)}")

        self.mode = mode
        self.d = Stilde.shape[1]
        self.dimEff = self.d if dimEff is None else dimEff
        self.single = single
        self.weight = weight
        self.coefficients = {"cA": cA, "cT": cT, "cScale": cScale, "cNR": cNR, "cL": cL}
        self.component_names = parse_mode(mode)

        if "non-rigid" in self.component_names and sigma is None:
            raise ValueError("The non-rigid mode needs the kernel scales 'sigma'.")
        self.sigma = sigma

        common = {"nt": nt, "integrator": integrator}
        builders = {
            "non-rigid": lambda: NonRigidHamiltonian(
                sigma, Stilde, cNR=cNR, kernel=kernel, normalize=normalize_kernel, **common
            ),
            "rotation": lambda: RotationHamiltonian(self.d, cA=cA, **common),
            "translation": lambda: TranslationHamiltonian(self.d, cT=cT, **common),
            "scaling": lambda: ScalingHamiltonian(
                self.d, cScale=cScale, dimEff=self.dimEff, single=single, **common
            ),
            "linear": lambda: GeneralLinearHamiltonian(self.d, cL=cL, **common),
        }
        components = [builders[n]() for n in self.component_names]
        self._hamiltonian = (
            components[0] if len(components) == 1 else MixedHamiltonian(components, **common)
        )

    # ------------------------------------------------------------------
    @property
    def hamiltonian(self):
        """Hamiltonian of the selected mode (owns the shooting methods)."""
        return self._hamiltonian

    @property
    def components(self):
        h = self._hamiltonian
        return h.components if isinstance(h, MixedHamiltonian) else [h]

    @property
    def has_closed_form(self):
        return self._hamiltonian.has_closed_form

    # ------------------------------------------------------------------
    def energy(self, px, pw, qx, qw):
        """Weighted Hamiltonian, the regularization term of the loss (gamma * H)."""
        return self.weight * self._hamiltonian(px, pw, qx, qw)

    def energies(self, px, pw, qx, qw):
        """Unweighted contribution of each elementary mode to H."""
        return self._hamiltonian.energies(px, pw, qx, qw)

    def shoot(self, px, pw, qx, qw, X=None, wX=None, nt=None):
        """Trajectory (list of states at t = k/nt), see Hamiltonian.shoot."""
        return self._hamiltonian.shoot(px, pw, qx, qw, X=X, wX=wX, nt=nt)

    def __call__(self, px, pw, qx, qw, X=None, wX=None):
        """Final state (t = 1) only; cheapest call for the loss."""
        return self._hamiltonian.endpoint(px, pw, qx, qw, X=X, wX=wX)

    def transform(self, px, pw, qx, qw, X, wX=None):
        """Push points X (and masses wX) forward by the deformation; returns (X1, wX1)."""
        return self(px, pw, qx, qw, X=X, wX=wX)[4:]

    def inverse_transform(self, px1, pw1, qx1, qw1, X, wX=None, nt=None):
        """Pull points X back with the inverse deformation, starting from the FINAL state."""
        return self._hamiltonian.shoot_backward(px1, pw1, qx1, qw1, X, wX=wX, nt=nt)[-1][4:]

    def flow_map(self, px, pw, qx, qw, t=1.0):
        """(Phi, b) with x -> Phi x + b, for modes with a closed-form flow."""
        if not self.has_closed_form:
            raise NotImplementedError(f"Mode '{self.mode}' has no closed-form flow map.")
        return self._hamiltonian.flow_map(px, pw, qx, qw, t=t)

    # ------------------------------------------------------------------
    def get_params(self):
        return {
            "mode": self.mode,
            "components": self.component_names,
            "sigma": self.sigma,
            **self.coefficients,
            "dimEff": self.dimEff,
            "single": self.single,
            "weight": self.weight,
            "hamiltonian": self._hamiltonian.get_params(),
        }

    def __repr__(self):
        return (
            f"Deformation(mode='{self.mode}', components={self.component_names}, "
            f"closed_form={self.has_closed_form}, weight={self.weight}, {self.coefficients})"
        )
