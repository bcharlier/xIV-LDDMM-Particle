# Must be defined before the imports below: Hamiltonian imports it from this package
TRANSFORMS = {
    "translation":   ("translation",),
    "rotation":      ("rotation",),
    "scale":         ("scale",),
    "rigid":         ("rotation", "translation"),
    "affine":        ("rotation", "translation", "scale"),
    "diffeo":        ("diffeo",),
    "scale+diffeo":  ("scale", "diffeo"),
    "rigid+diffeo":  ("rotation", "translation", "diffeo"),
    "affine+diffeo": ("rotation", "translation", "scale", "diffeo"),
}

from .Hamiltonian import Hamiltonian
from .Shooting import Shooting, ShootingGrid, ShootingBackwards
