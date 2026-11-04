TRANSFORMS = {
    "translation":   ("translation",),
    "rotation":      ("rotation",),
    "scale":         ("scale",),
    "rigid":         ("rotation", "translation"),
    "affine":        ("rotation", "translation", "scale"),
    "diffeo":        ("diffeo",),
    "rigid+diffeo":  ("rotation", "translation", "diffeo"),
    "affine+diffeo": ("rotation", "translation", "scale", "diffeo"),
}