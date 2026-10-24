"""
Attacks package.
"""

from flaas.attacks.attacks import (
    BaseAttack,
    SignFlipAttack,
    GaussianNoiseAttack,
    ScaledUpdateAttack,
    LabelFlippedDataset,
)

__all__ = [
    "BaseAttack",
    "SignFlipAttack",
    "GaussianNoiseAttack",
    "ScaledUpdateAttack",
    "LabelFlippedDataset",
]
