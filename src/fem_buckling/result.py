from dataclasses import dataclass

import numpy as np

from src.fem_buckling.mesh import Element, Node


@dataclass(frozen=True)
class ReactionForce:
    node: Node
    force: float


@dataclass(frozen=True)
class AxialForce:
    element: Element
    force: float


class AxialResults:
    def __init__(
        self,
        axial_displacements: np.ndarray,
        reaction_forces: list[ReactionForce],
        axial_forces: list[AxialForce],
    ):
        self.axial_displacements = axial_displacements
        self.reaction_forces = reaction_forces
        self.axial_forces = axial_forces
