from enum import Enum

from src.fem_buckling.mesh import Element, Node


class BoundaryConditionType(Enum):
    FREE = 0
    FIXED = 1


class SpringDirection(Enum):
    AXIAL = 0
    TRANSVERSE = 1
    ROTATIONAL = 2


class BoundaryCondition:
    def __init__(
        self, node: Node, bc_axial: int, bc_transverse: int, bc_rotational: int
    ):
        self.node = node
        self.bc_axial = BoundaryConditionType(bc_axial)
        self.bc_transverse = BoundaryConditionType(bc_transverse)
        self.bc_rotational = BoundaryConditionType(bc_rotational)

    def __repr__(self):
        return f"BoundaryCondition(bc_axial={self.bc_axial}, bc_transverse={self.bc_transverse}, bc_rotational={self.bc_rotational})"


class NodalLoad:
    def __init__(self, node: Node, force: float):
        self.node = node
        self.force = force

    def __repr__(self):
        return f"NodalLoad(node={self.node}, force={self.force})"


class ElementLoad:
    def __init__(self, element: Element, force: float):
        self.element = element
        self.force = force

    def __repr__(self):
        return f"ElementLoad(element={self.element}, force={self.force})"


class SpringSupport:
    def __init__(self, node: Node, direction: SpringDirection, stiffness: float):
        self.node = node
        self.direction = direction
        self.stiffness = stiffness

    def __repr__(self):
        return f"SpringSupport(node={self.node}, direction={self.direction}, stiffness={self.stiffness})"


class LoadCase:
    def __init__(
        self,
        boundary_conditions: list[BoundaryCondition],
        nodal_loads: list[NodalLoad],
        element_loads: list[ElementLoad],
        spring_supports: list[SpringSupport],
    ):
        self.boundary_conditions = boundary_conditions
        self.nodal_loads = nodal_loads
        self.element_loads = element_loads
        self.spring_supports = spring_supports
