from dataclasses import dataclass

import numpy as np

from src.fem_buckling.boundary_condition import BoundaryConditionType, SpringDirection
from src.fem_buckling.model_builder import Model


@dataclass(frozen=True)
class PartitionedSystem:
    K_ff: np.ndarray
    K_fc: np.ndarray
    K_cf: np.ndarray
    K_cc: np.ndarray
    f_f: np.ndarray
    f_c: np.ndarray
    free_dofs: np.ndarray
    constrained_dofs: np.ndarray


class AxialAssembler:
    def __init__(self, model: Model):
        self.model = model
        self.nodes = model.mesh.nodes
        self.elements = model.mesh.elements
        self.boundary_conditions = model.load_case.boundary_conditions
        self.nodal_loads = model.load_case.nodal_loads
        self.element_loads = model.load_case.element_loads
        self.spring_supports = model.load_case.spring_supports
        self.node_dofs_mapping = self.map_node_dofs()

    def map_node_dofs(self):
        dof_mapping = {}
        for i, node in enumerate(self.nodes):
            dof_mapping[node.id] = i
        return dof_mapping

    def assemble_global_elastic_stiffness_matrix(self):
        K = np.zeros((len(self.nodes), len(self.nodes)))

        for element in self.elements:
            node_1, node_2 = element.nodes
            dofs = [
                self.node_dofs_mapping[node_1.id],
                self.node_dofs_mapping[node_2.id],
            ]
            k_local = element.calculate_axial_stiffness_matrix()
            K[np.ix_(dofs, dofs)] += k_local

        axial_springs = [
            spring
            for spring in self.spring_supports
            if spring.direction == SpringDirection.AXIAL
        ]
        for spring in axial_springs:
            dof = self.node_dofs_mapping[spring.node.id]
            K[dof, dof] += spring.stiffness

        return K

    def assemble_global_force_vector(self):
        f = np.zeros(len(self.nodes))

        for nodal_load in self.nodal_loads:
            dof = self.node_dofs_mapping[nodal_load.node.id]
            f[dof] += nodal_load.force

        for element_load in self.element_loads:
            node_1, node_2 = element_load.element.nodes
            element_length = element_load.element.length
            dofs = [
                self.node_dofs_mapping[node_1.id],
                self.node_dofs_mapping[node_2.id],
            ]
            f[np.ix_(dofs)] += element_load.force * element_length / 2

        return f

    def get_partitioned_system(self) -> PartitionedSystem:
        K = self.assemble_global_elastic_stiffness_matrix()
        f = self.assemble_global_force_vector()

        all_dofs = np.array([self.node_dofs_mapping[node.id] for node in self.nodes])
        constrained_dofs = np.array(
            [
                self.node_dofs_mapping[bc.node.id]
                for bc in self.boundary_conditions
                if bc.bc_axial == BoundaryConditionType.FIXED
            ]
        )

        if len(constrained_dofs) == 0:
            raise ValueError("Nenhum GDL restrito encontrado.")

        free_dofs = np.setdiff1d(all_dofs, constrained_dofs)

        # f - free
        # c - constrained
        K_ff = K[np.ix_(free_dofs, free_dofs)]
        K_fc = K[np.ix_(free_dofs, constrained_dofs)]
        K_cf = K[np.ix_(constrained_dofs, free_dofs)]
        K_cc = K[np.ix_(constrained_dofs, constrained_dofs)]
        f_f = f[free_dofs]
        f_c = f[constrained_dofs]

        return PartitionedSystem(
            K_ff=K_ff,
            K_fc=K_fc,
            K_cf=K_cf,
            K_cc=K_cc,
            f_f=f_f,
            f_c=f_c,
            free_dofs=free_dofs,
            constrained_dofs=constrained_dofs,
        )
