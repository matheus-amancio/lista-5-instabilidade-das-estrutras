from dataclasses import dataclass

import numpy as np

from src.fem_buckling.boundary_condition import BoundaryConditionType, SpringDirection
from src.fem_buckling.model_builder import Model
from src.fem_buckling.result import AxialResults


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


@dataclass(frozen=True)
class PartitionedBucklingSystem:
    K_ff: np.ndarray
    Kg_ff: np.ndarray
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


class BucklingAssembler:
    def __init__(self, model: Model, axial_results: AxialResults):
        self.model = model
        self.axial_results = axial_results
        self.nodes = model.mesh.nodes
        self.elements = model.mesh.elements
        self.boundary_conditions = model.load_case.boundary_conditions
        self.spring_supports = model.load_case.spring_supports
        self.axial_forces = axial_results.axial_forces
        self.node_dofs_mapping = self.map_node_dofs()

    def map_node_dofs(self):
        dof_mapping = {}
        for i, node in enumerate(self.nodes):
            dof_mapping[node.id] = (2 * i, 2 * i + 1)
        return dof_mapping

    def assemble_global_elastic_stiffness_matrix(self):
        K = np.zeros((2 * len(self.nodes), 2 * len(self.nodes)))

        for element in self.elements:
            node_1, node_2 = element.nodes
            dofs = [
                self.node_dofs_mapping[node_1.id][0],
                self.node_dofs_mapping[node_1.id][1],
                self.node_dofs_mapping[node_2.id][0],
                self.node_dofs_mapping[node_2.id][1],
            ]
            k_local = element.calculate_transverse_stiffness_matrix()
            K[np.ix_(dofs, dofs)] += k_local

        transverse_springs = [
            spring
            for spring in self.spring_supports
            if spring.direction == SpringDirection.TRANSVERSE
        ]
        rotational_springs = [
            spring
            for spring in self.spring_supports
            if spring.direction == SpringDirection.ROTATIONAL
        ]
        for spring in transverse_springs:
            dof = self.node_dofs_mapping[spring.node.id][0]
            K[dof, dof] += spring.stiffness
        for spring in rotational_springs:
            dof = self.node_dofs_mapping[spring.node.id][1]
            K[dof, dof] += spring.stiffness

        return K

    def assemble_global_geometric_stiffness_matrix(self):
        Kg = np.zeros((2 * len(self.nodes), 2 * len(self.nodes)))

        for element in self.elements:
            node_1, node_2 = element.nodes
            dofs = [
                self.node_dofs_mapping[node_1.id][0],
                self.node_dofs_mapping[node_1.id][1],
                self.node_dofs_mapping[node_2.id][0],
                self.node_dofs_mapping[node_2.id][1],
            ]
            reference_axial_force = (-1) * next(
                (f.force for f in self.axial_forces if f.element.id == element.id)
            )  # convention: positive axial force is compression, negative is tension
            kg_local = element.calculate_geometric_stiffness_matrix(
                axial_reference_force_positive_in_compression=reference_axial_force
            )
            Kg[np.ix_(dofs, dofs)] += kg_local

        return Kg

    def get_partitioned_system(self) -> PartitionedBucklingSystem:
        K = self.assemble_global_elastic_stiffness_matrix()
        Kg = self.assemble_global_geometric_stiffness_matrix()

        all_dofs = np.array(
            [dof for node in self.nodes for dof in self.node_dofs_mapping[node.id]]
        )
        constrained_dofs = []
        for bc in self.boundary_conditions:
            node_dofs = self.node_dofs_mapping[bc.node.id]
            if bc.bc_transverse == BoundaryConditionType.FIXED:
                constrained_dofs.append(node_dofs[0])
            if bc.bc_rotational == BoundaryConditionType.FIXED:
                constrained_dofs.append(node_dofs[1])

        if len(constrained_dofs) == 0:
            raise ValueError("Nenhum GDL restrito encontrado.")

        free_dofs = np.setdiff1d(all_dofs, constrained_dofs)

        K_ff = K[np.ix_(free_dofs, free_dofs)]
        Kg_ff = Kg[np.ix_(free_dofs, free_dofs)]

        return PartitionedBucklingSystem(
            K_ff=K_ff,
            Kg_ff=Kg_ff,
            free_dofs=free_dofs,
            constrained_dofs=constrained_dofs,
        )
