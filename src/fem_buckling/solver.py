import numpy as np
from scipy.linalg import eigh

from src.fem_buckling.assembler import PartitionedBucklingSystem, PartitionedSystem
from src.fem_buckling.model_builder import Model
from src.fem_buckling.result import (
    AxialForce,
    AxialResults,
    BucklingMode,
    BucklingResults,
    ReactionForce,
)


class AxialSolver:
    def __init__(self, model: Model, partitioned_system: PartitionedSystem):
        self.model = model
        self.partitioned_system = partitioned_system
        self.node_dofs_mapping = self.map_node_dofs()

    def map_node_dofs(self):
        dof_mapping = {}
        for i, node in enumerate(self.model.mesh.nodes):
            dof_mapping[node.id] = i
        return dof_mapping

    def solve_displacements(self):
        K_ff = self.partitioned_system.K_ff
        f_f = self.partitioned_system.f_f

        return np.linalg.solve(K_ff, f_f)

    def solve_reaction_forces(self, u_f: np.ndarray):
        K_cf = self.partitioned_system.K_cf
        K_cc = self.partitioned_system.K_cc
        f_c = self.partitioned_system.f_c

        u_c = np.zeros(len(self.partitioned_system.constrained_dofs))
        R_c = K_cf @ u_f + K_cc @ u_c - f_c

        return R_c

    def solve(self):
        nodes = self.model.mesh.nodes
        elements = self.model.mesh.elements

        displacements = np.zeros(len(nodes))
        free_dofs = self.partitioned_system.free_dofs
        constrained_dofs = self.partitioned_system.constrained_dofs

        u_f = self.solve_displacements()
        displacements[np.ix_(free_dofs)] = u_f

        r_c = self.solve_reaction_forces(u_f)
        reaction_forces = []
        for dof in constrained_dofs:
            node = next(
                node for node in nodes if self.node_dofs_mapping[node.id] == dof
            )
            force = r_c[dof]
            reaction_force = ReactionForce(node=node, force=force)
            reaction_forces.append(reaction_force)

        axial_forces = []
        for element in elements:
            node_1, node_2 = element.nodes
            dof_1 = self.node_dofs_mapping[node_1.id]
            dof_2 = self.node_dofs_mapping[node_2.id]
            u_element = np.array([displacements[dof_1], displacements[dof_2]])
            axial_force = element.calculate_axial_force(u_element)
            axial_forces.append(AxialForce(element=element, force=axial_force))

        return AxialResults(
            model=self.model,
            axial_displacements=displacements,
            reaction_forces=reaction_forces,
            axial_forces=axial_forces,
        )


class BucklingSolver:
    def __init__(self, model: Model, partitioned_system: PartitionedBucklingSystem):
        self.model = model
        self.partitioned_system = partitioned_system
        self.node_dofs_mapping = self.map_node_dofs()

    def map_node_dofs(self):
        dof_mapping = {}
        for i, node in enumerate(self.model.mesh.nodes):
            dof_mapping[node.id] = (2 * i, 2 * i + 1)
        return dof_mapping

    def solve(self, num_modes: int) -> BucklingResults:
        K_ff = self.partitioned_system.K_ff
        Kg_ff = self.partitioned_system.Kg_ff

        eigenvalues, eigenvectors = eigh(
            K_ff, Kg_ff, subset_by_index=[0, num_modes - 1]
        )

        buckling_modes = []
        for k in range(num_modes):
            mode_shape = np.zeros(len(self.model.mesh.nodes) * 2)
            mode_shape[np.ix_(self.partitioned_system.free_dofs)] = eigenvectors[:, k]
            mode_shape = mode_shape / np.max(np.abs(mode_shape))
            buckling_modes.append(
                BucklingMode(
                    mode_number=k + 1,
                    lambda_cr=eigenvalues[k],
                    mode_shape=mode_shape,
                    node_dofs_mapping=self.node_dofs_mapping,
                )
            )

        return BucklingResults(
            model=self.model, num_modes=num_modes, buckling_modes=buckling_modes
        )
