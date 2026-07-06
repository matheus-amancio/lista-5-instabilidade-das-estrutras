import numpy as np

from src.fem_buckling.boundary_condition import (
    BoundaryCondition,
    ElementLoad,
    LoadCase,
    NodalLoad,
    SpringDirection,
    SpringSupport,
)
from src.fem_buckling.mesh import ElementProperties, Mesh, Node


def generate_node_coordinates(
    segment_info: dict, x0: float = 0.0, offset: int = 0
) -> np.ndarray:
    L = segment_info["L"]
    num_elements = segment_info["num_elements"]

    x1 = x0 + L
    return np.linspace(x0, x1, num_elements + 1)[offset:]


class ModelBuilder:
    def create_springs(self, info: dict, node: Node):
        spring_map = {
            "KL": SpringDirection.AXIAL,
            "KT": SpringDirection.TRANSVERSE,
            "KR": SpringDirection.ROTATIONAL,
        }

        springs = []
        for tag, direction in spring_map.items():
            stiffness = info.get(tag)
            if stiffness:
                springs.append(
                    SpringSupport(node=node, direction=direction, stiffness=stiffness)
                )
        return springs

    def build(self, parsed_data: dict):
        mesh = Mesh()
        mesh.reset_counters()

        segments_info = parsed_data["segments_info"]
        extreme_segment_nodes = []
        for i, seg_info in enumerate(segments_info):
            x0 = 0.0 if i == 0 else mesh.nodes[-1].x
            segment_node_coords = generate_node_coordinates(
                seg_info, x0=x0, offset=1 if i > 0 else 0
            )
            segment_nodes = []
            for x in segment_node_coords:
                segment_nodes.append(mesh.add_node(x))
            if i == 0:
                extreme_segment_nodes.append(segment_nodes[0])
            segment_props = ElementProperties(
                EA=seg_info["EA"],
                EI=seg_info["EI"],
                kl=seg_info["kl"],
                kt=seg_info["kt"],
                kr=seg_info["kr"],
                p=seg_info["p"],
            )
            if i > 0:
                mesh.add_element(
                    nodes=(extreme_segment_nodes[-1], segment_nodes[0]),
                    props=segment_props,
                )
            extreme_segment_nodes.append(segment_nodes[-1])

            for j in range(len(segment_nodes) - 1):
                mesh.add_element(
                    nodes=(segment_nodes[j], segment_nodes[j + 1]),
                    props=segment_props,
                )

        boundary_conditions = []
        nodal_loads = []
        spring_supports = []
        for i, node in enumerate(extreme_segment_nodes):
            node_info = parsed_data["nodes_info"][i]
            node_bc = BoundaryCondition(
                node=node,
                bc_axial=node_info["bc_axial"],
                bc_transverse=node_info["bc_transverse"],
                bc_rotational=node_info["bc_rotational"],
            )
            boundary_conditions.append(node_bc)
            if node_info.get("P"):
                node_load = NodalLoad(node=node, force=node_info["P"])
                nodal_loads.append(node_load)
            spring_supports.extend(self.create_springs(node_info, node))

        element_loads = []
        for element in mesh.elements:
            if element.props.p != 0.0:
                element_loads.append(
                    ElementLoad(element=element, force=element.props.p)
                )

        load_case = LoadCase(
            boundary_conditions=boundary_conditions,
            nodal_loads=nodal_loads,
            element_loads=element_loads,
            spring_supports=spring_supports,
        )

        return Model(mesh=mesh, load_case=load_case)


class Model:
    def __init__(self, mesh: Mesh, load_case: LoadCase):
        self.mesh = mesh
        self.load_case = load_case
        self.num_nodes = self.mesh.compute_num_nodes()
        self.num_elements = self.mesh.compute_num_elements()
