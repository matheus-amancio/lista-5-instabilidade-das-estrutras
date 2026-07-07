from dataclasses import dataclass

import numpy as np


class Node:
    def __init__(self, id: int, x: float):
        self.id = id
        self.x = x

    def __repr__(self):
        return f"Node(id={self.id}, x={round(self.x, 3)})"


@dataclass(frozen=True)
class ElementProperties:
    EA: float
    EI: float
    kl: float
    kt: float
    kr: float
    p: float


@dataclass(frozen=True)
class SegmentProperties:
    L: float
    ne: int
    EA: float
    EI: float
    kl: float
    kt: float
    kr: float
    p: float


class Element:
    def __init__(
        self, id: int, nodes: tuple[Node], props: ElementProperties, segment_idx: int
    ):
        self.id = id
        self.nodes = nodes
        self.props = props
        self.segment_idx = segment_idx
        self.length = abs(nodes[1].x - nodes[0].x)

    def __repr__(self):
        return f"Element(id={self.id}, node1_id={self.nodes[0].id}, node2_id={self.nodes[1].id}, segment_idx={self.segment_idx})"

    def calculate_axial_stiffness_matrix(self) -> np.ndarray:
        EA, L, kl = self.props.EA, self.length, self.props.kl
        return EA / L * np.array([[1, -1], [-1, 1]]) + kl * L / 6 * np.array(
            [[2, 1], [1, 2]]
        )

    def calculate_axial_force(self, u_element: np.ndarray) -> float:
        EA, L = self.props.EA, self.length
        u_j, u_i = u_element[1], u_element[0]
        return EA / L * (u_j - u_i)


class Segment:
    def __init__(self, id: int, extreme_nodes: tuple[Node], props: SegmentProperties):
        self.id = id
        self.extreme_nodes = extreme_nodes
        self.props = props

    def __repr__(self):
        return f"Segment(id={self.id}, node1_id={self.extreme_nodes[0].id}, node2_id={self.extreme_nodes[1].id})"


class Mesh:
    def __init__(self):
        self.nodes: list[Node] = []
        self.elements: list[Element] = []
        self.segments: list[Segment] = []

        self.reset_counters()

    def add_node(self, x: float) -> Node:
        node = Node(self.node_id_counter, x)
        self.nodes.append(node)
        self.node_id_counter += 1
        return node

    def add_element(
        self, nodes: tuple[Node], props: ElementProperties, segment_idx: int
    ) -> Element:
        element = Element(self.element_id_counter, nodes, props, segment_idx)
        self.elements.append(element)
        self.element_id_counter += 1
        return element

    def add_segment(
        self, extreme_nodes: tuple[Node], props: SegmentProperties
    ) -> Segment:
        segment = Segment(self.segment_id_counter, extreme_nodes, props)
        self.segments.append(segment)
        self.segment_id_counter += 1
        return segment

    def compute_num_nodes(self):
        return len(self.nodes)

    def compute_num_elements(self):
        return len(self.elements)

    def compute_num_segments(self):
        return len(self.segments)

    def reset_counters(self):
        self.node_id_counter = 1
        self.element_id_counter = 1
        self.segment_id_counter = 1
