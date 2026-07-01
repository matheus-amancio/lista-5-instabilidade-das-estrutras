from dataclasses import dataclass

import numpy as np


class Node:
    def __init__(self, id: int, x: float):
        self.id = id
        self.x = x

    def __repr__(self):
        return f"Node(id={self.id}, x={self.x})"


@dataclass(frozen=True)
class ElementProperties:
    EA: float
    EI: float
    kl: float
    kt: float
    kr: float
    p: float


class Element:
    def __init__(self, id: int, nodes: tuple[Node], props: ElementProperties):
        self.id = id
        self.nodes = nodes
        self.props = props
        self.length = abs(nodes[1].x - nodes[0].x)

    def __repr__(self):
        return f"Element(id={self.id})"

    def calculate_axial_stiffness_matrix(self):
        EA, L, kl = self.props.EA, self.length, self.props.kl
        return EA / L * np.array([[1, -1], [-1, 1]]) + kl * L / 6 * np.array(
            [[2, 1], [1, 2]]
        )


class Mesh:
    def __init__(self):
        self.nodes: list[Node] = []
        self.elements: list[Element] = []

        self.reset_counters()

    def add_node(self, x: float) -> Node:
        node = Node(self.node_id_counter, x)
        self.nodes.append(node)
        self.node_id_counter += 1
        return node

    def add_element(self, nodes: tuple[Node], props: ElementProperties) -> Element:
        element = Element(self.element_id_counter, nodes, props)
        self.elements.append(element)
        self.element_id_counter += 1
        return element

    def reset_counters(self):
        self.node_id_counter = 1
        self.element_id_counter = 1
