from dataclasses import dataclass


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

    def __repr__(self):
        return f"Element(id={self.id})"


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
