from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pytz
from jinja2 import Template
from prettytable import PrettyTable

from src.fem_buckling.boundary_condition import BoundaryConditionType, SpringDirection
from src.fem_buckling.mesh import Element, Node
from src.fem_buckling.model_builder import Model


def get_date_now():
    tz = pytz.timezone("America/Sao_Paulo")
    return datetime.now(tz).strftime("%d/%m/%Y")


def format_float(value, signed=False, precision=3):
    if isinstance(value, float):
        if signed:
            return f"{value:+.{precision}f}"
        return f"{value:.{precision}f}"
    return value


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


class ResultsWriter:
    def __init__(
        self,
        input_file_path: Path,
        input_data: dict,
        model: Model,
        axial_results: AxialResults,
    ):
        self.input_file_path = input_file_path
        self.output_file_name = input_file_path.with_suffix(".out")
        self.model = model
        self.axial_results = axial_results
        self.input_data = input_data
        self.output_file_content = dict()
        self.template_path = Path(__file__).parent / "assets" / "template.out"

    def get_formatted_results(self):
        self.write_model_info()
        self.write_node_table()
        self.write_segment_table()
        self.write_axial_displacements_table()
        self.write_axial_forces_table()

    def write_info(self, key, value):
        self.output_file_content[key] = value

    def write_model_info(self):
        self.write_info("input_file_name", self.input_file_path.name)
        self.write_info("execution_date", get_date_now())

    def write_node_table(self):
        table = PrettyTable()
        table.field_names = [
            "ID",
            "X",
            "Glib. Axial",
            "Glib. Transv.",
            "Glib. Rotat.",
            "KL",
            "KT",
            "KR",
            "P axial",
        ]
        for node in self.model.mesh.nodes:
            bc_axial = "LIVRE"
            bc_transverse = "LIVRE"
            bc_rotational = "LIVRE"
            for bc in self.model.load_case.boundary_conditions:
                if bc.node.id == node.id:
                    if bc.bc_axial == BoundaryConditionType.FIXED:
                        bc_axial = "RESTRINGIDO"
                    if bc.bc_transverse == BoundaryConditionType.FIXED:
                        bc_transverse = "RESTRINGIDO"
                    if bc.bc_rotational == BoundaryConditionType.FIXED:
                        bc_rotational = "RESTRINGIDO"
            KL = "-"
            KT = "-"
            KR = "-"
            for spring in self.model.load_case.spring_supports:
                if spring.node.id == node.id:
                    if spring.direction == SpringDirection.AXIAL:
                        KL = spring.stiffness
                    if spring.direction == SpringDirection.TRANSVERSE:
                        KT = spring.stiffness
                    if spring.direction == SpringDirection.ROTATIONAL:
                        KR = spring.stiffness
            P = "-"
            for load in self.model.load_case.nodal_loads:
                if load.node.id == node.id:
                    P = load.force
            table.add_row(
                [
                    node.id,
                    format_float(node.x),
                    bc_axial,
                    bc_transverse,
                    bc_rotational,
                    format_float(KL),
                    format_float(KT),
                    format_float(KR),
                    format_float(P, signed=True),
                ]
            )
        self.write_info("table_nodes", table.get_string())

    def write_segment_table(self):
        table = PrettyTable()
        table.field_names = [
            "ID",
            "L",
            "Num. Elements",
            "EA",
            "EI",
            "kl",
            "kt",
            "kr",
            "p",
        ]
        for segment in self.model.mesh.segments:
            table.add_row(
                [
                    segment.id,
                    format_float(segment.props.L),
                    segment.props.ne,
                    format_float(segment.props.EA),
                    format_float(segment.props.EI),
                    format_float(segment.props.kl),
                    format_float(segment.props.kt),
                    format_float(segment.props.kr),
                    format_float(segment.props.p, signed=True),
                ]
            )
        self.write_info("table_segments", table.get_string())

    def write_axial_displacements_table(self):
        table = PrettyTable()
        table.field_names = [
            "Node ID",
            "X",
            "Axial Displacement",
            "Residual/Reaction Force",
        ]
        for i, node in enumerate(self.model.mesh.nodes):
            axial_displacement = self.axial_results.axial_displacements[i]
            reaction_force = next(
                (
                    rf.force
                    for rf in self.axial_results.reaction_forces
                    if rf.node.id == node.id
                ),
                "-",
            )
            table.add_row(
                [
                    node.id,
                    format_float(node.x),
                    format_float(axial_displacement, signed=True),
                    format_float(reaction_force, signed=True),
                ]
            )
        self.write_info("table_axial_displacements", table.get_string())

    def write_axial_forces_table(self):
        table = PrettyTable()
        table.field_names = [
            "Element ID",
            "Segment ID",
            "Node 1 ID",
            "Node 2 ID",
            "X_ref",
            "Axial Force",
        ]
        for axial_force in self.axial_results.axial_forces:
            element = axial_force.element
            table.add_row(
                [
                    element.id,
                    self.model.mesh.segments[element.segment_idx].id,
                    element.nodes[0].id,
                    element.nodes[1].id,
                    format_float(element.nodes[0].x),
                    format_float(axial_force.force, signed=True),
                ]
            )
        self.write_info("table_axial_forces", table.get_string())

    def save(self):
        with open(self.template_path, "r") as template_file:
            template_content = Template(template_file.read())
        output_content = template_content.render(self.output_file_content)
        with open(self.output_file_name, "w") as output_file:
            output_file.write(output_content)
