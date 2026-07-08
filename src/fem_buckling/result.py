from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import pytz
from jinja2 import Template
from plotly.subplots import make_subplots
from prettytable import PrettyTable

from src.fem_buckling.boundary_condition import BoundaryConditionType, SpringDirection
from src.fem_buckling.mesh import Element, Node
from src.fem_buckling.model_builder import Model


def get_date_now():
    tz = pytz.timezone("America/Sao_Paulo")
    return datetime.now(tz).strftime("%d/%m/%Y")


def format_float(value, signed=False, precision=6):
    if isinstance(value, float):
        if signed:
            return f"{value:+.{precision}E}"
        return f"{value:.{precision}E}"
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
        model: Model,
        axial_displacements: np.ndarray,
        reaction_forces: list[ReactionForce],
        axial_forces: list[AxialForce],
    ):
        self.model = model
        self.nodes = model.mesh.nodes
        self.elements = model.mesh.elements
        self.axial_displacements = axial_displacements
        self.reaction_forces = reaction_forces
        self.axial_forces = axial_forces

    def plot_results(
        self,
        input_file_path: Path,
    ):
        plot_file_path = input_file_path.with_suffix(".html").with_name(
            f"{input_file_path.stem}_axial_results.html"
        )
        fig = make_subplots(
            2,
            1,
            shared_xaxes=True,
            vertical_spacing=0.1,
            subplot_titles=(
                "Deslocamento axial",
                "Esforço normal (positivo em tração)",
            ),
            x_title="Coordenada X",
        )

        fig.add_trace(
            go.Scatter(
                x=[node.x for node in self.nodes],
                y=self.axial_displacements,
                mode="markers+lines",
                showlegend=False,
                name="Deslocamento axial",
            ),
            row=1,
            col=1,
        )

        for element, axial_force in zip(self.elements, self.axial_forces):
            x_coords = [element.nodes[0].x, element.nodes[1].x]
            y_coords = [axial_force.force, axial_force.force]
            fig.add_trace(
                go.Scatter(
                    x=x_coords,
                    y=y_coords,
                    mode="markers+lines",
                    showlegend=False,
                    line=dict(color="red"),
                    name="Esforço normal",
                ),
                row=2,
                col=1,
            )

        fig.update_layout(
            template="plotly_white",
            yaxis_title="Axial Displacement",
            yaxis2_title="Axial Force",
            yaxis2=dict(
                range=[
                    min(min(af.force for af in self.axial_forces) * 1.1, 0.0),
                    max(0.0, max(af.force for af in self.axial_forces) * 1.1),
                ]
            ),
        )
        fig.update_yaxes(exponentformat="e", showexponent="all", row=1, col=1)

        fig.update_traces(
            hovertemplate="x=%{x}<br>y=%{y:.3e}<extra></extra>", row=1, col=1
        )

        fig.write_html(
            plot_file_path,
            auto_open=True,
            config={"scrollZoom": True},
            include_plotlyjs=True,
        )


@dataclass(frozen=True)
class BucklingMode:
    mode_number: int
    lambda_cr: float
    mode_shape: np.ndarray
    node_dofs_mapping: dict[int, tuple[int, int]]


class BucklingResults:
    def __init__(
        self, model: Model, num_modes: int, buckling_modes: list[BucklingMode]
    ):
        self.model = model
        self.num_modes = num_modes
        self.buckling_modes = buckling_modes

    def plot_mode_shapes(self, input_file_path: Path):
        elements = self.model.mesh.elements
        for mode in self.buckling_modes:
            plot_file_path = input_file_path.with_suffix(".html").with_name(
                f"{input_file_path.stem}_buckling_mode_{mode.mode_number}.html"
            )
            x = []
            y = []
            element_x_extreme_points = []
            element_y_extreme_points = []
            for element in elements:
                node_1, node_2 = element.nodes
                dofs_node_1 = mode.node_dofs_mapping[node_1.id]
                dofs_node_2 = mode.node_dofs_mapping[node_2.id]
                mode_displacements = np.array(
                    [
                        mode.mode_shape[dofs_node_1[0]],
                        mode.mode_shape[dofs_node_1[1]],
                        mode.mode_shape[dofs_node_2[0]],
                        mode.mode_shape[dofs_node_2[1]],
                    ]
                )
                x_local_range = np.linspace(0, element.length, 100)
                x_global_range = np.linspace(node_1.x, node_2.x, 100)
                transverse_displacements = []
                for x_k in x_local_range:
                    transverse_displacements.append(
                        element.transverse_displacement(x_k, mode_displacements)
                    )
                x.extend(x_global_range)
                y.extend(transverse_displacements)
                element_x_extreme_points.extend([node_1.x, node_2.x])
                element_y_extreme_points.extend(
                    [transverse_displacements[0], transverse_displacements[-1]]
                )
            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=y,
                    mode="lines",
                    name=f"Modo {mode.mode_number}",
                )
            )

            fig.update_layout(
                template="plotly_white",
                title=f"Modo de flambagem {mode.mode_number} (λ_cr = {mode.lambda_cr:.3e})",
                xaxis_title="Coordenada X",
                yaxis_title="Deslocamento Transversal",
            )

            fig.update_yaxes(exponentformat="e", showexponent="all")

            fig.update_traces(hovertemplate="x=%{x}<br>y=%{y:.3e}<extra></extra>")

            fig.add_trace(
                go.Scatter(
                    x=element_x_extreme_points,
                    y=element_y_extreme_points,
                    mode="markers",
                    name="Nós",
                    marker=dict(color="red", size=8),
                )
            )

            fig.write_html(
                plot_file_path,
                auto_open=True,
                config={"scrollZoom": True},
                include_plotlyjs=True,
            )


class ResultsWriter:
    def __init__(
        self,
        input_file_path: Path,
        input_data: dict,
        model: Model,
        axial_results: AxialResults,
        buckling_results: BucklingResults,
    ):
        self.input_file_path = input_file_path
        self.output_file_name = input_file_path.with_suffix(".out")
        self.model = model
        self.axial_results = axial_results
        self.buckling_results = buckling_results
        self.input_data = input_data
        self.output_file_content = dict()
        self.template_path = Path(__file__).parent / "assets" / "template.out"

    def get_formatted_results(self):
        self.write_model_info()
        self.write_node_table()
        self.write_segment_table()
        self.write_axial_displacements_table()
        self.write_axial_forces_table()
        self.write_modes()

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
            "Num. de Elementos",
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
            "Nó ID",
            "X",
            "Deslocamento Axial",
            "Força Residual/Reação",
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
            "Elemento ID",
            "Trecho ID",
            "Nó 1 ID",
            "Nó 2 ID",
            "X_ref",
            "Força Axial",
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

    def map_buckling_node_dofs(self):
        dof_mapping = {}
        for i, node in enumerate(self.model.mesh.nodes):
            dof_mapping[node.id] = (2 * i, 2 * i + 1)
        return dof_mapping

    def write_modes(self):
        nodes = self.model.mesh.nodes
        dof_mapping = self.map_buckling_node_dofs()
        buckling_modes_data = []
        for mode in self.buckling_results.buckling_modes:
            mode_info = {
                "number": mode.mode_number,
                "lambda_cr": format_float(mode.lambda_cr),
            }
            table = PrettyTable()
            table.field_names = ["Nó ID", "X", "v", "theta"]
            for node in nodes:
                dof_v, dof_theta = dof_mapping[node.id]
                v = mode.mode_shape[dof_v]
                theta = mode.mode_shape[dof_theta]
                table.add_row(
                    [
                        node.id,
                        format_float(node.x),
                        format_float(v, signed=True),
                        format_float(theta, signed=True),
                    ]
                )
            mode_info["table"] = table.get_string()
            buckling_modes_data.append(mode_info)
        self.write_info("buckling_modes", buckling_modes_data)

    def save(self):
        with open(self.template_path, "r", encoding="utf-8") as template_file:
            template_content = Template(template_file.read())
        output_content = template_content.render(self.output_file_content)
        with open(self.output_file_name, "w", encoding="utf-8") as output_file:
            output_file.write(output_content)
