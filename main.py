import logging
from pathlib import Path

import typer

from src.fem_buckling.assembler import AxialAssembler
from src.fem_buckling.model_builder import ModelBuilder
from src.fem_buckling.parser import InputReader
from src.fem_buckling.solver import AxialSolver

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

app = typer.Typer()


@app.command(help="Run the FEM buckling analysis for a given input file.")
def run_analysis(path: str = typer.Argument(..., help="Path to the input file")):
    input_reader = InputReader()
    logging.info(f"Reading input file: {path}")
    input_reader.read(Path(path))
    input_data = input_reader.parse()

    builder = ModelBuilder()
    model = builder.build(input_data)

    axial_assembler = AxialAssembler(model)
    axial_system = axial_assembler.get_partitioned_system()
    axial_solver = AxialSolver(model, axial_system)
    axial_result = axial_solver.solve()
    print(f"Axial Displacements: {axial_result.axial_displacements}")
    print(f"Reaction Forces: {axial_result.reaction_forces}")
    print(f"Axial Forces: {axial_result.axial_forces}")


if __name__ == "__main__":
    app()
