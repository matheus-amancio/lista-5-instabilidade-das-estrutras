import logging
from pathlib import Path

import typer

from src.fem_buckling.assembler import AxialAssembler, BucklingAssembler
from src.fem_buckling.model_builder import ModelBuilder
from src.fem_buckling.parser import InputReader
from src.fem_buckling.result import ResultsWriter
from src.fem_buckling.solver import AxialSolver, BucklingSolver

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

app = typer.Typer()


@app.command(help="Run the FEM buckling analysis for a given input file.")
def run_analysis(path: str = typer.Argument(..., help="Path to the input file")):
    try:
        input_reader = InputReader()
        logging.info(f"Reading input file: {path}")
        input_reader.read(Path(path))
        input_data = input_reader.parse()

        logging.info("Building the model from input data.")
        builder = ModelBuilder()
        model = builder.build(input_data)

        logging.info("Assembling the axial system and solving for displacements.")
        axial_assembler = AxialAssembler(model)
        axial_system = axial_assembler.get_partitioned_system()
        axial_solver = AxialSolver(model, axial_system)
        axial_results = axial_solver.solve()

        logging.info("Assembling the buckling system and solving for buckling modes.")
        buckling_assembler = BucklingAssembler(model, axial_results)
        buckling_assembler.assemble_global_geometric_stiffness_matrix()
        buckling_system = buckling_assembler.get_partitioned_system()
        buckling_solver = BucklingSolver(model, buckling_system)
        buckling_results = buckling_solver.solve(
            num_modes=input_data["num_buckling_modes"]
        )

        logging.info("Writing results to output files.")
        results_writer = ResultsWriter(
            Path(path), input_data, model, axial_results, buckling_results
        )
        results_writer.get_formatted_results()
        results_writer.save()

        logging.info("Plotting results.")
        axial_results.plot_results(Path(path))
        buckling_results.plot_mode_shapes(Path(path))
        logging.info("Analysis completed successfully.")
    except Exception as e:
        logging.error(f"An error occurred: {e}")
        raise


if __name__ == "__main__":
    app()
