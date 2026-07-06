from pathlib import Path

from src.fem_buckling.assembler import AxialAssembler
from src.fem_buckling.model_builder import ModelBuilder
from src.fem_buckling.parser import InputReader
from src.fem_buckling.solver import AxialSolver


def main():
    input_reader = InputReader()
    input_reader.read(Path("src/examples/ex02.ie"))
    input_data = input_reader.parse()

    builder = ModelBuilder()
    model = builder.build(input_data)

    axial_assembler = AxialAssembler(model)
    axial_system = axial_assembler.get_partitioned_system()
    axial_solver = AxialSolver(model, axial_system)
    axial_result = axial_solver.solve()
    print("Axial Displacements:", axial_result.axial_displacements)
    print(f"Reaction Forces: {axial_result.reaction_forces}")
    print(f"Axial Forces: {axial_result.axial_forces}")


if __name__ == "__main__":
    main()
