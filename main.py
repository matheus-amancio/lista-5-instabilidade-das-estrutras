from pathlib import Path

from src.fem_buckling.assembler import AxialAssembler
from src.fem_buckling.model_builder import ModelBuilder
from src.fem_buckling.parser import InputReader


def main():
    input_reader = InputReader()
    input_reader.read(Path("src/examples/ex02.ie"))
    input_data = input_reader.parse()

    builder = ModelBuilder()
    model = builder.build(input_data)

    axial_assembler = AxialAssembler(model)
    static_system = axial_assembler.get_partitioned_system()
    print(f"Partitioned System: {static_system}")


if __name__ == "__main__":
    main()
