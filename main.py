from pathlib import Path

from src.fem_buckling.assembler import StaticAssembler
from src.fem_buckling.model_builder import ModelBuilder
from src.fem_buckling.parser import InputReader


def main():
    input_reader = InputReader()
    input_reader.read(Path("src/examples/ex01.ie"))
    input_data = input_reader.parse()

    builder = ModelBuilder()
    model = builder.build(input_data)

    static_assembler = StaticAssembler(model)
    static_system = static_assembler.get_partitioned_system()
    print(f"Partitioned System: {static_system}")


if __name__ == "__main__":
    main()
