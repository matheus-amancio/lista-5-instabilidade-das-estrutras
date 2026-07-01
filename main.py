from pathlib import Path

from src.fem_buckling.parser import InputReader


def main():
    input_reader = InputReader()
    input_reader.read(Path("src/examples/ex01.ie"))
    input_reader.parse()


if __name__ == "__main__":
    main()
