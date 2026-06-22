from pathlib import Path


def read_input_file(path: Path):
    if not path.is_file():
        raise FileNotFoundError(f"File not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def parse_input(path: Path):
    content = read_input_file(path)
    lines = content.splitlines()
    print(lines)


if __name__ == "__main__":
    parse_input(Path(__file__).parent.parent / "examples/ex01.ie")
