from pathlib import Path

from src.fem_buckling.boundary_condition import BoundaryConditionType


class InputReader:
    def __init__(self):
        self.lines: list[str] = []

    def read(self, path: Path):
        if not path.is_file():
            raise FileNotFoundError(f"File not found: {path}")
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
            self.lines = content.splitlines()

    def parse(self):
        num_segments = int(self.lines[0])
        nodes_info = []
        segments_info = []
        for i in range(2, 2 * num_segments + 2, 2):
            nodes_info.append(self.get_node_info(self.lines[i - 1]))
            element_splitted_line = self.lines[i].split()
            segments_info.append(
                {
                    "L": float(element_splitted_line[0]),
                    "num_elements": int(element_splitted_line[1]),
                    "EA": float(element_splitted_line[2]),
                    "EI": float(element_splitted_line[3]),
                    "kl": float(element_splitted_line[4]),
                    "kt": float(element_splitted_line[5]),
                    "kr": float(element_splitted_line[6]),
                    "p": float(element_splitted_line[7]),
                }
            )
        final_node_info = self.get_node_info(self.lines[-2])
        nodes_info.append(final_node_info)
        num_buckling_modes = int(self.lines[-1])
        return {
            "num_segments": num_segments,
            "nodes_info": nodes_info,
            "segments_info": segments_info,
            "num_buckling_modes": num_buckling_modes,
        }

    def get_node_info(self, line: list[str]) -> dict:
        splitted_line = line.split()
        node_info = dict()
        node_info["bc_axial"] = int(splitted_line[0])
        aux = 1
        if node_info["bc_axial"] == BoundaryConditionType.FREE.value:
            node_info["KL"] = float(splitted_line[1])
            node_info["P"] = float(splitted_line[2])
            aux = 3

        node_info["bc_transverse"] = int(splitted_line[aux])
        aux += 1
        if node_info["bc_transverse"] == BoundaryConditionType.FREE.value:
            node_info["KT"] = float(splitted_line[aux])
            aux += 1

        node_info["bc_rotational"] = int(splitted_line[aux])
        aux += 1
        if node_info["bc_rotational"] == BoundaryConditionType.FREE.value:
            node_info["KR"] = float(splitted_line[aux])

        return node_info
