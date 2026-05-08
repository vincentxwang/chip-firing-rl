#!/usr/bin/env python3
"""
Visualize gonality/chip-firing multigraphs from adjacency matrices or edge lists.

Examples:
  python visualize_gonality_graph.py --example --show
  python visualize_gonality_graph.py known_gsgs.txt --graph-index 1 --output graph.png
  python visualize_gonality_graph.py --matrix "0 2 0 1; 2 0 1 0; 0 1 0 1; 1 0 1 0"
"""

from __future__ import annotations

import argparse
import ast
import math
import os
import re
from collections import Counter
from pathlib import Path
from typing import Iterable

os.environ.setdefault("MPLCONFIGDIR", "/tmp/gonality-graph-matplotlib-cache")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/gonality-graph-cache")

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
import networkx as nx


EXAMPLE_MATRIX = [
    [0, 2, 0, 1],
    [2, 0, 1, 0],
    [0, 1, 0, 1],
    [1, 0, 1, 0],
]


def parse_matrix_text(text: str) -> list[list[int]]:
    """Parse whitespace/comma/semicolon separated adjacency matrix text."""
    text = text.strip()
    if not text:
        raise ValueError("matrix text is empty")

    try:
        parsed = ast.literal_eval(text)
        if isinstance(parsed, list) and all(isinstance(row, list) for row in parsed):
            return validate_matrix([[int(value) for value in row] for row in parsed])
    except (SyntaxError, ValueError):
        pass

    rows: list[list[int]] = []
    for raw_line in re.split(r"[;\n]+", text):
        line = raw_line.strip().strip("[]")
        if not line:
            continue
        values = [value for value in re.split(r"[\s,]+", line) if value]
        rows.append([int(value) for value in values])
    return validate_matrix(rows)


def validate_matrix(matrix: list[list[int]]) -> list[list[int]]:
    if not matrix:
        raise ValueError("matrix has no rows")
    n = len(matrix)
    if any(len(row) != n for row in matrix):
        raise ValueError("matrix must be square")
    for i in range(n):
        if matrix[i][i] != 0:
            raise ValueError("matrix diagonal must be zero")
        for j in range(i + 1, n):
            if matrix[i][j] != matrix[j][i]:
                raise ValueError("matrix must be symmetric")
            if matrix[i][j] < 0:
                raise ValueError("edge multiplicities must be nonnegative")
    return matrix


def matrix_to_edges(matrix: list[list[int]]) -> list[tuple[int, int]]:
    edges: list[tuple[int, int]] = []
    for i, row in enumerate(matrix, start=1):
        for j, multiplicity in enumerate(row[i:], start=i + 1):
            edges.extend((i, j) for _ in range(multiplicity))
    return edges


def parse_graph_edges_text(text: str) -> list[tuple[int, int]] | None:
    match = re.search(r"Edges=\[(.*?)\]\)", text, flags=re.S)
    if not match:
        return None
    pairs = re.findall(r"\((\d+),\s*(\d+)\)", match.group(1))
    if not pairs:
        return None
    return [(int(u), int(v)) for u, v in pairs]


def extract_matrices_from_file(path: Path) -> list[list[list[int]]]:
    text = path.read_text()
    matrices: list[list[list[int]]] = []
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        if "Adjacency Matrix:" not in lines[i]:
            i += 1
            continue

        block: list[str] = []
        i += 1
        while i < len(lines):
            line = lines[i].strip()
            if not line or line.startswith("="):
                break
            if re.fullmatch(r"[-\d\s,;]+", line):
                block.append(line)
                i += 1
                continue
            break

        if block:
            matrices.append(parse_matrix_text("\n".join(block)))
    return matrices


def extract_graph_edges_from_file(path: Path) -> list[list[tuple[int, int]]]:
    text = path.read_text()
    return [parse_graph_edges_text(match.group(0)) or [] for match in re.finditer(r"Graph\(V=.*?Edges=\[.*?\]\)", text)]


def edge_list_to_multigraph(edges: Iterable[tuple[int, int]]) -> nx.MultiGraph:
    graph = nx.MultiGraph()
    for u, v in edges:
        graph.add_edge(u, v)
    return graph


def node_label(node: int, center_node: int | None) -> str:
    if node == center_node:
        return r"$v_0$"
    return rf"$v_{{{node}}}$"


def choose_center_node(graph: nx.MultiGraph) -> int | None:
    simple = nx.Graph(graph)
    n = simple.number_of_nodes()
    if n < 5:
        return None

    degrees = dict(simple.degree())
    node, degree = max(degrees.items(), key=lambda item: (item[1], -item[0]))
    if degree >= max(4, n - 2):
        return node
    return None


def circular_positions(nodes: list[int], center_node: int | None) -> dict[int, tuple[float, float]]:
    outer_nodes = [node for node in nodes if node != center_node]
    radius = 1.7 if len(outer_nodes) <= 8 else 2.15
    start_angle = math.pi / 2
    positions: dict[int, tuple[float, float]] = {}

    for index, node in enumerate(outer_nodes):
        angle = start_angle - (2 * math.pi * index / max(1, len(outer_nodes)))
        positions[node] = (radius * math.cos(angle), radius * math.sin(angle))

    if center_node is not None:
        positions[center_node] = (0.0, 0.0)
    return positions


def draw_edge(
    ax: plt.Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    color: str,
    width: float,
    radius: float,
    zorder: int,
) -> None:
    patch = FancyArrowPatch(
        start,
        end,
        arrowstyle="-",
        connectionstyle=f"arc3,rad={radius}",
        color=color,
        linewidth=width,
        shrinkA=13,
        shrinkB=13,
        zorder=zorder,
    )
    ax.add_patch(patch)


def perimeter_edges(graph: nx.MultiGraph, positions: dict[int, tuple[float, float]], center_node: int | None) -> set[tuple[int, int]]:
    outer_nodes = [node for node in graph.nodes if node != center_node]
    if len(outer_nodes) < 3:
        return set()

    ordered = sorted(outer_nodes, key=lambda node: math.atan2(positions[node][1], positions[node][0]), reverse=True)
    simple = nx.Graph(graph)
    edges: set[tuple[int, int]] = set()
    for index, node in enumerate(ordered):
        next_node = ordered[(index + 1) % len(ordered)]
        if simple.has_edge(node, next_node):
            edges.add(tuple(sorted((node, next_node))))
    return edges


def draw_multigraph(
    graph: nx.MultiGraph,
    output: Path | None,
    show: bool,
    title: str,
    seed: int,
) -> None:
    if graph.number_of_nodes() == 0:
        raise ValueError("graph has no nodes")

    red = "#d90012"
    label_red = "#b64f61"
    gray = "#555555"
    black = "#000000"

    nodes = sorted(graph.nodes)
    center_node = choose_center_node(graph)
    pos = circular_positions(nodes, center_node)
    highlighted_edges = perimeter_edges(graph, pos, center_node)

    scale = 5.2 if graph.number_of_nodes() <= 8 else 7.2
    fig, ax = plt.subplots(figsize=(scale, scale))
    ax.set_aspect("equal")
    ax.axis("off")

    edge_counts = Counter(tuple(sorted((u, v))) for u, v in graph.edges())

    for (u, v), count in edge_counts.items():
        is_highlighted = (u, v) in highlighted_edges
        color = red if is_highlighted and count == 1 else gray
        width = 4.2 if is_highlighted and count == 1 else 1.8
        zorder = 2 if is_highlighted and count == 1 else 1

        if count == 1:
            radii = [0.0]
        elif count == 2:
            radii = [-0.24, 0.24]
        else:
            midpoint = (count - 1) / 2
            radii = [(index - midpoint) * 0.18 for index in range(count)]

        for radius in radii:
            draw_edge(ax, pos[u], pos[v], color, width, radius, zorder)

    for node in nodes:
        x, y = pos[node]
        is_center = node == center_node
        ax.scatter(
            [x],
            [y],
            s=185 if is_center else 260,
            c=black if is_center else red,
            edgecolors=black if is_center else "white",
            linewidths=1.1,
            zorder=4,
        )

    label_radius = 0.38 if graph.number_of_nodes() <= 8 else 0.28
    for node in nodes:
        x, y = pos[node]
        if node == center_node:
            dx, dy = 0.0, -0.34
        else:
            length = math.hypot(x, y) or 1.0
            dx = label_radius * x / length
            dy = label_radius * y / length
        ax.text(
            x + dx,
            y + dy,
            node_label(node, center_node),
            ha="center",
            va="center",
            color=black if node == center_node else label_red,
            fontsize=19 if graph.number_of_nodes() <= 8 else 14,
            fontweight="semibold",
            family="DejaVu Serif",
            zorder=5,
        )

    if title:
        ax.set_title(title, fontsize=14, pad=12)

    margin = 0.85
    xs = [point[0] for point in pos.values()]
    ys = [point[1] for point in pos.values()]
    ax.set_xlim(min(xs) - margin, max(xs) + margin)
    ax.set_ylim(min(ys) - margin, max(ys) + margin)
    plt.tight_layout(pad=0.15)

    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)


def select_graph(args: argparse.Namespace) -> tuple[nx.MultiGraph, str]:
    if args.example:
        return edge_list_to_multigraph(matrix_to_edges(EXAMPLE_MATRIX)), "Example Gonality Multigraph"

    if args.matrix:
        matrix = parse_matrix_text(args.matrix)
        return edge_list_to_multigraph(matrix_to_edges(matrix)), "Adjacency Matrix Multigraph"

    if args.edges:
        edges = [(int(u), int(v)) for u, v in re.findall(r"\((\d+),\s*(\d+)\)", args.edges)]
        if not edges:
            raise ValueError("--edges must contain pairs like '(1, 2), (1, 2), (2, 3)'")
        return edge_list_to_multigraph(edges), "Edge List Multigraph"

    if args.input_file:
        path = Path(args.input_file)
        matrices = extract_matrices_from_file(path)
        if matrices:
            index = args.graph_index - 1
            if index < 0 or index >= len(matrices):
                raise ValueError(f"--graph-index must be between 1 and {len(matrices)}")
            return edge_list_to_multigraph(matrix_to_edges(matrices[index])), f"{path.name} matrix #{args.graph_index}"

        edge_graphs = [edges for edges in extract_graph_edges_from_file(path) if edges]
        if edge_graphs:
            index = args.graph_index - 1
            if index < 0 or index >= len(edge_graphs):
                raise ValueError(f"--graph-index must be between 1 and {len(edge_graphs)}")
            return edge_list_to_multigraph(edge_graphs[index]), f"{path.name} edge graph #{args.graph_index}"

        raise ValueError(f"could not find an adjacency matrix or Graph(...Edges=[...]) block in {path}")

    raise ValueError("provide --example, --matrix, --edges, or an input file")


def main() -> None:
    parser = argparse.ArgumentParser(description="Render ChipFiring/Gonality multigraphs.")
    parser.add_argument("input_file", nargs="?", help="Text file containing an 'Adjacency Matrix:' block or Graph(...Edges=[...]).")
    parser.add_argument("--matrix", help="Adjacency matrix as Python list text, newline text, or semicolon-separated rows.")
    parser.add_argument("--edges", help="Edge pairs such as '(1, 2), (1, 2), (2, 3)'.")
    parser.add_argument("--graph-index", type=int, default=1, help="1-based graph number to render from a file.")
    parser.add_argument("--output", type=Path, default=Path("gonality_graph.png"), help="Output image path.")
    parser.add_argument("--show", action="store_true", help="Open an interactive matplotlib window.")
    parser.add_argument("--example", action="store_true", help="Render the example matrix from compute_gonality.jl.")
    parser.add_argument("--seed", type=int, default=7, help="Layout seed for reproducible drawings.")
    parser.add_argument("--title", help="Override the plot title.")
    args = parser.parse_args()

    graph, default_title = select_graph(args)
    draw_multigraph(graph, args.output, args.show, args.title or "", args.seed)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
