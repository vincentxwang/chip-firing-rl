#!/usr/bin/env python3
"""
Cross-entropy generator for gonality-saving multigraph candidates.

The search space is loopless multigraphs on a configurable number of labeled
vertices, with each unordered vertex pair assigned multiplicity 0 through a
configurable maximum.

Reward:
  if gon(G) < 5: gon(G)
  else: 50 + 1000 / |E| - 10 * tw(G)

The script logs and visualizes candidates where gon(G) differs from
gon(sigma_2(G)), using the current sigma_2 subdivision as the computable proxy
for the uniform-gonality comparison used elsewhere in this repo.
"""

from __future__ import annotations

import argparse
import itertools
import json
import random
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
VISUALIZER_DIR = REPO_ROOT / "Representation"
sys.path.insert(0, str(VISUALIZER_DIR))

from visualize_gonality_graph import draw_multigraph, edge_list_to_multigraph, matrix_to_edges


jl = None
_to_julia_matrix = None


@dataclass(frozen=True)
class SearchSpace:
    vertices: int
    max_multiplicity: int
    edge_pairs: tuple[tuple[int, int], ...]
    n_options: int


@dataclass(frozen=True)
class Evaluation:
    reward: float
    gon: int | None
    sigma2_gon: int | None
    treewidth: int | None
    edge_count: int
    connected: bool
    saving_checked: bool
    is_saving: bool


def make_search_space(vertices: int, max_multiplicity: int) -> SearchSpace:
    if vertices < 2:
        raise ValueError("--vertices must be at least 2")
    if max_multiplicity < 1:
        raise ValueError("--max-multiplicity must be at least 1")
    edge_pairs = tuple((i, j) for i in range(vertices) for j in range(i + 1, vertices))
    return SearchSpace(vertices, max_multiplicity, edge_pairs, max_multiplicity + 1)


def load_julia() -> None:
    global jl
    if jl is not None:
        return

    from juliacall import Main as julia_main

    jl = julia_main
    jl.seval("using ChipFiring")
    jl.seval("using Graphs")
    jl.seval("using TreeWidthSolver")


def matrix_key(matrix: list[list[int]], space: SearchSpace) -> tuple[int, ...]:
    return tuple(matrix[i][j] for i, j in space.edge_pairs)


def vertex_signature(matrix: list[list[int]], vertex: int, space: SearchSpace) -> tuple[int, ...]:
    row = matrix[vertex]
    multiplicity_counts = tuple(sum(1 for value in row if value == mult) for mult in range(1, space.max_multiplicity + 1))
    return multiplicity_counts + (sum(row),)


def canonical_permutation_count(groups: list[list[int]]) -> int:
    total = 1
    for group in groups:
        for value in range(2, len(group) + 1):
            total *= value
    return total


def canonical_isomorphism_key(matrix: list[list[int]], space: SearchSpace, max_permutations: int) -> tuple[int, ...]:
    signatures: dict[tuple[int, ...], list[int]] = {}
    for vertex in range(space.vertices):
        signatures.setdefault(vertex_signature(matrix, vertex, space), []).append(vertex)

    ordered_groups = [signatures[key] for key in sorted(signatures)]
    permutation_count = canonical_permutation_count(ordered_groups)
    if permutation_count > max_permutations:
        raise ValueError(
            f"canonical labeling would require {permutation_count} permutations; "
            f"increase --max-canonical-permutations or pass --allow-isomorphic-savings"
        )

    best: tuple[int, ...] | None = None
    for group_permutations in itertools.product(*(itertools.permutations(group) for group in ordered_groups)):
        permutation = tuple(vertex for group in group_permutations for vertex in group)
        key = tuple(
            matrix[permutation[i]][permutation[j]]
            for i in range(space.vertices)
            for j in range(i + 1, space.vertices)
        )
        if best is None or key < best:
            best = key
    if best is None:
        raise RuntimeError("failed to compute canonical isomorphism key")
    return best


def matrix_from_key(key: tuple[int, ...], space: SearchSpace) -> list[list[int]]:
    matrix = [[0 for _ in range(space.vertices)] for _ in range(space.vertices)]
    for (i, j), multiplicity in zip(space.edge_pairs, key):
        matrix[i][j] = multiplicity
        matrix[j][i] = multiplicity
    return matrix


def edge_count(matrix: list[list[int]], space: SearchSpace) -> int:
    return sum(matrix[i][j] for i, j in space.edge_pairs)


def is_connected(matrix: list[list[int]], space: SearchSpace) -> bool:
    parent = list(range(space.vertices))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        root_a = find(a)
        root_b = find(b)
        if root_a != root_b:
            parent[root_b] = root_a

    for i, j in space.edge_pairs:
        if matrix[i][j] > 0:
            union(i, j)
    return len({find(i) for i in range(space.vertices)}) == 1


def to_julia_matrix(matrix: list[list[int]]):
    global _to_julia_matrix
    if jl is None:
        raise RuntimeError("Julia is not loaded")
    if _to_julia_matrix is None:
        _to_julia_matrix = jl.seval("py_list -> [Int64(py_list[i][j]) for i in 1:length(py_list), j in 1:length(py_list[1])]")
    return _to_julia_matrix(matrix)


def evaluate(matrix: list[list[int]], cache: dict[tuple[int, ...], Evaluation], check_all_savings: bool, space: SearchSpace) -> Evaluation:
    key = matrix_key(matrix, space)
    if key in cache:
        return cache[key]

    edges = edge_count(matrix, space)
    if edges == 0 or not is_connected(matrix, space):
        result = Evaluation(0.0, None, None, None, edges, False, False, False)
        cache[key] = result
        return result

    try:
        jl_matrix = to_julia_matrix(matrix)
        graph = jl.ChipFiringGraph(jl_matrix)
        gon = int(jl.compute_gonality(graph))
    except Exception:
        result = Evaluation(0.0, None, None, None, edges, False, False, False)
        cache[key] = result
        return result

    sigma2_gon: int | None = None
    saving_checked = check_all_savings or gon >= 5
    if saving_checked:
        sigma2_gon = int(jl.compute_gonality(jl.subdivide(graph, 2)))

    if gon < 5:
        result = Evaluation(float(gon), gon, sigma2_gon, None, edges, True, saving_checked, sigma2_gon is not None and gon != sigma2_gon)
        cache[key] = result
        return result

    treewidth = int(jl.exact_treewidth(jl.SimpleGraph(graph.adj_matrix)))
    reward = 50.0 + (1000.0 / edges) - (10.0 * treewidth)
    result = Evaluation(reward, gon, sigma2_gon, treewidth, edges, True, saving_checked, sigma2_gon is not None and gon != sigma2_gon)
    cache[key] = result
    return result


def sample_matrix(theta: list[list[float]], rng: random.Random, space: SearchSpace) -> list[list[int]]:
    key: list[int] = []
    for probs in theta:
        r = rng.random()
        total = 0.0
        choice = space.n_options - 1
        for index, prob in enumerate(probs):
            total += prob
            if r <= total:
                choice = index
                break
        key.append(choice)
    return matrix_from_key(tuple(key), space)


def update_theta(theta: list[list[float]], elite_matrices: list[list[list[int]]], learning_rate: float, pseudocount: float, space: SearchSpace) -> list[list[float]]:
    counts = [[pseudocount for _ in range(space.n_options)] for _ in space.edge_pairs]
    for matrix in elite_matrices:
        for edge_index, (i, j) in enumerate(space.edge_pairs):
            counts[edge_index][matrix[i][j]] += 1.0

    updated: list[list[float]] = []
    for old_probs, edge_counts in zip(theta, counts):
        total = sum(edge_counts)
        observed = [count / total for count in edge_counts]
        updated.append([
            ((1.0 - learning_rate) * old_prob) + (learning_rate * observed_prob)
            for old_prob, observed_prob in zip(old_probs, observed)
        ])
    return updated


def write_matrix(path: Path, matrix: list[list[int]]) -> None:
    with path.open("w") as handle:
        for row in matrix:
            handle.write(" ".join(str(value) for value in row) + "\n")


def read_matrix(path: Path) -> list[list[int]]:
    rows = []
    for line in path.read_text().splitlines():
        stripped = line.strip()
        if stripped:
            rows.append([int(value) for value in stripped.split()])
    return rows


def append_jsonl(path: Path, record: dict) -> None:
    with path.open("a") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")


def next_known_gsg_id(directory: Path) -> int:
    highest = 0
    for path in directory.glob("gsg_*_gon_*_sigma2_*_v_*_e_*.txt"):
        match = re.match(r"gsg_(\d+)_gon_", path.name)
        if match:
            highest = max(highest, int(match.group(1)))
    return highest + 1


def known_gsg_filename(graph_id: int, matrix: list[list[int]], eval_result: Evaluation) -> str:
    vertices = len(matrix)
    return (
        f"gsg_{graph_id:04d}"
        f"_gon_{eval_result.gon}"
        f"_sigma2_{eval_result.sigma2_gon}"
        f"_v_{vertices}"
        f"_e_{eval_result.edge_count}.txt"
    )


def save_known_gsg_candidate(known_gsg_dir: Path, graph_id: int, matrix: list[list[int]], eval_result: Evaluation) -> Path:
    target_dir = known_gsg_dir / f"V{len(matrix)}"
    target_dir.mkdir(parents=True, exist_ok=True)
    matrix_path = target_dir / known_gsg_filename(graph_id, matrix, eval_result)
    write_matrix(matrix_path, matrix)
    return matrix_path


def save_candidate(
    output_dir: Path,
    generation: int,
    rank: int,
    matrix: list[list[int]],
    eval_result: Evaluation,
    known_gsg_dir: Path | None,
    known_gsg_id: int | None,
) -> Path | None:
    stem = f"gen_{generation:04d}_rank_{rank:02d}_gon_{eval_result.gon}_sigma2_{eval_result.sigma2_gon}_tw_{eval_result.treewidth}_e_{eval_result.edge_count}"
    matrix_path = output_dir / f"{stem}.txt"
    image_path = output_dir / f"{stem}.png"
    write_matrix(matrix_path, matrix)
    known_gsg_path = None
    if known_gsg_dir is not None and known_gsg_id is not None:
        known_gsg_path = save_known_gsg_candidate(known_gsg_dir, known_gsg_id, matrix, eval_result)

    graph = edge_list_to_multigraph(matrix_to_edges(matrix))
    draw_multigraph(graph, image_path, show=False, title="", seed=7)

    append_jsonl(
        output_dir / "savings.jsonl",
        {
            "generation": generation,
            "rank": rank,
            "reward": eval_result.reward,
            "gon": eval_result.gon,
            "sigma2_gon": eval_result.sigma2_gon,
            "treewidth": eval_result.treewidth,
            "edge_count": eval_result.edge_count,
            "matrix_file": matrix_path.name,
            "image_file": image_path.name,
            "known_gsg_file": str(known_gsg_path) if known_gsg_path is not None else None,
        },
    )
    return known_gsg_path


def savings_key(matrix: list[list[int]], space: SearchSpace, args: argparse.Namespace) -> tuple[int, ...]:
    if args.allow_isomorphic_savings:
        return matrix_key(matrix, space)
    return canonical_isomorphism_key(matrix, space, args.max_canonical_permutations)


def load_existing_known_gsg_keys(known_gsg_dir: Path | None, space: SearchSpace, args: argparse.Namespace) -> set[tuple[int, ...]]:
    keys: set[tuple[int, ...]] = set()
    if known_gsg_dir is None or args.allow_isomorphic_savings:
        return keys

    target_dir = known_gsg_dir / f"V{space.vertices}"
    if not target_dir.exists():
        return keys

    for path in target_dir.glob("gsg_*_gon_*_sigma2_*_v_*_e_*.txt"):
        matrix = read_matrix(path)
        if len(matrix) != space.vertices:
            continue
        keys.add(canonical_isomorphism_key(matrix, space, args.max_canonical_permutations))
    return keys


def run(args: argparse.Namespace) -> None:
    load_julia()
    space = make_search_space(args.vertices, args.max_multiplicity)
    rng = random.Random(args.seed)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    initial_weights = [args.zero_weight, args.one_weight]
    initial_weights.extend(args.high_multiplicity_weight for _ in range(2, space.n_options))
    weight_total = sum(initial_weights)
    theta = [[weight / weight_total for weight in initial_weights] for _ in space.edge_pairs]

    cache: dict[tuple[int, ...], Evaluation] = {}
    seen_savings = load_existing_known_gsg_keys(args.known_gsg_dir, space, args)
    preexisting_savings = len(seen_savings)
    next_gsg_id = next_known_gsg_id(args.known_gsg_dir / f"V{space.vertices}") if args.known_gsg_dir else None
    started_at = time.time()

    for generation in range(1, args.generations + 1):
        population = [sample_matrix(theta, rng, space) for _ in range(args.population_size)]
        evaluated = [(matrix, evaluate(matrix, cache, args.check_all_savings, space)) for matrix in population]
        evaluated.sort(key=lambda item: item[1].reward, reverse=True)

        n_elite = max(2, round(args.population_size * args.elite_frac))
        elite_matrices = [matrix for matrix, _ in evaluated[:n_elite]]
        theta = update_theta(theta, elite_matrices, args.learning_rate, args.pseudocount, space)

        best_matrix, best_eval = evaluated[0]
        avg_reward = sum(result.reward for _, result in evaluated) / len(evaluated)
        connected = sum(1 for _, result in evaluated if result.connected)
        savings_this_gen = 0

        for rank, (matrix, eval_result) in enumerate(evaluated[: args.save_top], start=1):
            key = savings_key(matrix, space, args)
            if eval_result.is_saving and key not in seen_savings:
                seen_savings.add(key)
                savings_this_gen += 1
                known_gsg_id = None
                if next_gsg_id is not None:
                    known_gsg_id = next_gsg_id
                    next_gsg_id += 1
                save_candidate(output_dir, generation, rank, matrix, eval_result, args.known_gsg_dir, known_gsg_id)

        append_jsonl(
            output_dir / "generations.jsonl",
            {
                "generation": generation,
                "vertices": space.vertices,
                "max_multiplicity": space.max_multiplicity,
                "best_reward": best_eval.reward,
                "avg_reward": avg_reward,
                "best_gon": best_eval.gon,
                "best_sigma2_gon": best_eval.sigma2_gon,
                "best_treewidth": best_eval.treewidth,
                "best_edge_count": best_eval.edge_count,
                "connected": connected,
                "cache_size": len(cache),
                "savings_found_total": len(seen_savings) - preexisting_savings,
                "preexisting_known_gsgs": preexisting_savings,
            },
        )

        if generation % args.progress_every == 0 or savings_this_gen:
            elapsed = time.time() - started_at
            print(
                f"gen={generation} "
                f"best={best_eval.reward:.3f} "
                f"avg={avg_reward:.3f} "
                f"gon={best_eval.gon} "
                f"sigma2={best_eval.sigma2_gon} "
                f"tw={best_eval.treewidth} "
                f"edges={best_eval.edge_count} "
                f"connected={connected}/{args.population_size} "
                f"savings_total={len(seen_savings) - preexisting_savings} "
                f"elapsed={elapsed:.1f}s",
                flush=True,
            )

        if args.stop_after_savings is not None and len(seen_savings) - preexisting_savings >= args.stop_after_savings:
            break

    print(f"Done. Savings candidates found: {len(seen_savings) - preexisting_savings}")
    print(f"Output directory: {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate multigraph candidates for gonality savings.")
    parser.add_argument("--vertices", type=int, default=7, help="Number of graph vertices. Default: 7.")
    parser.add_argument("--max-multiplicity", type=int, default=2, help="Maximum edge multiplicity between two vertices. Default: 2.")
    parser.add_argument("--generations", type=int, default=500)
    parser.add_argument("--population-size", type=int, default=120)
    parser.add_argument("--elite-frac", type=float, default=0.05)
    parser.add_argument("--learning-rate", type=float, default=0.3)
    parser.add_argument("--pseudocount", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--output-dir", type=Path, default=Path("gonality_savings_runs"))
    parser.add_argument(
        "--known-gsg-dir",
        type=Path,
        help="Also export savings as known_gsgs-style adjacency matrices under this directory, e.g. known_gsgs/V7/gsg_0001_gon_6_sigma2_5_v_7_e_15.txt.",
    )
    parser.add_argument("--progress-every", type=int, default=1)
    parser.add_argument("--save-top", type=int, default=10, help="Check/save savings among this many top-ranked graphs per generation.")
    parser.add_argument("--stop-after-savings", type=int)
    parser.add_argument("--check-all-savings", action="store_true", help="Also compute sigma_2 gonality for graphs with gon(G) < 5.")
    parser.add_argument("--allow-isomorphic-savings", action="store_true", help="Disable canonical isomorphism filtering when saving savings candidates.")
    parser.add_argument("--max-canonical-permutations", type=int, default=100000, help="Maximum vertex relabelings allowed for exact canonical labeling.")
    parser.add_argument("--zero-weight", type=float, default=2.0)
    parser.add_argument("--one-weight", type=float, default=1.5)
    parser.add_argument("--high-multiplicity-weight", type=float, default=1.0, help="Initial weight for multiplicities 2 through --max-multiplicity.")
    args = parser.parse_args()

    run(args)


if __name__ == "__main__":
    main()
