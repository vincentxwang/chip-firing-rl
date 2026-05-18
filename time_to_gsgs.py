#!/usr/bin/env python3
"""
Measure time to generate unique gonality-saving graphs.

This is a timing-oriented wrapper around gonality_savings_generator.py. It uses
the same cross-entropy sampler and reward, but reports how long it takes to find
N new non-isomorphic savings graphs and exports them in known_gsgs format.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import gonality_savings_generator as gen


def run_trial(args: argparse.Namespace) -> dict:
    gen.load_julia()

    space = gen.make_search_space(args.vertices, args.max_multiplicity)
    rng = random.Random(args.seed)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    initial_weights = [args.zero_weight, args.one_weight]
    initial_weights.extend(args.high_multiplicity_weight for _ in range(2, space.n_options))
    weight_total = sum(initial_weights)
    theta = [[weight / weight_total for weight in initial_weights] for _ in space.edge_pairs]

    cache: dict[tuple[int, ...], gen.Evaluation] = {}
    seen_savings = gen.load_existing_known_gsg_keys(args.known_gsg_dir, space, args)
    preexisting_count = len(seen_savings)
    next_gsg_id = gen.next_known_gsg_id(args.known_gsg_dir / f"V{space.vertices}")
    started_at = time.time()

    generated_count = 0
    evaluated_count = 0
    connected_count = 0

    for generation in range(1, args.max_generations + 1):
        population = [gen.sample_matrix(theta, rng, space) for _ in range(args.population_size)]
        evaluated = [(matrix, gen.evaluate(matrix, cache, args.check_all_savings, space)) for matrix in population]
        evaluated.sort(key=lambda item: item[1].reward, reverse=True)

        evaluated_count += len(evaluated)
        connected_count += sum(1 for _, result in evaluated if result.connected)

        n_elite = max(2, round(args.population_size * args.elite_frac))
        elite_matrices = [matrix for matrix, _ in evaluated[:n_elite]]
        theta = gen.update_theta(theta, elite_matrices, args.learning_rate, args.pseudocount, space)

        generation_new = 0
        for rank, (matrix, eval_result) in enumerate(evaluated[: args.save_top], start=1):
            if not eval_result.is_saving:
                continue

            key = gen.savings_key(matrix, space, args)
            if key in seen_savings:
                continue

            seen_savings.add(key)
            generation_new += 1
            generated_count += 1
            known_path = gen.save_known_gsg_candidate(args.known_gsg_dir, next_gsg_id, matrix, eval_result)
            next_gsg_id += 1

            gen.save_candidate(
                output_dir,
                generation,
                rank,
                matrix,
                eval_result,
                known_gsg_dir=None,
                known_gsg_id=None,
            )
            gen.append_jsonl(
                output_dir / "timing_savings.jsonl",
                {
                    "generation": generation,
                    "rank": rank,
                    "elapsed_seconds": time.time() - started_at,
                    "new_savings_found": generated_count,
                    "known_gsg_file": str(known_path),
                    "gon": eval_result.gon,
                    "sigma2_gon": eval_result.sigma2_gon,
                    "treewidth": eval_result.treewidth,
                    "edge_count": eval_result.edge_count,
                    "reward": eval_result.reward,
                },
            )

            if generated_count >= args.target_count:
                break

        best_eval = evaluated[0][1]
        elapsed = time.time() - started_at
        if generation % args.progress_every == 0 or generation_new or generated_count >= args.target_count:
            print(
                f"V={space.vertices} max_mult={space.max_multiplicity} "
                f"gen={generation} "
                f"new={generated_count}/{args.target_count} "
                f"best={best_eval.reward:.3f} "
                f"gon={best_eval.gon} "
                f"sigma2={best_eval.sigma2_gon} "
                f"tw={best_eval.treewidth} "
                f"edges={best_eval.edge_count} "
                f"connected={sum(1 for _, result in evaluated if result.connected)}/{args.population_size} "
                f"elapsed={elapsed:.1f}s",
                flush=True,
            )

        gen.append_jsonl(
            output_dir / "timing_generations.jsonl",
            {
                "generation": generation,
                "vertices": space.vertices,
                "max_multiplicity": space.max_multiplicity,
                "elapsed_seconds": elapsed,
                "new_savings_found": generated_count,
                "preexisting_known_gsgs": preexisting_count,
                "best_reward": best_eval.reward,
                "best_gon": best_eval.gon,
                "best_sigma2_gon": best_eval.sigma2_gon,
                "best_treewidth": best_eval.treewidth,
                "best_edge_count": best_eval.edge_count,
                "cache_size": len(cache),
                "evaluated_total": evaluated_count,
                "connected_total": connected_count,
            },
        )

        if generated_count >= args.target_count:
            break

    elapsed = time.time() - started_at
    summary = {
        "vertices": space.vertices,
        "max_multiplicity": space.max_multiplicity,
        "target_count": args.target_count,
        "new_savings_found": generated_count,
        "preexisting_known_gsgs": preexisting_count,
        "elapsed_seconds": elapsed,
        "max_generations": args.max_generations,
        "population_size": args.population_size,
        "seed": args.seed,
        "completed_target": generated_count >= args.target_count,
        "output_dir": str(output_dir),
        "known_gsg_dir": str(args.known_gsg_dir),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(f"Done: found {generated_count}/{args.target_count} new savings in {elapsed:.1f}s")
    return summary


def add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--vertices", type=int, default=7)
    parser.add_argument("--max-multiplicity", type=int, default=2)
    parser.add_argument("--target-count", type=int, default=5)
    parser.add_argument("--max-generations", type=int, default=1000)
    parser.add_argument("--population-size", type=int, default=400)
    parser.add_argument("--elite-frac", type=float, default=0.15)
    parser.add_argument("--learning-rate", type=float, default=0.08)
    parser.add_argument("--pseudocount", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--save-top", type=int, default=80)
    parser.add_argument("--progress-every", type=int, default=10)
    parser.add_argument("--known-gsg-dir", type=Path, default=Path("known_gsgs"))
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--check-all-savings", action="store_true")
    parser.add_argument("--allow-isomorphic-savings", action="store_true")
    parser.add_argument("--max-canonical-permutations", type=int, default=100000)
    parser.add_argument("--zero-weight", type=float, default=1.2)
    parser.add_argument("--one-weight", type=float, default=2.0)
    parser.add_argument("--high-multiplicity-weight", type=float, default=1.4)


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure time to generate unique gonality-saving graphs.")
    add_common_args(parser)
    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = Path(f"time_to_gsgs_V{args.vertices}_m{args.max_multiplicity}_seed{args.seed}")

    run_trial(args)


if __name__ == "__main__":
    main()
