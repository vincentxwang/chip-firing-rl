#!/usr/bin/env python3
"""
DiGress-based generator for gonality savings graph candidates.

Trains a discrete denoising diffusion model (DiGress) on known genus-9
trivalent GSGs from known_gsgs.txt, then generates new candidate multigraphs.
Candidates that pass structural validation (trivalent, V=16, E=24, connected)
are saved for gon/ugon verification via ChipFiring.jl.

Requires: torch, torch_geometric, pytorch_lightning, networkx, matplotlib
GPU recommended. Designed to run in Google Colab with T4 runtime.
"""

import copy
import os
import re
import random

import matplotlib.pyplot as plt
import networkx as nx

NUM_NODE_TYPES = 1
NUM_EDGE_TYPES = 3  # {no edge, single, double}
NUM_NODES = 16
TARGET_EDGES = 24


def parse_known_gsgs(filepath, target_genus=9):
    with open(filepath, "r") as f:
        lines = f.readlines()

    start_idx = None
    end_idx = len(lines)
    for i, line in enumerate(lines):
        if line.startswith("#") and f"trivalent genus {target_genus} multigraphs" in line:
            start_idx = i
        elif start_idx is not None and line.startswith("#") and "trivalent genus" in line:
            end_idx = i
            break

    if start_idx is None:
        raise ValueError(f"No section found for genus {target_genus}")

    pattern = re.compile(
        r'Graph\(V=(\d+), E=(\d+), Edges=\[(.*?)\]\)"\s*\(gon1=(\d+), gon2=(\d+)\)'
    )
    edge_pat = re.compile(r'\((\d+),\s*(\d+)\)')

    graphs = []
    for line in lines[start_idx:end_idx]:
        m = pattern.search(line)
        if not m:
            continue
        V, E, edges_str, gon1, gon2 = m.groups()
        V, E, gon1, gon2 = int(V), int(E), int(gon1), int(gon2)

        G = nx.MultiGraph()
        G.add_nodes_from(range(1, V + 1))
        for u, v in edge_pat.findall(edges_str):
            G.add_edge(int(u), int(v))
        G.graph["gon1"] = gon1
        G.graph["gon2"] = gon2
        graphs.append(G)

    return graphs


def multigraph_to_pyg(G):
    """Convert NetworkX MultiGraph to PyG Data object for DiGress."""
    import torch
    from torch_geometric.data import Data

    nodes = sorted(G.nodes())
    n = len(nodes)
    X = torch.ones(n, NUM_NODE_TYPES, dtype=torch.float)
    y = torch.zeros([1, 0]).float()

    edge_indices = []
    edge_attrs = []
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            mult = G.number_of_edges(nodes[i], nodes[j])
            if mult > 0:
                edge_indices.append([i, j])
                attr = [0.0] * NUM_EDGE_TYPES
                attr[min(mult, NUM_EDGE_TYPES - 1)] = 1.0
                edge_attrs.append(attr)

    if edge_indices:
        edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attrs, dtype=torch.float)
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_attr = torch.zeros((0, NUM_EDGE_TYPES), dtype=torch.float)

    return Data(
        x=X, edge_index=edge_index, edge_attr=edge_attr,
        y=y, n_nodes=n * torch.ones(1, dtype=torch.long),
    )


def check_validity(G):
    """Check trivalent, correct size, connected."""
    if G.number_of_nodes() != NUM_NODES:
        return False
    if G.number_of_edges() != TARGET_EDGES:
        return False
    if not all(G.degree(v) == 3 for v in G.nodes()):
        return False
    if G.number_of_nodes() > 0 and not nx.is_connected(G):
        return False
    return True


def graph_to_edge_list(G):
    V = G.number_of_nodes()
    E = G.number_of_edges()
    edges = sorted(G.edges())
    edge_str = ", ".join(f"({u}, {v})" for u, v in edges)
    return f"Graph(V={V}, E={E}, Edges=[{edge_str}])"


def dense_sample_to_multigraph(atom_types, edge_types):
    """Convert DiGress dense output back to NetworkX MultiGraph."""
    n = atom_types.shape[0]
    G = nx.MultiGraph()
    G.add_nodes_from(range(1, n + 1))
    for i in range(n):
        for j in range(i + 1, n):  # upper triangle only
            mult = int(edge_types[i, j].item())
            for _ in range(mult):
                G.add_edge(i + 1, j + 1)
    return G


def edge_swap(G, num_swaps=5):
    H = copy.deepcopy(G)
    edges = list(H.edges(keys=True))
    for _ in range(num_swaps):
        if len(edges) < 2:
            break
        idx1, idx2 = random.sample(range(len(edges)), 2)
        u1, v1, k1 = edges[idx1]
        u2, v2, k2 = edges[idx2]
        if len({u1, v1, u2, v2}) < 4:
            continue
        H.remove_edge(u1, v1, key=k1)
        H.remove_edge(u2, v2, key=k2)
        H.add_edge(u1, u2)
        H.add_edge(v1, v2)
        edges = list(H.edges(keys=True))
    return H


def generate_edge_swap_candidates(graphs, n_candidates=50, swaps=5, max_attempts=5000):
    candidates = []
    seen = set()
    attempts = 0
    while len(candidates) < n_candidates and attempts < max_attempts:
        source = random.choice(graphs)
        mutant = edge_swap(source, num_swaps=swaps)
        if not check_validity(mutant):
            attempts += 1
            continue
        sig = tuple(sorted(mutant.edges()))
        if sig in seen:
            attempts += 1
            continue
        seen.add(sig)
        candidates.append(mutant)
        attempts += 1
    return candidates


def save_candidates(graphs, path, header=""):
    with open(path, "w") as f:
        if header:
            f.write(f"# {header}\n\n")
        for i, G in enumerate(graphs):
            f.write(f'Candidate #{i+1}: "{graph_to_edge_list(G)}"\n')


def visualize_candidates(graphs, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    for i, G in enumerate(graphs):
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        pos = nx.spring_layout(G, seed=42)
        nx.draw(G, pos, ax=ax, with_labels=True,
                node_color='red', node_size=400,
                font_size=10, font_color='white',
                edge_color='black', width=1.5)
        V = G.number_of_nodes()
        E = G.number_of_edges()
        ax.set_title(f"Candidate #{i+1} (V={V}, E={E})")
        plt.savefig(f"{output_dir}/candidate_{i+1}.png", dpi=150, bbox_inches='tight')
        plt.close()


def print_summary(generated, valid, known):
    known_sigs = {tuple(sorted(G.edges())) for G in known}
    novel = sum(1 for G in valid if tuple(sorted(G.edges())) not in known_sigs)
    n = len(generated)
    v = len(valid)
    print(f"Generated: {n}")
    print(f"Structurally valid: {v}/{n} ({100*v/n:.1f}%)")
    print(f"Novel: {novel}/{v}")


if __name__ == "__main__":
    # quick test
    graphs = parse_known_gsgs("known_gsgs.txt", target_genus=9)
    print(f"Loaded {len(graphs)} genus 9 GSGs")

    random.seed(42)
    candidates = generate_edge_swap_candidates(graphs, n_candidates=5)
    print(f"Generated {len(candidates)} test candidates")
    for i, G in enumerate(candidates):
        print(f"  #{i+1}: valid={check_validity(G)}, {graph_to_edge_list(G)[:60]}...")
