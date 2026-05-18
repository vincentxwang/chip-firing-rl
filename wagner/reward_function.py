## Note: You can blackbox everything here that is not calcScore. calcScore takes in the edge list (basically the upper half of the multiplicity matrix.) it directly converts this to a 
# multiplicity matrix in the function so you can use that to figure out how to pass things appropriately into calcScore.

import numpy as np

N = 7   # Number of vertices in the graph
MYN = int(N * (N - 1) / 2)  # Length of the word (edges in complete graph)

EDGE_MULTIPLICITY = 2
MAX_EDGES_TO_CHECK = 18

TRIU_I, TRIU_J = np.triu_indices(N, k=1)

from juliacall import Main as jl

jl.seval("import Pkg")

# Install required Julia packages
packages = ["ChipFiring", "Graphs", "TreeWidthSolver"]
for pkg in packages:
    jl.seval(f'if !haskey(Pkg.project().dependencies, "{pkg}") Pkg.add("{pkg}") end')

# Initialize Julia libraries exactly ONCE
jl.seval("using ChipFiring")
jl.seval("using Graphs")
jl.seval("using TreeWidthSolver")

jl.seval("""
function fast_convert(np_array)
    return convert(Matrix{Int64}, np_array)
end
""")
fast_convert = jl.fast_convert


def compute_gon_2_subdivision(g, gonality, n, num_nonzero_edges):
    """
    Computes gonality of the 2-subdivision using backtracking.
    Receives the pre-built Julia graph 'g' to prevent redundant memory allocation.
    """
    # 1. Check easy conditions
    if check_non_gsg(g, gonality, n, num_nonzero_edges):
        return gonality

    sub_g = jl.subdivide(g, 2)
    
    rank_to_check = gonality

    while rank_to_check > 0:
        if jl.compute_gonality(sub_g, min_d=rank_to_check, max_d=rank_to_check) == -1:
            return rank_to_check + 1
        rank_to_check -= 1

    print("if this is printed vincent screwed up")
    return jl.compute_gonality(sub_g)


def check_non_gsg(g, gonality, n, num_edges):
    """Checks for Gonality Savings Graphs efficiently."""
    
    if n <= 5:
        return True
    if gonality <= 4:
        return True

    # THIS CONDITION IS BOGUS, BUT NEED TO WORK WELL
    if num_edges > MAX_EDGES_TO_CHECK:
        return True

    genus = jl.compute_genus(g)
    if genus <= 5:
        return True

    return False


def is_connected_fast(edges, N):
    """fast DSU connectivity check."""
    parent = list(range(N))

    def find(i):
        if parent[i] == i:
            return i
        parent[i] = find(parent[i])
        return parent[i]

    def union(i, j):
        root_i = find(i)
        root_j = find(j)
        if root_i != root_j:
            parent[root_i] = root_j

    for idx in range(len(edges)):
        if edges[idx] > 0:
            union(TRIU_I[idx], TRIU_J[idx])

    root = find(0)
    return all(find(i) == root for i in range(1, N))


# Create the cache dictionary globally
score_cache = {}

def calcScore(state):
    """Calculates the reward for a given word using gonality and treewidth."""
    edges = state[:MYN]
    state_key = tuple(edges)

    if state_key in score_cache:
        return score_cache[state_key]

    num_edges_total_weight = np.sum(edges)

    if num_edges_total_weight == 0 or not is_connected_fast(edges, N):
        score_cache[state_key] = 0.0
        return 0.0

    # Build Adjacency Matrix
    adj = np.zeros((N, N), dtype=int)
    adj[TRIU_I, TRIU_J] = edges
    adj[TRIU_J, TRIU_I] = edges 

    try:
        # 1. Build the Julia Object EXACTLY ONCE here
        jl_matrix = fast_convert(adj)
        graph = jl.ChipFiringGraph(jl_matrix)
        
        # 2. Calculate Standard Gonality
        gon = int(jl.compute_gonality(graph))

        # 3. Fast Exit
        if gon < 5:
            score_cache[state_key] = float(gon)
            return float(gon)
    
        # 4. Count unique non-zero edges (ignores multiplicity of 2) instantly
        num_edges = np.sum(edges)
        
        # 5. Calculate Subdivision Gonality (passing pre-built data)
        gon2 = compute_gon_2_subdivision(graph, gon, N, num_edges)

        # 6. Calculate Treewidth & Reward
        treewidth = int(jl.exact_treewidth(jl.SimpleGraph(graph.adj_matrix)))
        reward = 50.0 + (1000.0 / num_edges_total_weight) - (10.0 * treewidth) + 100 * (gon - gon2) + np.random.normal(0, 0.1)

        score_cache[state_key] = reward

        # print(reward)

        return reward

    except Exception as e:
        print(f"Failed to calculate score: {e}")
        score_cache[state_key] = 0.0
        return 0.0

print(calcScore([1,1,0,2,0,1,1,1,0,1,0,1,0,1,0,1,0,1,0,0,2]))