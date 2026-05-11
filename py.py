## this is some boiler plate code just to call compute_gonality from Julia. 

from juliacall import Main as jl

jl.include("compute_gonality.jl")


# computes gon(G)
def compute_gonality(multigraph_matrix):
    to_julia_matrix = jl.seval("py_list -> [Int64(py_list[i][j]) for i in 1:length(py_list), j in 1:length(py_list[1])]")
    jl_matrix = to_julia_matrix(multigraph_matrix)
    g = jl.ChipFiringGraph(jl_matrix)
    gonality = jl.compute_gonality(g)
    return gonality


# This does not compute uniform gonality (ugon(G)), but this quantity is >= uniform gonality 
# to our knowledge, for all known graphs, this quantity is equal to uniform gonality
def compute_gon_2_subdivision(multigraph_matrix):
    to_julia_matrix = jl.seval("py_list -> [Int64(py_list[i][j]) for i in 1:length(py_list), j in 1:length(py_list[1])]")
    jl_matrix = to_julia_matrix(multigraph_matrix)
    g = jl.ChipFiringGraph(jl_matrix)
    gonality = jl.compute_gonality(g)

    # check easy conditions
    if check_non_gsg(multigraph_matrix, gonality):
        return gonality
    
    # backtracking from gonality should be easier to compute?

    rank_to_check = gonality

    while rank_to_check > 0:
        if jl.compute_gonality(jl.subdivide(g, 2), min_d=rank_to_check, max_d=rank_to_check, verbose=True) == -1:
            return rank_to_check + 1
        rank_to_check -= 1
    
    print("if this is printed vincent screwed up")

    return jl.compute_gonality(jl.subdivide(g, 2))

def compute_treewidth(multigraph_matrix):
    to_julia_matrix = jl.seval("py_list -> [Int64(py_list[i][j]) for i in 1:length(py_list), j in 1:length(py_list[1])]")
    jl_matrix = to_julia_matrix(multigraph_matrix)
    g = jl.ChipFiringGraph(jl_matrix)
    tw = jl.exact_treewidth(jl.SimpleGraph(g.adj_matrix))
    return tw

# returns TRUE if we have determined that the graph is not a gonality savings graph. otherwise returns false (indeterminate if so).
def check_non_gsg(multigraph_matrix, gonality):

    to_julia_matrix = jl.seval("py_list -> [Int64(py_list[i][j]) for i in 1:length(py_list), j in 1:length(py_list[1])]")
    jl_matrix = to_julia_matrix(multigraph_matrix)
    g = jl.ChipFiringGraph(jl_matrix)

    genus = jl.compute_genus(g)
    n = len(multigraph_matrix)

    # <= 5 vertices?
    if n <= 5:
        return True

    # gonality <= 4? 
    if gonality <= 4:
        return True
    
    # genus <= 5? Theorem 5.8
    if genus <= 5:
        return True
    

    #### check edge condition here! (lemma 5.12)

    num_edges_gbar = sum(
        1 
        for i in range(n) 
        for j in range(i + 1, n) 
        if multigraph_matrix[i][j] > 0
    )
    
    n_choose_2 = (n * (n - 1)) // 2
    
    if num_edges_gbar > n_choose_2 - (n - 2):
        return True

    return False

# demo

python_matrix = [
    [0, 2, 0, 1],
    [2, 0, 1, 0],
    [0, 1, 0, 1],
    [1, 0, 1, 0]   
]

python_matrix = [
    [0, 2, 0, 1, 1, 1, 1],
    [2, 0, 2, 0, 1, 1, 0],
    [0, 2, 0, 0, 0, 0, 0],
    [1, 0, 0, 0, 2, 0, 0],
    [1, 1, 0, 2, 0, 0, 1],
    [1, 1, 0, 0, 0, 0, 2],
    [1, 0, 0, 0, 1, 2, 0]
]

py_gonality = compute_gonality(python_matrix)
py_gonality2 = compute_gon_2_subdivision(python_matrix)
py_tw = compute_treewidth(python_matrix)
py_check_non_gsg = check_non_gsg(python_matrix, py_gonality)

print(f"Gonality computed from Python: {py_gonality}")
print(f"Gonsig2G computed from Python: {py_gonality2}")
print(f"TW computed from Python: {py_tw}")
print(f"Check computed from Python: {py_check_non_gsg}")