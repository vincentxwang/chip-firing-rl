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
    gonality = jl.compute_gonality(jl.subdivide(g, 2))
    return gonality



# demo

python_matrix = [
    [0, 2, 0, 1],
    [2, 0, 1, 0],
    [0, 1, 0, 1],
    [1, 0, 1, 0]   
]

py_gonality = compute_gonality(python_matrix)
py_gonality2 = compute_gon_2_subdivision(python_matrix)

print(f"Gonality computed from Python: {py_gonality}")
print(f"UGonality computed from Python: {py_gonality2}")