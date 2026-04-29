using Pkg
Pkg.add("ChipFiring")
Pkg.add("Graphs")
Pkg.add("TreeWidthSolver")

using ChipFiring
using Graphs
using TreeWidthSolver

multiplicity_matrix = [
    0 2 0 1;
    2 0 1 0;
    0 1 0 1;
    1 0 1 0   
]

g = ChipFiringGraph(multiplicity_matrix)

println(compute_gonality(g))

function test(a,b)
    return a + b
end

