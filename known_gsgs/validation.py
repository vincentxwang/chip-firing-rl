import os
import shutil
import re
import networkx as nx
from juliacall import Main as jl

# Load Julia code
jl.include("compute_gonality.jl")


# CHANGE THIS !!!!!!
folder_to_vertices = {
    # "V7": [7],
    # "V8": [8],
    # "V9": [9]
    "V10": [10],
    # "V14": [14],
    # "V16": [16],
    # "V18": [18]
}

# -------------------------------------------------------------------
# JULIA CHIP-FIRING FUNCTIONS
# -------------------------------------------------------------------

def compute_gonality(multigraph_matrix):
    to_julia_matrix = jl.seval("py_list -> [Int64(py_list[i][j]) for i in 1:length(py_list), j in 1:length(py_list[1])]")
    jl_matrix = to_julia_matrix(multigraph_matrix)
    g = jl.ChipFiringGraph(jl_matrix)
    return jl.compute_gonality(g)

def check_non_gsg(multigraph_matrix, gonality):
    to_julia_matrix = jl.seval("py_list -> [Int64(py_list[i][j]) for i in 1:length(py_list), j in 1:length(py_list[1])]")
    jl_matrix = to_julia_matrix(multigraph_matrix)
    g = jl.ChipFiringGraph(jl_matrix)

    genus = jl.compute_genus(g)
    n = len(multigraph_matrix)

    if n <= 5: return True
    if gonality <= 4: return True
    if genus <= 5: return True

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

def compute_gon_2_subdivision(multigraph_matrix):
    to_julia_matrix = jl.seval("py_list -> [Int64(py_list[i][j]) for i in 1:length(py_list), j in 1:length(py_list[1])]")
    jl_matrix = to_julia_matrix(multigraph_matrix)
    g = jl.ChipFiringGraph(jl_matrix)
    gonality = jl.compute_gonality(g)

    if check_non_gsg(multigraph_matrix, gonality):
        return gonality
    
    rank_to_check = gonality
    while rank_to_check > 0:
        if jl.compute_gonality(jl.subdivide(g, 2), min_d=rank_to_check, max_d=rank_to_check) == -1:
            return rank_to_check + 1
        rank_to_check -= 1

    return jl.compute_gonality(jl.subdivide(g, 2))

# -------------------------------------------------------------------
# HELPER FUNCTIONS
# -------------------------------------------------------------------

def read_matrix_from_file(filepath):
    matrix = []
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                matrix.append([int(x) for x in line.split()])
    return matrix

def count_edges(matrix):
    """Counts edges in a symmetric adjacency/multiplicity matrix."""
    return sum(sum(row) for row in matrix) // 2

def matrix_to_multigraph(matrix):
    """Converts a symmetric multiplicity matrix into a NetworkX MultiGraph."""
    G = nx.MultiGraph()
    num_nodes = len(matrix)
    G.add_nodes_from(range(num_nodes))
    for i in range(num_nodes):
        for j in range(i, num_nodes):
            multiplicity = matrix[i][j]
            for _ in range(multiplicity):
                G.add_edge(i, j)
    return G

# -------------------------------------------------------------------
# MAIN VALIDATION SCRIPT
# -------------------------------------------------------------------



def validate_known_gsgs(folder_path):
    if not os.path.exists(folder_path):
        print(f"❌ Error: Folder '{folder_path}' does not exist.")
        return

    folder_name = os.path.basename(os.path.normpath(folder_path))
    expected_vertices_list = folder_to_vertices.get(folder_name, None)

    txt_files = sorted([f for f in os.listdir(folder_path) if f.endswith(".txt")])
    if not txt_files:
        print(f"No .txt files found in '{folder_path}'.")
        return

    print(f"Validating {len(txt_files)} graphs in '{folder_path}'...")
    print("=" * 70)

    # Regex to extract metadata: e.g., gsg_0077_gon_6_sigma2_5_v_7_e_15.txt
    filename_pattern = re.compile(r"gon_(\d+)_sigma2_(\d+)_v_(\d+)_e_(\d+)\.txt")

    bad_files = []          # List of tuples: (filename, reason)
    valid_graphs = {}       # filename -> nx.MultiGraph (for isomorphism check)

    for filename in txt_files:
        filepath = os.path.join(folder_path, filename)
        print(f"Checking: {filename}...")

        # 1. Filename Parse Validation
        match = filename_pattern.search(filename)
        if not match:
            bad_files.append((filename, "Filename does not match expected format"))
            continue
        
        expected_gon = int(match.group(1))
        expected_sigma2 = int(match.group(2))
        expected_v = int(match.group(3))
        expected_e = int(match.group(4))

        # 2. Folder-Vertex Validation
        if expected_vertices_list and expected_v not in expected_vertices_list:
             bad_files.append((filename, f"Vertices ({expected_v}) do not match folder rules {expected_vertices_list}"))
             continue

        # 3. Matrix Structure Validation
        matrix = read_matrix_from_file(filepath)
        actual_v = len(matrix)
        if actual_v != expected_v:
            bad_files.append((filename, f"Matrix dimensions ({actual_v}) do not match filename v ({expected_v})"))
            continue
        
        actual_e = count_edges(matrix)
        if actual_e != expected_e:
            bad_files.append((filename, f"Matrix edges ({actual_e}) do not match filename e ({expected_e})"))
            continue

        # 4. Gonality & Savings Validation
        try:
            actual_gon = compute_gonality(matrix)
            actual_sigma2 = compute_gon_2_subdivision(matrix)

            if actual_gon != expected_gon:
                bad_files.append((filename, f"Computed gon({actual_gon}) != filename gon({expected_gon})"))
                continue
            
            if actual_sigma2 != expected_sigma2:
                bad_files.append((filename, f"Computed sigma2({actual_sigma2}) != filename sigma2({expected_sigma2})"))
                continue

            if actual_gon == actual_sigma2:
                bad_files.append((filename, "Graph does NOT have gonality savings (gon == sigma2)"))
                continue

        except Exception as e:
            bad_files.append((filename, f"Julia computation error: {e}"))
            continue

        # If it passed all the above, add to valid pool for isomorphism check
        valid_graphs[filename] = matrix_to_multigraph(matrix)

    # 5. Isomorphism Validation
    print("-" * 70)
    print(f"Checking {len(valid_graphs)} structurally valid graphs for isomorphism...")
    
    isomorphic_classes = [] 
    for filename, G in valid_graphs.items():
        found_class = False
        for group in isomorphic_classes:
            representative_filename = group[0]
            representative_graph = valid_graphs[representative_filename]
            
            if nx.is_isomorphic(G, representative_graph):
                group.append(filename)
                found_class = True
                break
                
        if not found_class:
            isomorphic_classes.append([filename])

    # Flag duplicates as bad files
    for group in isomorphic_classes:
        if len(group) > 1:
            group.sort() # Keep the first one, flag the rest
            duplicates = group[1:]
            for dup in duplicates:
                bad_files.append((dup, f"Isomorphic duplicate of {group[0]}"))

    # 6. Evict Bad Files
    print("=" * 70)
    print("CLEANUP LOG")
    print("=" * 70)
    
    current_dir = os.getcwd()

    if not bad_files:
        print("✅ All files are valid! No files were moved.")
    else:
        for filename, reason in bad_files:
            src = os.path.join(folder_path, filename)
            dst = os.path.join(current_dir, filename)
            
            # Avoid overwriting files in the current dir if they already exist
            if os.path.exists(dst):
                base, ext = os.path.splitext(filename)
                dst = os.path.join(current_dir, f"{base}_bad{ext}")

            try:
                shutil.move(src, dst)
                print(f"  🚨 Moved: {filename}")
                print(f"     Reason: {reason}")
            except Exception as e:
                print(f"  ❌ Failed to move {filename}: {e}")

    # 7. Summary
    print("=" * 70)
    print("SUMMARY")
    print(f"Total graphs evaluated: {len(txt_files)}")
    print(f"Total passed: {len(txt_files) - len(bad_files)}")
    print(f"Total failed & moved out: {len(bad_files)}")
    print("=" * 70)


if __name__ == "__main__":
    # Specify the target folder to validate
    # target_folder = "../known_gsgs/V7" 
    # validate_known_gsgs(target_folder)
    
    # Example: Run it on all folders listed in your dictionary
    base_dir = "known_gsgs"
    for folder_name in folder_to_vertices.keys():
        target_path = os.path.join(base_dir, folder_name)
        if os.path.exists(target_path):
            validate_known_gsgs(target_path)
            print("\n")