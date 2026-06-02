import os
import shutil
import re
import numpy as np
from juliacall import Main as jl

# 1. Load your base Julia code
jl.include("compute_gonality.jl")

# 2. Define the heavy logic directly in Julia ONCE to prevent JIT/memory crashes
jl.seval("""
function check_non_gsg_wrapper(mat_in::AbstractMatrix{Int64}, gonality::Int)
    # Convert the Python/Numpy array into a standard Julia Matrix
    mat = Matrix(mat_in) 
    
    g = ChipFiringGraph(mat)
    genus = compute_genus(g)
    n = size(mat, 1)

    if n <= 5 return true end
    if gonality <= 4 return true end
    if genus <= 5 return true end

    num_edges_gbar = 0
    for i in 1:n
        for j in (i + 1):n
            if mat[i, j] > 0
                num_edges_gbar += 1
            end
        end
    end
    
    n_choose_2 = div(n * (n - 1), 2)
    if num_edges_gbar > n_choose_2 - (n - 2)
        return true
    end

    return false
end

function get_gonality_and_savings(mat_in::AbstractMatrix{Int64})
    # Convert the Python/Numpy array into a standard Julia Matrix
    mat = Matrix(mat_in)
    g = ChipFiringGraph(mat)
    
    # Base Gonality
    gon = compute_gonality(g)
    
    # If no savings possible, return early
    if check_non_gsg_wrapper(mat, gon)
        return gon, gon 
    end
    
    rank_to_check = gon
    while rank_to_check > 0
        if compute_gonality(subdivide(g, 2), min_d=rank_to_check, max_d=rank_to_check) == -1
            return gon, rank_to_check + 1
        end
        rank_to_check -= 1
    end
    
    sub_gon = compute_gonality(subdivide(g, 2))
    return gon, sub_gon
end
""")
# -------------------------------------------------------------------
# PYTHON HELPER FUNCTIONS
# -------------------------------------------------------------------

def compute_graph_stats(multigraph_matrix):
    """Passes the matrix to the Julia wrapper safely via Numpy to get both metrics."""
    np_matrix = np.array(multigraph_matrix, dtype=np.int64)
    gon, sub_gon = jl.get_gonality_and_savings(np_matrix)
    return int(gon), int(sub_gon)

def prune_leaves(matrix):
    """
    Iteratively removes nodes of degree 1 (leaves) and degree 0 (isolated).
    A node is a leaf if the sum of its row/column is exactly 1.
    """
    current_mat = np.array(matrix)
    
    while True:
        if current_mat.shape[0] <= 1:
            break
            
        # The sum of a row is the degree of the node
        degrees = np.sum(current_mat, axis=1)
        
        # Find leaves (degree == 1) or isolated nodes (degree == 0)
        nodes_to_remove = np.where((degrees == 1) | (degrees == 0))[0]
        
        if len(nodes_to_remove) == 0:
            break
            
        # Keep only the nodes that have degree > 1
        keep_indices = np.where((degrees > 1))[0]
        
        # Slice the matrix to keep only valid rows and columns
        current_mat = current_mat[keep_indices][:, keep_indices]
        
    return current_mat.tolist()

def process_and_prune_folder(target_folder):
    if not os.path.exists(target_folder):
        print(f"❌ Error: Folder '{target_folder}' does not exist.")
        return

    # Create a safe place to output the newly computed pruned matrices
    output_folder = "./pruned_graphs_output"
    os.makedirs(output_folder, exist_ok=True)
    
    current_dir = os.getcwd()
    txt_files = sorted([f for f in os.listdir(target_folder) if f.endswith(".txt")])

    if not txt_files:
        print(f"No .txt files found in '{target_folder}'.")
        return

    print(f"Scanning {len(txt_files)} graphs in '{target_folder}' for leaves...")
    print("=" * 70)

    # Regex to extract the original ID so we can track it
    filename_pattern = re.compile(r"gsg_(.*?)_gon")
    
    leaves_found_count = 0

    for filename in txt_files:
        filepath = os.path.join(target_folder, filename)
        
        # 1. Read matrix and check for leaves
        matrix = []
        with open(filepath, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    matrix.append([int(x) for x in line.split()])
        
        np_mat = np.array(matrix)
        degrees = np.sum(np_mat, axis=1)
        
        # If no node has a degree of exactly 1, skip it
        if not np.any(degrees == 1):
            continue
            
        # --- A LEAF WAS FOUND ---
        leaves_found_count += 1
        print(f"🍂 Leaf found in: {filename}")
        
        # 2. Extract original ID
        match = filename_pattern.search(filename)
        original_id = match.group(1) if match else "unknown"

        # 3. Move the original file out of the target folder
        src = filepath
        dst = os.path.join(current_dir, f"original_has_leaf_{filename}")
        shutil.move(src, dst)
        print("  -> Moved original file to current directory.")

        # 4. Prune the matrix
        pruned_matrix = prune_leaves(matrix)
        v_new = len(pruned_matrix)
        
        if v_new <= 2:
            print(f"  -> Graph pruned to nothing (it was a tree). Discarding.")
            continue
            
        e_new = sum(sum(row) for row in pruned_matrix) // 2
        print(f"  -> Pruned matrix from {len(matrix)} vertices to {v_new} vertices.")

        # 5. Compute new quantities via Julia
        try:
            new_gon, new_sigma2 = compute_graph_stats(pruned_matrix)
            
            # Format the new filename: gsg_{ID}_pruned_gon_X_sigma2_Y_v_V_e_E.txt
            new_filename = f"gsg_P{original_id}_gon_{new_gon}_sigma2_{new_sigma2}_v_{v_new}_e_{e_new}.txt"
            new_filepath = os.path.join(output_folder, new_filename)
            
            # 6. Save the new matrix
            with open(new_filepath, 'w') as f:
                for row in pruned_matrix:
                    f.write(" ".join(map(str, row)) + "\n")
                    
            print(f"  -> Saved recomputed graph: {new_filename}")
            
        except Exception as e:
            print(f"  ❌ Error recomputing Julia stats for {filename}: {e}")

    print("=" * 70)
    print("SUMMARY")
    print(f"Total graphs scanned: {len(txt_files)}")
    print(f"Graphs containing leaves (removed & pruned): {leaves_found_count}")
    if leaves_found_count > 0:
        print(f"New pruned graphs saved to: {output_folder}")
    print("=" * 70)

if __name__ == "__main__":
    # Point this to the folder you want to clean
    target_dir = "known_gsgs/V10"
    process_and_prune_folder(target_dir)