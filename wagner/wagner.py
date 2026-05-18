#!/usr/bin/env python3
import os
import time
import math
import pickle
import random
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt



# ==========================================
# Julia Environment Setup
# ==========================================
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

# --- PyTorch Imports ---
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim



# ==========================================
# Hyperparameters & Constants
# ==========================================
N = 7   # Number of vertices in the graph
MYN = int(N * (N - 1) / 2)  # Length of the word (edges in complete graph)

EDGE_MULTIPLICITY = 2
MAX_EDGES_TO_CHECK = 18

LEARNING_RATE = 0.005
N_SESSIONS = 200        # Number of new sessions per iteration
PERCENTILE = 90         # Top 100-X percentile we learn from
SUPER_PERCENTILE = 92   # Top 100-X percentile that survives to next iteration

FIRST_LAYER_NEURONS = 128
SECOND_LAYER_NEURONS = 64
THIRD_LAYER_NEURONS = 4

OBSERVATION_SPACE = 2 * MYN
LEN_GAME = MYN
INF = 1000000

# Pre-calculate upper triangle indices for rapid adjacency matrix generation
TRIU_I, TRIU_J = np.triu_indices(N, k=1)

# OPTIMIZATION 1: Force CPU. 
# For tiny networks, the PCIe bus transfer to/from the GPU is slower than the actual math.
device = torch.device("cpu")
print(f"Using compute device: {device} (Optimized for small MLP latency)")


# ==========================================
# PyTorch Model Definition
# ==========================================
class GraphGenerator(nn.Module):
    def __init__(self, obs_space, edge_multiplicity):
        super(GraphGenerator, self).__init__()
        self.fc1 = nn.Linear(obs_space, FIRST_LAYER_NEURONS)
        self.fc2 = nn.Linear(FIRST_LAYER_NEURONS, SECOND_LAYER_NEURONS)
        self.fc3 = nn.Linear(SECOND_LAYER_NEURONS, THIRD_LAYER_NEURONS)
        self.out = nn.Linear(THIRD_LAYER_NEURONS, edge_multiplicity + 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        # Note: nn.CrossEntropyLoss handles the softmax internally during training.
        return self.out(x)

# Initialize model, optimizer, and loss function
model = GraphGenerator(OBSERVATION_SPACE, EDGE_MULTIPLICITY).to(device)
optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE)
loss_fn = nn.CrossEntropyLoss()

print(model)

# ==========================================
# Helper Functions
# ==========================================

# REALLY IMPORTANT: THIS WILL JUST NOT COMPUTE ANY GRAPH THAT IS NOT "SPARSE", BECAUSE WE DON'T CARE ABOUT THEM.
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


def generate_session(agent, n_sessions, verbose=False):
    agent.eval() 
    
    states = np.zeros([n_sessions, OBSERVATION_SPACE, LEN_GAME], dtype=int)
    actions = np.zeros([n_sessions, LEN_GAME], dtype=int)
    state_next = np.zeros([n_sessions, OBSERVATION_SPACE], dtype=int)

    states[:, MYN, 0] = 1
    total_score = np.zeros([n_sessions])

    times = {'pred': 0, 'play': 0, 'score': 0, 'record': 0}

    for step in range(1, LEN_GAME + 1):
        tic = time.time()
        current_states = states[:, :, step - 1]
        
        # OPTIMIZATION 2: torch.as_tensor avoids copying data. No .to(device) calls.
        current_states_t = torch.as_tensor(current_states, dtype=torch.float32)
        with torch.no_grad():
            logits = agent(current_states_t)
            # F.softmax directly to numpy array
            prob = F.softmax(logits, dim=-1).numpy() 
            
        times['pred'] += time.time() - tic

        tic = time.time()
        cum_probs = np.cumsum(prob, axis=1)
        random_rolls = np.random.rand(n_sessions, 1)
        chosen_actions = np.argmax(random_rolls < cum_probs, axis=1)

        actions[:, step - 1] = chosen_actions

        state_next = current_states.copy()
        mask = chosen_actions > 0
        state_next[mask, step - 1] = chosen_actions[mask]
        state_next[:, MYN + step - 1] = 0

        if step < LEN_GAME:
            state_next[:, MYN + step] = 1
        times['play'] += time.time() - tic

        terminal = step == LEN_GAME

        tic = time.time()
        if terminal:
            for i in range(n_sessions):
                total_score[i] = calcScore(state_next[i])
        times['score'] += time.time() - tic

        tic = time.time()
        if not terminal:
            states[:, :, step] = state_next
        times['record'] += time.time() - tic

    if verbose:
        print(f"Predict: {times['pred']:.2f}s, Play: {times['play']:.2f}s, "
              f"ScoreCalc: {times['score']:.2f}s, Record: {times['record']:.2f}s")

    return states, actions, total_score


def select_elites(states_batch, actions_batch, rewards_batch, percentile=50):
    reward_threshold = np.percentile(rewards_batch, percentile)
    elite_indices = np.where(rewards_batch >= reward_threshold)[0]
    elite_states = states_batch[elite_indices].reshape(-1, OBSERVATION_SPACE)
    elite_actions = actions_batch[elite_indices].reshape(-1)
    return elite_states, elite_actions


def select_super_sessions(states_batch, actions_batch, rewards_batch, percentile=90):
    reward_threshold = np.percentile(rewards_batch, percentile)
    super_indices = np.where(rewards_batch >= reward_threshold)[0]
    return states_batch[super_indices], actions_batch[super_indices], rewards_batch[super_indices]

# ==========================================
# Main Training Loop
# ==========================================
if __name__ == "__main__":
    super_states = np.empty((0, LEN_GAME, OBSERVATION_SPACE), dtype=int)
    super_actions = np.empty((0, LEN_GAME), dtype=int)
    super_rewards = np.array([])

    myRand = random.randint(0, 1000)
    print(f"Run ID: {myRand}")

    for i in range(1000000):
        tic = time.time()
        sessions = generate_session(model, N_SESSIONS, verbose=False)
        sessgen_time = time.time() - tic

        tic = time.time()
        states_batch = np.transpose(sessions[0], axes=[0, 2, 1])
        actions_batch = sessions[1]
        rewards_batch = sessions[2]

        if len(super_states) > 0:
            states_batch = np.vstack((states_batch, super_states))
            actions_batch = np.vstack((actions_batch, super_actions))
            rewards_batch = np.concatenate((rewards_batch, super_rewards))
        randomcomp_time = time.time() - tic

        tic = time.time()
        elite_states, elite_actions = select_elites(states_batch, actions_batch, rewards_batch, percentile=PERCENTILE)
        select1_time = time.time() - tic

        tic = time.time()
        super_states, super_actions, super_rewards = select_super_sessions(states_batch, actions_batch, rewards_batch, percentile=SUPER_PERCENTILE)
        select2_time = time.time() - tic

        tic = time.time()
        sort_indices = np.argsort(super_rewards)[::-1]
        super_states = super_states[sort_indices]
        super_actions = super_actions[sort_indices]
        super_rewards = super_rewards[sort_indices]
        select3_time = time.time() - tic

        tic = time.time()
        
        # OPTIMIZATION 3: Manual barebones batching instead of DataLoader.
        model.train() 
        t_states = torch.as_tensor(elite_states, dtype=torch.float32)
        t_actions = torch.as_tensor(elite_actions, dtype=torch.long)
        
        batch_size = 32
        dataset_size = len(t_states)
        indices = torch.randperm(dataset_size)
        
        for start_idx in range(0, dataset_size, batch_size):
            batch_idx = indices[start_idx : start_idx + batch_size]
            batch_states = t_states[batch_idx]
            batch_actions = t_actions[batch_idx]
            
            optimizer.zero_grad()
            logits = model(batch_states)
            loss = loss_fn(logits, batch_actions)
            loss.backward()
            optimizer.step()
            
        fit_time = time.time() - tic

        tic = time.time()
        mean_all_reward = np.mean(np.sort(rewards_batch)[-100:])
        mean_best_reward = np.mean(super_rewards)
        score_time = time.time() - tic

        print(f"\nGeneration {i}. Best individuals: {np.sort(super_rewards)[::-1][:10]}")
        print(f"Mean reward: {mean_all_reward:.4f} | Sessgen: {sessgen_time:.2f}s | Fit: {fit_time:.2f}s")
        if len(super_actions) > 0:
            print(f"Best graph edges: {super_actions[0]}")

        # Logging
        if i % 20 == 1:
            with open(f'best_species_pickle_{myRand}.pkl', 'wb') as fp:
                pickle.dump(super_actions, fp)
            with open(f'best_species_txt_{myRand}.txt', 'w') as f:
                for item in super_actions:
                    f.write(f"{item}\n")
            with open(f'best_species_rewards_{myRand}.txt', 'w') as f:
                for item in super_rewards:
                    f.write(f"{item}\n")
            with open(f'best_100_rewards_{myRand}.txt', 'a') as f:
                f.write(f"{mean_all_reward}\n")
            with open(f'best_elite_rewards_{myRand}.txt', 'a') as f:
                f.write(f"{mean_best_reward}\n")

        if i % 200 == 2:
            with open(f'best_species_timeline_txt_{myRand}.txt', 'a') as f:
                f.write(f"{super_actions[0]}\n")