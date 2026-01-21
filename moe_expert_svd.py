#!/usr/bin/env python3
"""
FFN Clustering with Shared V-Matrix SVD for MoE Models

This script implements and analyzes clustering of Feed-Forward Networks (FFN) layers
in Mixture of Experts (MoE) models using SVD decomposition and V-matrix sharing strategies.

## Overview
- Extract FFN layers from MoE models
- Apply SVD decomposition to FFN weight matrices
- Cluster FFNs based on similarity
- Analyze potential for shared V-matrices within clusters
- Evaluate compression ratios and clustering quality
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from scipy.spatial.distance import pdist, squareform
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)
torch.manual_seed(42)

print("Libraries imported successfully!")


# ============================================================================
# Section 1: Load MoE Model and Extract FFN Layers
# ============================================================================
# We'll load a MoE-based language model and extract all FFN layers for analysis.
# Common MoE models include:
# - Qwen MoE models
# - Mixtral models
# - Custom MoE implementations

def extract_mlp_experts(model: nn.Module):
    """Extract MLP experts from model"""
    mlp_experts = {}
    
    # Handle different model structures
    if hasattr(model, 'model') and hasattr(model.model, 'layers'):
        for layer_idx, layer in enumerate(model.model.layers):
            mlp_experts[layer_idx] = layer.mlp.experts      # list of experts

    return mlp_experts

def extract_expert_weights(mlp_experts: dict, layer_idx: int, device='cpu'):
    """Extract weight matrices from FFN experts"""
    weights_data = {'gate_proj': [], 'up_proj': [], 'down_proj': []}
    
    for mlp_expert in mlp_experts[layer_idx]:
        # Extract individual weights
        if hasattr(mlp_expert, 'gate_proj'):
            weights_data['gate_proj'].append(mlp_expert.gate_proj.weight.data.to(device))
        if hasattr(mlp_expert, 'up_proj'):
            weights_data['up_proj'].append(mlp_expert.up_proj.weight.data.to(device))
        if hasattr(mlp_expert, 'down_proj'):
            weights_data['down_proj'].append(mlp_expert.down_proj.weight.data.to(device))
    return weights_data

# Load the Qwen3-30B model from HuggingFace Hub
print("=" * 80)
print("LOADING QWEN3-30B-A3B-INSTRUCT-2507 MODEL")
print("=" * 80)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"\n✓ Using device: {device}")
if device == "cuda":
    print(f"✓ GPU: {torch.cuda.get_device_name(0)}")

try:
    print("\nAttempting to load Qwen3-30B model...")
    print("(First load may take 2-5 minutes, downloading ~60GB)\n")
    
    from transformers import AutoModelForCausalLM
    
    model_name = "Qwen/Qwen3-30B-A3B-Instruct-2507"
    
    # Load model with GPU if available
    if device == "cuda":
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto"
        )
    else:
        print("⚠️  Warning: GPU not available, loading to CPU (may be very slow)")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="cpu"
        )
    
    mlp_experts = extract_mlp_experts(model)
    print(f"✓ Model loaded successfully!")
    n_layers = len(mlp_experts)
    print(f"✓ Found {n_layers} layers with MLP experts")
    print(f"✓ Hidden size: {model.config.hidden_size}, MoE intermediate size: {model.config.moe_intermediate_size}")
        
except ImportError as e:
    print(f"⚠️  Import error: {e}")
    print("   Falling back to synthetic weights...")
    model = None
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
except Exception as e:
    print(f"❌ Error loading model: {e}")
    print("\nFalling back to synthetic weights...")
    model = None
    device = "cuda" if torch.cuda.is_available() else "cpu"

# ============================================================================
# Section 2: Perform SVD on FFN Weights
# ============================================================================
# Apply Singular Value Decomposition to decompose each FFN weight matrix
# into U, Sigma, and V components.

def compute_svd(weight_matrix, full_matrices=False):
    """Compute SVD decomposition of weight matrix"""
    # Handle GPU tensors
    # if weight_matrix.is_cuda:
    #     # Move to CPU for SVD computation (more stable)
    #     weight_matrix = weight_matrix
    
    U, S, Vt = torch.linalg.svd(weight_matrix, full_matrices=full_matrices)
    return U, S, Vt

# Compute SVD for all FFN layers
print("Computing SVD decomposition for all FFN weight matrices...")
print("This may take a few minutes depending on model size...\n")

# Extract MLP experts
# For experiment, just perform on layer 0
print("\nExtracting MLP experts...")
layer_idx = 0
expert_weights = extract_expert_weights(mlp_experts, layer_idx=layer_idx, device=device)     # dict[weight_name] -> list of weight

# Store U, S, Vt components
svd_components = {}

# for layer_idx, expert_weights_by_proj in expert_weights.items():
#     svd_components[layer_idx] = {}
#     for proj_name, weight_list in expert_weights_by_proj.items():
#         print(f"Processing Layer {layer_idx} - {proj_name}... ", end="", flush=True)
#         svd_components[layer_idx][proj_name] = {'U': [], 'S': [], 'V': []}
#         for expert_idx, weight_matrix in enumerate(weight_list):
#             U, S, Vt = compute_svd(weight_matrix=weight_matrix.to(torch.float32), full_matrices=False)
            
#             svd_components[layer_idx][proj_name]['U'].append(U)
#             svd_components[layer_idx][proj_name]['S'].append(S)
#             svd_components[layer_idx][proj_name]['V'].append(Vt)
            
for proj_name, weight_list in expert_weights.items():
    print(f"processing Layer {layer_idx} - {proj_name}... ", end="", flush=True)
    svd_components[proj_name] = {'U': [], 'S': [], 'V': []}
    for expert_idx, weight_matrix in enumerate(weight_list):
        U, S, Vt = compute_svd(weight_matrix=weight_matrix.to(torch.float32), full_matrices=False)
        U, S, Vt = U.to(torch.float16), S.to(torch.float16), Vt.to(torch.float16)
        
        svd_components[proj_name]['U'].append(U)
        svd_components[proj_name]['S'].append(S)
        svd_components[proj_name]['Vt'].append(Vt)

print(f"\n✓ SVD completed for layer {layer_idx}")

# ============================================================================
# Section 3: Expert Clustering & V-Matrix Sharing
# ============================================================================
# Using top-K alignment for similarity and greedy clustering for Vt sharing

def compute_top_k_similarity(Vt1, Vt2, k=8):
    """
    Compare top-K rows of V-matrices for similarity.
    
    Top-K rows are already ordered by importance (singular values).
    Greedy matching finds best alignment between corresponding rows.
    
    Args:
        Vt1, Vt2: V-transpose matrices [rank, features]
        k: Number of top rows to compare (most important)
    
    Returns:
        float in [0, 1]: mean alignment score of top-K rows
    """
    Vt1 = Vt1.cpu().numpy() if torch.is_tensor(Vt1) else Vt1
    Vt2 = Vt2.cpu().numpy() if torch.is_tensor(Vt2) else Vt2
    
    # Extract top-K rows (ordered by importance)
    k1 = min(k, Vt1.shape[0])
    k2 = min(k, Vt2.shape[0])
    
    V1_top = Vt1[:k1, :]
    V2_top = Vt2[:k2, :]
    
    # Normalize rows (each row is unit vector)
    V1_norm = V1_top / (np.linalg.norm(V1_top, axis=1, keepdims=True) + 1e-8)
    V2_norm = V2_top / (np.linalg.norm(V2_top, axis=1, keepdims=True) + 1e-8)
    
    # Greedy matching: for each row in V1, find best match in V2
    sim_matrix = np.abs(V1_norm @ V2_norm.T)  # [k1, k2]
    best_matches = sim_matrix.max(axis=1)     # Best match per V1 row
    
    return float(np.mean(best_matches))


def compute_similarity_matrix(Vt_list, k=8):
    """
    Compute pairwise top-K similarity matrix for all experts.
    
    Args:
        Vt_list: List of V-transpose matrices
        k: Number of top rows to use
    
    Returns:
        [n_experts, n_experts] similarity matrix
    """
    n = len(Vt_list)
    sim_matrix = np.zeros((n, n))
    
    print(f"\nComputing top-K={k} V-matrix similarity...")
    
    for i in range(n):
        for j in range(i, n):
            sim = compute_top_k_similarity(Vt_list[i], Vt_list[j], k=k)
            sim_matrix[i, j] = sim
            sim_matrix[j, i] = sim
            
            if (i + 1) % 5 == 0 or i == n - 1:
                print(f"  Progress: {i + 1}/{n} experts processed")
    
    return sim_matrix


def greedy_clustering(sim_matrix, threshold=0.85):
    """
    Greedy clustering: group experts with similarity >= threshold.
    
    Uses greedy connected-components approach:
    - Start with first unassigned expert
    - Add all experts with sim >= threshold as same cluster
    - Move to next unassigned expert
    
    Args:
        sim_matrix: [n_experts, n_experts] similarity matrix
        threshold: Similarity threshold for same cluster
    
    Returns:
        List[List[int]]: clusters, each is list of expert indices
    """
    n = sim_matrix.shape[0]
    assigned = set()
    clusters = []
    
    for i in range(n):
        if i in assigned:
            continue
        
        # Start new cluster with expert i
        cluster = [i]
        queue = [i]
        assigned.add(i)
        
        # BFS: find all connected experts
        while queue:
            current = queue.pop(0)
            
            for j in range(n):
                if j not in assigned and sim_matrix[current, j] >= threshold:
                    cluster.append(j)
                    queue.append(j)
                    assigned.add(j)
        
        clusters.append(sorted(cluster))
    
    return clusters


def assign_shared_vt(clusters, Vt_list, S_list, method='max_svd'):
    """
    Assign a shared V-matrix to each cluster.
    
    Methods:
    - 'max_svd': Use V-matrix of expert with highest singular values (sum(S))
    - 'average': Average all V-matrices in cluster (approximate)
    
    Args:
        clusters: List[List[int]] from greedy_clustering
        Vt_list: List of all V-matrices
        S_list: List of all singular value vectors
        method: How to select shared V-matrix
    
    Returns:
        Dict: {cluster_id: shared_Vt_tensor, ...}
    """
    shared_vt_map = {}
    
    for cluster_id, members in enumerate(clusters):
        if method == 'max_svd':
            # Find expert with highest sum of singular values
            best_expert = max(members, key=lambda i: S_list[i].sum().item())
            shared_vt = Vt_list[best_expert]
            
        elif method == 'average':
            # Average V-matrices (need to handle shape differences)
            # Pad all to max shape, average, then trim
            max_rows = max(Vt_list[i].shape[0] for i in members)
            max_cols = max(Vt_list[i].shape[1] for i in members)
            
            vt_padded = []
            for i in members:
                v = Vt_list[i].cpu().numpy() if torch.is_tensor(Vt_list[i]) else Vt_list[i]
                v_pad = np.zeros((max_rows, max_cols))
                v_pad[:v.shape[0], :v.shape[1]] = v
                vt_padded.append(v_pad)
            
            avg_vt = np.mean(vt_padded, axis=0)
            shared_vt = torch.from_numpy(avg_vt).float()
        
        else:
            raise ValueError(f"Unknown method: {method}")
        
        shared_vt_map[cluster_id] = shared_vt
    
    return shared_vt_map


# ============================================================================
# Analyze V-matrix Similarity & Find Clusters
# ============================================================================

print("\n" + "=" * 80)
print("SECTION 3: EXPERT CLUSTERING & V-MATRIX SHARING")
print("=" * 80)

if 'down_proj' in svd_components:
    Vt_list = svd_components['down_proj']['Vt']
    S_list = svd_components['down_proj']['S']
    
    # Compute top-K similarity
    sim_matrix = compute_similarity_matrix(Vt_list, k=8)
    
    print(f"\n✓ Similarity matrix computed for {len(Vt_list)} experts\n")
    
    # Print similarity matrix
    print("TOP-K V-MATRIX SIMILARITY SCORES:\n")
    for i in range(len(Vt_list)):
        print(f"Expert {i}: {sim_matrix[i].round(4)}")
    print()
    
    # Similarity statistics
    print("\n" + "=" * 80)
    print("SIMILARITY STATISTICS")
    print("=" * 80)
    
    upper_tri = np.triu_indices_from(sim_matrix, k=1)
    similarities = sim_matrix[upper_tri]
    
    print(f"Mean similarity: {np.mean(similarities):.4f}")
    print(f"Std deviation:  {np.std(similarities):.4f}")
    print(f"Min:            {np.min(similarities):.4f}")
    print(f"Max:            {np.max(similarities):.4f}")
    print()
    
    # Find most similar pairs
    print("Most similar expert pairs:")
    sorted_pairs = []
    for i in range(len(Vt_list)):
        for j in range(i + 1, len(Vt_list)):
            sorted_pairs.append((sim_matrix[i, j], i, j))
    sorted_pairs.sort(reverse=True)
    
    for sim, i, j in sorted_pairs[:5]:
        print(f"  Expert {i} ↔ Expert {j}: {sim:.4f}")
    
    # ========================================================================
    # Clustering with multiple thresholds
    # ========================================================================
    
    print("\n" + "=" * 80)
    print("CLUSTERING RESULTS")
    print("=" * 80)
    
    best_clustering = None
    best_threshold = None
    
    for threshold in [0.9, 0.85, 0.80, 0.75, 0.70, 0.65]:
        clusters = greedy_clustering(sim_matrix, threshold=threshold)
        
        print(f"\nThreshold {threshold}: {len(clusters)} clusters")
        for cid, members in enumerate(clusters):
            avg_sim = np.mean([sim_matrix[members[i], members[j]] 
                              for i in range(len(members)) 
                              for j in range(i+1, len(members))])
            avg_sim = avg_sim if len(members) > 1 else 1.0
            print(f"  Cluster {cid}: {members} (internal avg sim: {avg_sim:.4f})")
        
        # Recommend this clustering if reasonable
        if best_clustering is None and len(clusters) <= len(Vt_list) // 2 + 1:
            best_clustering = clusters
            best_threshold = threshold
    
    # ========================================================================
    # Assign shared V-matrix to each cluster
    # ========================================================================
    
    print("\n" + "=" * 80)
    print("SHARED V-MATRIX ASSIGNMENT")
    print("=" * 80)
    
    if best_clustering is None:
        best_clustering = greedy_clustering(sim_matrix, threshold=0.65)
        best_threshold = 0.65
    
    shared_vt_map = assign_shared_vt(best_clustering, Vt_list, S_list, method='max_svd')
    
    print(f"\nUsing threshold {best_threshold} → {len(best_clustering)} clusters\n")
    
    for cluster_id, members in enumerate(best_clustering):
        shared_expert_idx = max(members, key=lambda i: S_list[i].sum().item())
        print(f"Cluster {cluster_id}:")
        print(f"  Members: {members}")
        print(f"  Shared Vt from: Expert {shared_expert_idx}")
        print(f"  Shared Vt shape: {shared_vt_map[cluster_id].shape}")
        print(f"  Singular values (sum): {[S_list[i].sum().item():.2f} for i in members]}")
        print()

