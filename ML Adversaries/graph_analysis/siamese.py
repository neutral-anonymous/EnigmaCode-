#!/usr/bin/env python3
"""Siamese Network analysis"""
import os
import pickle
import numpy as np
import time
import argparse
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Union
from collections import defaultdict, Counter
import itertools
from scipy import stats
from scipy.stats import skew
import multiprocessing as mp
from functools import partial
import random

# PyTorch with maximum optimization
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

# System optimization
# CPU_CORES = os.cpu_count() or 8
# torch.set_num_threads(min(CPU_CORES, 32))  # Maximum threading
# print(f"Using {min(CPU_CORES, 32)} CPU threads for PyTorch operations")

# Reproducibility
random.seed(1337)
np.random.seed(1337)
torch.manual_seed(1337)

# CONFIGURATION  

SERVER_DATA_CONFIGS = {
    "prepared_dataset": {
        "base_path": "/data/prepared_dataset",
        "files": {
            "gallery": "gallery_index.pkl",
            "opt2_train": "pairs_option2_train.pkl",  # ALL 10K (original, enigma_1) pairs
            "opt2_test": "pairs_option2_test.pkl",    # 500 enigma_2 test graphs
        }
    }
}

# Optimized hyperparameter grids (focused on best combinations)
HYPERPARAMETER_GRIDS = {
    "siamese_nre": {
        "hidden_dims": [128, 256],
        "num_layers": [3, 4], 
        "learning_rates": [0.001, 0.01],
        "dropout_rates": [0.2, 0.3],
        "temperature": [0.07, 0.1],  # InfoNCE temperature
        "loss_type": ["supervised_contrastive", "batch_triplet", "infonce"],  # Loss function options
        "epochs": [100],  # Faster iterations for debugging
        "patience": [15]
    },
    "siamese_simgnn": {
        "hidden_dims": [128, 256],
        "num_layers": [3, 4],
        "attention_heads": [4, 8],
        "learning_rates": [0.001, 0.01],
        "dropout_rates": [0.2, 0.3],
        "temperature": [0.07, 0.1],
        "loss_type": ["supervised_contrastive", "batch_triplet"],  # Exclude InfoNCE for other models
        "epochs": [100],
        "patience": [15]
    },
    "siamese_gin": {
        "hidden_dims": [128, 256],
        "num_layers": [3, 4, 5],
        "learning_rates": [0.001, 0.01],
        "dropout_rates": [0.2],
        "temperature": [0.07, 0.1],
        "loss_type": ["supervised_contrastive", "batch_triplet"],
        "epochs": [100], 
        "patience": [15]
    }
}

# ULTRA-OPTIMIZED DATA LOADING WITH COMPREHENSIVE FEATURES

def load_qubo_from_file(filepath):
    """Load QUBO with comprehensive error handling and key normalization"""
    try:
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
    except Exception:
        return {}
    
    # Normalize structure: prefer 'Q' key, else use dict directly
    if isinstance(data, dict):
        if 'Q' in data and isinstance(data['Q'], dict):
            return data['Q']
        return data
    if hasattr(data, 'items'):
        return dict(data.items())
    return {}

def parse_qubo_key(key):
    """Parse all possible QUBO key formats to (i, j) tuple"""
    if isinstance(key, tuple) and len(key) == 2:
        try:
            return int(key[0]), int(key[1])
        except Exception:
            return None
    if isinstance(key, str):
        for sep in [',', '_', '-', ' ', ':']:
            if sep in key:
                parts = key.split(sep)
                if len(parts) == 2:
                    try:
                        return int(parts[0]), int(parts[1])
                    except Exception:
                        continue
        try:
            i = int(key)
            return i, i  # Diagonal element
        except Exception:
            return None
    if isinstance(key, (int, float)):
        try:
            i = int(key)
            return i, i
        except Exception:
            return None
    return None

def exhaustive_flatten(val):
    """Flatten any value type to scalar float"""
    if val is None:
        return 0.0
    if isinstance(val, (int, float, np.integer, np.floating)):
        try:
            v = float(val)
            if np.isnan(v) or np.isinf(v):
                return 0.0
            return v
        except Exception:
            return 0.0
    if isinstance(val, str):
        try:
            return float(val)
        except Exception:
            return 0.0
    if isinstance(val, dict):
        nums = [exhaustive_flatten(v) for v in val.values()]
        return float(np.mean(nums)) if nums else 0.0
    if isinstance(val, (list, tuple, np.ndarray)):
        try:
            flat = np.array(val).flatten()
            nums = [exhaustive_flatten(x) for x in flat]
            return float(np.mean(nums)) if nums else 0.0
        except Exception:
            return 0.0
    return 0.0

def comprehensive_qubo_to_graph(qubo_mapping, max_nodes=100):
    """Convert QUBO to graph with comprehensive 10D node features"""
    if not isinstance(qubo_mapping, dict) or len(qubo_mapping) == 0:
        # Return minimal valid graph
        return {
            'node_features': np.zeros((1, 10), dtype=np.float32),
            'adj_matrix': np.zeros((1, 1), dtype=np.float32)
        }
    
    # FIXED: Handle Ising format correctly
    h_terms = qubo_mapping.get('h', {})
    j_terms = qubo_mapping.get('J', {})
    
    # If it's not Ising format, fall back to old parsing
    if 'h' not in qubo_mapping and 'J' not in qubo_mapping:
        # Extract nodes and edges using old method
        nodes = set()
        edges = []
        edge_weights = []
        node_weights = {}
        
        for key, value in qubo_mapping.items():
            parsed = parse_qubo_key(key)
            if parsed is None:
                continue
            i, j = parsed
            val = exhaustive_flatten(value)
            nodes.add(i)
            nodes.add(j)
            if i == j:
                node_weights[i] = val
            else:
                edges.append((i, j))
                edge_weights.append(val)
    else:
        # NEW: Proper Ising format parsing
        nodes = set()
        edges = []
        edge_weights = []
        node_weights = {}
        
        # Process linear terms (h)
        if isinstance(h_terms, dict):
            for var_idx, coeff in h_terms.items():
                try:
                    idx = int(var_idx)
                    nodes.add(idx)
                    node_weights[idx] = float(coeff)
                except:
                    continue
        elif isinstance(h_terms, (list, np.ndarray)):
            for idx, coeff in enumerate(h_terms):
                if coeff != 0:
                    nodes.add(idx)
                    node_weights[idx] = float(coeff)
        
        # Process quadratic terms (J)
        if isinstance(j_terms, dict):
            for var_pair, coeff in j_terms.items():
                try:
                    if isinstance(var_pair, tuple) and len(var_pair) == 2:
                        i, j = int(var_pair[0]), int(var_pair[1])
                    elif isinstance(var_pair, str) and ',' in var_pair:
                        i, j = map(int, var_pair.split(','))
                    else:
                        continue
                    
                    nodes.add(i)
                    nodes.add(j)
                    if i != j:  # Only add edge if not diagonal
                        edges.append((i, j))
                        edge_weights.append(float(coeff))
                    else:
                        # Diagonal quadratic term - add to node weight
                        node_weights[i] = node_weights.get(i, 0) + float(coeff)
                except:
                    continue
    
    if not nodes:
        return {
            'node_features': np.zeros((1, 10), dtype=np.float32),
            'adj_matrix': np.zeros((1, 1), dtype=np.float32)
        }
    
    # Limit graph size for memory efficiency
    node_list = sorted(list(nodes))
    if len(node_list) > max_nodes:
        node_list = node_list[:max_nodes]
    
    node_index = {n: idx for idx, n in enumerate(node_list)}
    n_nodes = len(node_list)
    
    # Build adjacency matrix
    adj = np.zeros((n_nodes, n_nodes), dtype=np.float32)
    for (i, j), w in zip(edges, edge_weights):
        if i in node_index and j in node_index:
            ii, jj = node_index[i], node_index[j]
            adj[ii, jj] = w
            adj[jj, ii] = w  # Symmetric
    
    # Comprehensive 10D node features
    node_features = np.zeros((n_nodes, 10), dtype=np.float32)
    for idx in range(n_nodes):
        # Basic graph properties
        degree = np.sum(adj[idx] != 0)
        weighted_degree = np.sum(np.abs(adj[idx]))
        
        # Neighbor weight statistics
        neighbors = adj[idx][adj[idx] != 0]
        w_mean = float(np.mean(neighbors)) if len(neighbors) > 0 else 0.0
        w_std = float(np.std(neighbors)) if len(neighbors) > 0 else 0.0
        w_max = float(np.max(neighbors)) if len(neighbors) > 0 else 0.0
        w_min = float(np.min(neighbors)) if len(neighbors) > 0 else 0.0
        
        # Advanced features
        pos_edges = np.sum(adj[idx] > 0)
        neg_edges = np.sum(adj[idx] < 0)
        node_centrality = float(idx) / max(n_nodes, 1)  # Normalized position
        
        node_features[idx] = np.array([
            node_weights.get(node_list[idx], 0.0),  # Diagonal value
            degree,                                  # Node degree
            weighted_degree,                         # Weighted degree
            w_mean,                                  # Mean edge weight
            w_std,                                   # Std edge weight
            w_max,                                   # Max edge weight
            w_min,                                   # Min edge weight
            pos_edges,                               # Positive edges
            neg_edges,                               # Negative edges
            node_centrality                          # Normalized position
        ], dtype=np.float32)
    
    # Add self-loops for better GNN propagation
    adj = adj + np.eye(n_nodes, dtype=np.float32)
    
    # Extract graph-level features - avoid infinite recursion by using basic features
    graph_features = np.concatenate([
        np.array([n_nodes, np.sum(adj > 0)]),  # Basic structure
        np.mean(node_features, axis=0),        # Average node features  
        np.std(node_features, axis=0),         # Node feature diversity
        np.array([np.trace(adj), np.sum(adj)]) # Matrix properties
    ])
    
    # Pad to 70 dimensions
    if len(graph_features) < 70:
        graph_features = np.concatenate([graph_features, np.zeros(70 - len(graph_features))])
    else:
        graph_features = graph_features[:70]
    
    return {
        'node_features': node_features,
        'adj_matrix': adj,
        'graph_features': graph_features.astype(np.float32)
    }

def extract_comprehensive_features(qubo_mapping):
    """Extract comprehensive 70D graph-level features with enhanced discriminative power"""
    if not isinstance(qubo_mapping, dict) or len(qubo_mapping) == 0:
        return np.zeros(70, dtype=np.float32)
    
    # Convert to graph for structural analysis
    graph_data = comprehensive_qubo_to_graph(qubo_mapping)
    adj_matrix = graph_data['adj_matrix']
    node_features = graph_data['node_features']
    n_nodes = adj_matrix.shape[0]
    
    features = []
    
    # 1. BASIC QUBO STATISTICS (20 features)
    values = []
    diagonal_terms = []
    off_diagonal_terms = []
    
    # FIXED: Handle both Ising format and old parsed format
    if 'h' in qubo_mapping or 'J' in qubo_mapping:
        # Ising format: extract values from h and J
        h_terms = qubo_mapping.get('h', {})
        j_terms = qubo_mapping.get('J', {})
        
        # Process linear terms (h)
        if isinstance(h_terms, dict):
            for var_idx, coeff in h_terms.items():
                try:
                    val = float(coeff)
                    values.append(val)
                    diagonal_terms.append(val)  # h terms are diagonal
                except:
                    continue
        elif isinstance(h_terms, (list, np.ndarray)):
            for coeff in h_terms:
                if coeff != 0:
                    values.append(float(coeff))
                    diagonal_terms.append(float(coeff))
        
        # Process quadratic terms (J)
        if isinstance(j_terms, dict):
            for var_pair, coeff in j_terms.items():
                try:
                    val = float(coeff)
                    values.append(val)
                    if isinstance(var_pair, tuple) and len(var_pair) == 2:
                        i, j = var_pair
                        if i == j:
                            diagonal_terms.append(val)
                        else:
                            off_diagonal_terms.append(val)
                    else:
                        off_diagonal_terms.append(val)  # Assume off-diagonal
                except:
                    continue
    else:
        # Old format: parse keys manually
        for key, value in qubo_mapping.items():
            parsed = parse_qubo_key(key)
            if parsed is None:
                continue
            i, j = parsed
            val = exhaustive_flatten(value)
            values.append(val)
            if i == j:
                diagonal_terms.append(val)
            else:
                off_diagonal_terms.append(val)
    
    # QUBO statistics
    features.extend([
        len(qubo_mapping),                                                    # Total terms
        len(diagonal_terms),                                                  # Diagonal terms
        len(off_diagonal_terms),                                             # Off-diagonal terms
        np.mean(values) if values else 0,                                    # Mean value
        np.std(values) if values else 0,                                     # Std value
        np.min(values) if values else 0,                                     # Min value
        np.max(values) if values else 0,                                     # Max value
        np.sum(values) if values else 0,                                     # Sum values
        np.mean(diagonal_terms) if diagonal_terms else 0,                   # Mean diagonal
        np.std(diagonal_terms) if diagonal_terms else 0,                    # Std diagonal
        np.mean(off_diagonal_terms) if off_diagonal_terms else 0,           # Mean off-diagonal
        np.std(off_diagonal_terms) if off_diagonal_terms else 0,            # Std off-diagonal
        len([v for v in values if v > 0]) / len(values) if values else 0,   # Positive ratio
        len([v for v in values if v < 0]) / len(values) if values else 0,   # Negative ratio
        np.median(values) if values else 0,                                  # Median
        np.percentile(values, 25) if values else 0,                         # Q1
        np.percentile(values, 75) if values else 0,                         # Q3
        np.var(values) if values else 0,                                     # Variance
        np.sum([abs(v) for v in values]) if values else 0,                  # Sum absolute
        len(set(values)) if values else 0                                    # Unique values
    ])
    
    # 2. GRAPH STRUCTURAL FEATURES (15 features)
    if n_nodes > 1:
        n_edges = np.sum(adj_matrix != 0) // 2  # Undirected edges
        density = n_edges / (n_nodes * (n_nodes - 1) / 2) if n_nodes > 1 else 0
        
        # Degree statistics
        degrees = np.sum(adj_matrix != 0, axis=1)
        degree_mean = np.mean(degrees)
        degree_std = np.std(degrees)
        degree_max = np.max(degrees)
        
        # Spectral features
        try:
            eigenvals = np.linalg.eigvals(adj_matrix)
            eigenvals = eigenvals[np.isfinite(eigenvals)]
            spectral_radius = np.max(np.abs(eigenvals)) if len(eigenvals) > 0 else 0
            spectral_gap = (np.sort(np.abs(eigenvals))[-1] - 
                           np.sort(np.abs(eigenvals))[-2]) if len(eigenvals) > 1 else 0
            trace = np.trace(adj_matrix)
        except Exception:
            spectral_radius = spectral_gap = trace = 0
        
        # Clustering coefficient
        try:
            clustering_coeffs = []
            for i in range(n_nodes):
                neighbors = np.where(adj_matrix[i] != 0)[0]
                if len(neighbors) < 2:
                    clustering_coeffs.append(0)
                else:
                    possible_edges = len(neighbors) * (len(neighbors) - 1) / 2
                    actual_edges = np.sum(adj_matrix[np.ix_(neighbors, neighbors)] != 0) / 2
                    clustering_coeffs.append(actual_edges / possible_edges)
            avg_clustering = np.mean(clustering_coeffs)
        except Exception:
            avg_clustering = 0
        
        # Triangle count
        try:
            adj_binary = (adj_matrix != 0).astype(int)
            triangle_count = np.trace(np.linalg.matrix_power(adj_binary, 3)) // 6
        except Exception:
            triangle_count = 0
        
        features.extend([
            n_nodes, n_edges, density, degree_mean, degree_std, degree_max,
            spectral_radius, spectral_gap, trace, avg_clustering, triangle_count,
            len(np.unique(degrees)),            # Distinct degrees
            np.sum(degrees == 0),               # Isolated nodes
            np.sum(np.diag(adj_matrix) != 0),   # Self-loops
            np.sum(adj_matrix < 0) // 2         # Negative edges
        ])
    else:
        features.extend([0] * 15)
    
    # 3. WEIGHT DISTRIBUTION FEATURES (15 features)
    if values:
        abs_values = [abs(v) for v in values]
        features.extend([
            np.mean(abs_values),                                                    # Mean absolute
            np.std(abs_values),                                                     # Std absolute
            skew(values) if len(values) > 2 else 0,                               # Skewness
            np.percentile(abs_values, 90),                                         # 90th percentile
            np.percentile(abs_values, 10),                                         # 10th percentile
            len([v for v in values if abs(v) > np.mean(abs_values)]) / len(values), # Above-mean ratio
            np.max(abs_values) / (np.mean(abs_values) + 1e-8),                    # Max/mean ratio
            np.std(values) / (np.mean(abs_values) + 1e-8),                        # Std/mean ratio
            np.sum([v for v in values if v > 0]) / (np.sum(abs_values) + 1e-8),   # Positive weight ratio
            np.sum([v for v in values if v < 0]) / (np.sum(abs_values) + 1e-8),   # Negative weight ratio
            len([v for v in values if abs(v) < 1e-6]) / len(values),              # Near-zero ratio
            len([v for v in values if abs(v) > 1]) / len(values),                 # Large values ratio
            (np.mean([abs(v) for v in diagonal_terms]) / 
             (np.mean([abs(v) for v in off_diagonal_terms]) + 1e-8)) if off_diagonal_terms else 0, # Diag/off-diag ratio
            len(diagonal_terms) / (len(off_diagonal_terms) + 1),                  # Term count ratio
            (np.std(diagonal_terms) / 
             (np.std(off_diagonal_terms) + 1e-8)) if off_diagonal_terms else 0   # Std ratio
        ])
    else:
        features.extend([0] * 15)
    
    # 4. ENHANCED DISCRIMINATIVE FEATURES (20 features)
    # These are designed to capture invariant properties of original graphs
    
    # QUBO sparsity patterns
    try:
        # Variable interaction patterns
        max_var_id = max(max(parse_qubo_key(k) or (0, 0)) for k in qubo_mapping.keys())
        interaction_matrix = np.zeros((max_var_id + 1, max_var_id + 1))
        
        for key in qubo_mapping.keys():
            parsed = parse_qubo_key(key)
            if parsed:
                i, j = parsed
                interaction_matrix[i, j] = 1
                if i != j:
                    interaction_matrix[j, i] = 1
        
        # Sparsity features
        total_possible = (max_var_id + 1) ** 2
        actual_interactions = np.sum(interaction_matrix)
        sparsity_ratio = actual_interactions / total_possible if total_possible > 0 else 0
        
        # Connected components in interaction graph
        try:
            from scipy.sparse.csgraph import connected_components
            n_components, _ = connected_components(interaction_matrix, directed=False)
        except:
            n_components = 1
        
        # Variable usage frequency
        var_counts = np.zeros(max_var_id + 1)
        for key in qubo_mapping.keys():
            parsed = parse_qubo_key(key)
            if parsed:
                i, j = parsed
                var_counts[i] += 1
                if i != j:
                    var_counts[j] += 1
        
        usage_stats = [
            np.mean(var_counts) if len(var_counts) > 0 else 0,
            np.std(var_counts) if len(var_counts) > 0 else 0,
            np.max(var_counts) if len(var_counts) > 0 else 0,
            np.min(var_counts) if len(var_counts) > 0 else 0,
            len(var_counts[var_counts > 0]) / len(var_counts) if len(var_counts) > 0 else 0  # Active ratio
        ]
        
        enhanced_features = [
            max_var_id,
            sparsity_ratio,
            n_components,
            actual_interactions,
        ] + usage_stats
        
        # Coefficient correlation patterns
        try:
            diag_vals = [qubo_mapping.get(f"({i},{i})", 0) for i in range(max_var_id + 1)]
            off_diag_vals = []
            for i in range(max_var_id + 1):
                for j in range(i + 1, max_var_id + 1):
                    val = qubo_mapping.get(f"({i},{j})", 0)
                    if val != 0:
                        off_diag_vals.append(val)
            
            diag_vals = np.array([exhaustive_flatten(v) for v in diag_vals if v != 0])
            off_diag_vals = np.array([exhaustive_flatten(v) for v in off_diag_vals])
            
            diag_mean = np.mean(diag_vals) if len(diag_vals) > 0 else 0
            diag_std = np.std(diag_vals) if len(diag_vals) > 0 else 0
            off_diag_mean = np.mean(off_diag_vals) if len(off_diag_vals) > 0 else 0
            off_diag_std = np.std(off_diag_vals) if len(off_diag_vals) > 0 else 0
            
            # Ratio features to capture structure differences
            diag_off_ratio = diag_mean / off_diag_mean if off_diag_mean != 0 else 0
            std_ratio = diag_std / off_diag_std if off_diag_std != 0 else 0
            
            enhanced_features.extend([
                diag_mean, diag_std, off_diag_mean, off_diag_std,
                diag_off_ratio, std_ratio,
                len(diag_vals), len(off_diag_vals)
            ])
            
        except Exception:
            enhanced_features.extend([0] * 8)
            
    except Exception:
        enhanced_features = [0] * 20
    
    # Add enhanced features
    features.extend(enhanced_features)
    
    # Ensure exactly 70 features
    features = features[:70]
    while len(features) < 70:
        features.append(0)
    
    return np.array(features, dtype=np.float32)

# ULTRA-FAST CACHED DATASET WITH PROPER SIAMESE LEARNING

class SiameseGraphDataset(Dataset):
    """
    ULTRA-OPTIMIZED dataset for Siamese network training
    - Pre-caches ALL graphs in RAM (eliminates I/O)
    - Proper InfoNCE contrastive learning setup
    - Balanced positive/negative sampling
    """
    
    def __init__(self, records, mode='train', negative_ratio=1.0, max_cache_size=None):
        self.records = records
        self.mode = mode
        self.negative_ratio = negative_ratio
        self.graph_cache = {}
        
        print(f"🔥 Pre-caching {len(records)} graph pairs for {mode} mode...")
        start_time = time.time()
        
        # Extract all unique graph paths
        unique_paths = set()
        for record in records:
            unique_paths.add(record['original_path'])
            unique_paths.add(record['obfuscated_path'])
        
        if max_cache_size and len(unique_paths) > max_cache_size:
            print(f"⚠️ Limiting cache to {max_cache_size} graphs")
            unique_paths = list(unique_paths)[:max_cache_size]
        
        # Load and cache all unique graphs
        failed_loads = 0
        for i, path in enumerate(unique_paths):
            try:
                if i % 1000 == 0:
                    print(f"    Loading graph {i+1}/{len(unique_paths)}...")
                
                qubo = load_qubo_from_file(path)
                graph_data = comprehensive_qubo_to_graph(qubo)
                
                # Cache as tensors for immediate use
                self.graph_cache[path] = {
                    'node_features': torch.FloatTensor(graph_data['node_features']),
                    'adj_matrix': torch.FloatTensor(graph_data['adj_matrix']),
                    'features': torch.FloatTensor(extract_comprehensive_features(qubo))
                }
            except Exception as e:
                failed_loads += 1
                if failed_loads < 10:
                    print(f"    ⚠️ Failed to load {path}: {e}")
        
        # Keep ALL valid records - will load on demand if not cached
        self.valid_records = records
        
        # Create training samples for Siamese learning
        if mode == 'train':
            self.training_samples = self._create_siamese_samples()
        else:
            # For validation/test, just use positive pairs
            self.training_samples = [
                {
                    'anchor_path': r['obfuscated_path'],
                    'positive_path': r['original_path'],
                    'gallery_id': r['gallery_id'],
                    'graph_type': r.get('graph_type', 'unknown')
                }
                for r in self.valid_records
            ]
        
        cache_time = time.time() - start_time
        cache_size_mb = sum(
            (g['node_features'].numel() + g['adj_matrix'].numel() + g['features'].numel()) * 4
            for g in self.graph_cache.values()
        ) / (1024 * 1024)
        
        print(f"✅ Cached {len(self.graph_cache)} graphs in {cache_time:.1f}s")
        print(f"   Valid records: {len(self.valid_records)}/{len(records)}")
        print(f"   Training samples: {len(self.training_samples)}")
        print(f"   Cache size: {cache_size_mb:.1f} MB")
        print(f"   Failed loads: {failed_loads}")
    
    def _get_graph(self, path):
        """Get graph from cache or load on demand"""
        if path in self.graph_cache:
            return self.graph_cache[path]
        
        # Load on demand
        try:
            qubo = load_qubo_from_file(path)
            graph_data = comprehensive_qubo_to_graph(qubo)
            
            graph_dict = {
                'node_features': torch.FloatTensor(graph_data['node_features']),
                'adj_matrix': torch.FloatTensor(graph_data['adj_matrix']),
                'features': torch.FloatTensor(extract_comprehensive_features(qubo))
            }
            
            # Optionally cache for future use (if cache not full)
            if len(self.graph_cache) < 10000:  # Reasonable cache limit
                self.graph_cache[path] = graph_dict
                
            return graph_dict
        except Exception as e:
            raise RuntimeError(f"Failed to load graph {path}: {e}")
    
    def _create_siamese_samples(self):
        """Create balanced positive/negative samples for contrastive learning"""
        samples = []
        
        # Group records by gallery_id for efficient negative sampling
        gallery_groups = defaultdict(list)
        for record in self.valid_records:
            gallery_groups[record['gallery_id']].append(record)
        
        gallery_ids = list(gallery_groups.keys())
        
        # Create positive pairs
        for record in self.valid_records:
            samples.append({
                'anchor_path': record['obfuscated_path'],     # enigma_1 for train, enigma_2 for test
                'positive_path': record['original_path'],     # corresponding original
                'negative_paths': [],                         # Will be filled during training
                'gallery_id': record['gallery_id'],
                'graph_type': record.get('graph_type', 'unknown'),
                'label': 1.0
            })
        
        # For each positive, pre-select some hard negatives (different gallery_ids)
        np.random.seed(42)  # Reproducible negatives
        for sample in samples:
            current_gallery = sample['gallery_id']
            
            # Select random different gallery_ids
            other_galleries = [gid for gid in gallery_ids if gid != current_gallery]
            num_negatives = min(int(self.negative_ratio * 1), len(other_galleries))
            
            if num_negatives > 0:
                selected_galleries = np.random.choice(other_galleries, num_negatives, replace=False)
                negative_paths = []
                for neg_gallery in selected_galleries:
                    # Pick random record from this gallery
                    neg_record = np.random.choice(gallery_groups[neg_gallery])
                    negative_paths.append(neg_record['original_path'])
                sample['negative_paths'] = negative_paths
        
        return samples
    
    def __len__(self):
        return len(self.training_samples)
    
    def __getitem__(self, idx):
        """Return anchor, positive, and negatives for contrastive learning"""
        sample = self.training_samples[idx]
        
        # Get graphs (cached or load on demand)
        anchor_graph = self._get_graph(sample['anchor_path'])
        positive_graph = self._get_graph(sample['positive_path'])
        
        result = {
            'anchor_nodes': anchor_graph['node_features'],
            'anchor_adj': anchor_graph['adj_matrix'],
            'anchor_features': anchor_graph['features'],
            'positive_nodes': positive_graph['node_features'],
            'positive_adj': positive_graph['adj_matrix'], 
            'positive_features': positive_graph['features'],
            'gallery_id': sample['gallery_id'],
            'graph_type': sample['graph_type']
        }
        
        # Add negatives for training mode
        if self.mode == 'train' and sample.get('negative_paths'):
            negatives = []
            for neg_path in sample['negative_paths']:
                try:
                    neg_graph = self._get_graph(neg_path)
                    negatives.append({
                        'nodes': neg_graph['node_features'],
                        'adj': neg_graph['adj_matrix'],
                        'features': neg_graph['features']
                    })
                except Exception:
                    continue  # Skip failed negatives
            result['negatives'] = negatives
        
        return result

def siamese_collate_fn(batch):
    """Custom collate function for variable-size graphs in Siamese learning"""
    # Find max nodes across all graphs in batch
    max_nodes = 1
    for sample in batch:
        max_nodes = max(max_nodes, sample['anchor_nodes'].size(0))
        max_nodes = max(max_nodes, sample['positive_nodes'].size(0))
        if 'negatives' in sample:
            for neg in sample['negatives']:
                max_nodes = max(max_nodes, neg['nodes'].size(0))
    
    # Pad and stack anchors
    anchor_nodes_batch = []
    anchor_adj_batch = []
    anchor_features_batch = []
    
    # Pad and stack positives
    positive_nodes_batch = []
    positive_adj_batch = []
    positive_features_batch = []
    
    # Collect negatives
    all_negatives = []
    
    # Other metadata
    gallery_ids = []
    graph_types = []
    
    for sample in batch:
        # Pad anchor
        anchor_nodes = sample['anchor_nodes']
        anchor_adj = sample['anchor_adj']
        pad_size = max_nodes - anchor_nodes.size(0)
        if pad_size > 0:
            anchor_nodes = F.pad(anchor_nodes, (0, 0, 0, pad_size))
            anchor_adj = F.pad(anchor_adj, (0, pad_size, 0, pad_size))
        
        anchor_nodes_batch.append(anchor_nodes)
        anchor_adj_batch.append(anchor_adj)
        anchor_features_batch.append(sample['anchor_features'])
        
        # Pad positive
        pos_nodes = sample['positive_nodes']
        pos_adj = sample['positive_adj']
        pad_size = max_nodes - pos_nodes.size(0)
        if pad_size > 0:
            pos_nodes = F.pad(pos_nodes, (0, 0, 0, pad_size))
            pos_adj = F.pad(pos_adj, (0, pad_size, 0, pad_size))
            
        positive_nodes_batch.append(pos_nodes)
        positive_adj_batch.append(pos_adj)
        positive_features_batch.append(sample['positive_features'])
        
        # Collect negatives
        if 'negatives' in sample:
            for neg in sample['negatives']:
                neg_nodes = neg['nodes']
                neg_adj = neg['adj']
                pad_size = max_nodes - neg_nodes.size(0)
                if pad_size > 0:
                    neg_nodes = F.pad(neg_nodes, (0, 0, 0, pad_size))
                    neg_adj = F.pad(neg_adj, (0, pad_size, 0, pad_size))
                
                all_negatives.append({
                    'nodes': neg_nodes,
                    'adj': neg_adj,
                    'features': neg['features']
                })
        
        gallery_ids.append(sample['gallery_id'])
        graph_types.append(sample['graph_type'])
    
    result = {
        'anchor_nodes': torch.stack(anchor_nodes_batch),
        'anchor_adj': torch.stack(anchor_adj_batch),
        'anchor_features': torch.stack(anchor_features_batch),
        'positive_nodes': torch.stack(positive_nodes_batch),
        'positive_adj': torch.stack(positive_adj_batch),
        'positive_features': torch.stack(positive_features_batch),
        'gallery_ids': gallery_ids,
        'graph_types': graph_types
    }
    
    # Add negatives if present
    if all_negatives:
        neg_nodes = torch.stack([neg['nodes'] for neg in all_negatives])
        neg_adj = torch.stack([neg['adj'] for neg in all_negatives])
        neg_features = torch.stack([neg['features'] for neg in all_negatives])
        
        result.update({
            'negative_nodes': neg_nodes,
            'negative_adj': neg_adj,
            'negative_features': neg_features
        })
    
    return result

# MODERN SIAMESE GRAPH NEURAL NETWORKS 

class GINLayer(nn.Module):
    """Graph Isomorphism Network layer - stronger than basic GCN"""
    def __init__(self, input_dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.eps = nn.Parameter(torch.zeros(1))
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, adj):
        # GIN aggregation: h' = MLP((1 + ε) * h + sum(neighbors))
        neighbor_sum = torch.bmm(adj, x)  # Aggregate neighbors
        self_plus_neighbors = (1 + self.eps) * x + neighbor_sum
        out = self.mlp(self_plus_neighbors)
        return self.dropout(out)

class SiameseGraphEncoder(nn.Module):
    """Siamese graph encoder for learning graph embeddings"""
    def __init__(self, node_dim=10, hidden_dim=128, num_layers=3, dropout=0.2, 
                 architecture='gin', use_features=True):
        super().__init__()
        self.architecture = architecture
        self.use_features = use_features
        
        # Node embedding layers
        if architecture == 'gin':
            self.layers = nn.ModuleList([
                GINLayer(node_dim, hidden_dim, dropout)
            ] + [
                GINLayer(hidden_dim, hidden_dim, dropout) for _ in range(num_layers - 1)
            ])
        else:  # Basic GCN
            self.layers = nn.ModuleList()
            self.layers.append(nn.Linear(node_dim, hidden_dim))
            for _ in range(num_layers - 1):
                self.layers.append(nn.Linear(hidden_dim, hidden_dim))
        
        # Graph-level pooling
        self.global_pool = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Feature fusion if using both graph and node representations
        if use_features:
            self.feature_encoder = nn.Sequential(
                nn.Linear(70, hidden_dim // 2),  # 70D enhanced features to smaller dim
                nn.ReLU(),
                nn.Dropout(dropout)
            )
            self.fusion = nn.Linear(hidden_dim + hidden_dim // 2, hidden_dim)
        
        self.final_projection = nn.Linear(hidden_dim, hidden_dim)
        
    def forward(self, nodes, adj, features=None):
        # Node-level encoding
        h = nodes
        
        if self.architecture == 'gin':
            for layer in self.layers:
                h = layer(h, adj)
        else:  # Basic GCN
            for i, layer in enumerate(self.layers):
                if i > 0:
                    # Simple graph convolution
                    h = torch.bmm(adj, h)
                h = F.relu(layer(h))
        
        # Global pooling (mean across nodes)
        graph_emb = torch.mean(h, dim=1)  # [batch_size, hidden_dim]
        graph_emb = self.global_pool(graph_emb)
        
        # Feature fusion
        if self.use_features and features is not None:
            feature_emb = self.feature_encoder(features)
            graph_emb = torch.cat([graph_emb, feature_emb], dim=1)
            graph_emb = self.fusion(graph_emb)
        
        # Final projection and normalization
        graph_emb = self.final_projection(graph_emb)
        graph_emb = F.normalize(graph_emb, p=2, dim=1)  # L2 normalize for cosine similarity
        
        return graph_emb

class SiameseGraphNetwork(nn.Module):
    """Complete Siamese network for graph similarity learning"""
    def __init__(self, node_dim=10, hidden_dim=128, num_layers=3, dropout=0.2, 
                 architecture='gin', use_features=True):
        super().__init__()
        self.encoder = SiameseGraphEncoder(
            node_dim=node_dim,
            hidden_dim=hidden_dim, 
            num_layers=num_layers,
            dropout=dropout,
            architecture=architecture,
            use_features=use_features
        )
        
    def forward(self, nodes, adj, features=None):
        """Encode graph to embedding"""
        return self.encoder(nodes, adj, features)
    
    def compute_similarity(self, emb1, emb2):
        """Compute cosine similarity between embeddings"""
        return F.cosine_similarity(emb1, emb2, dim=1)

def infonce_loss(anchor_emb, positive_emb, negative_emb, temperature=0.07):
    """InfoNCE contrastive loss for Siamese learning"""
    # Compute similarities
    pos_sim = F.cosine_similarity(anchor_emb, positive_emb, dim=1) / temperature  # [B]
    
    if negative_emb is not None and negative_emb.size(0) > 0:
        # Reshape for broadcasting: anchor [B, D], negatives [N, D] -> [B, N]
        anchor_expanded = anchor_emb.unsqueeze(1)  # [B, 1, D]
        neg_expanded = negative_emb.unsqueeze(0)   # [1, N, D]
        neg_sim = F.cosine_similarity(anchor_expanded, neg_expanded, dim=2) / temperature  # [B, N]
        
        # Concatenate positive and negative similarities
        logits = torch.cat([pos_sim.unsqueeze(1), neg_sim], dim=1)  # [B, 1+N]
        labels = torch.zeros(anchor_emb.size(0), dtype=torch.long, device=anchor_emb.device)
        
        return F.cross_entropy(logits, labels)
    else:
        # If no negatives, just maximize positive similarity
        return -torch.mean(pos_sim)

def batch_all_triplet_loss(anchor_emb, positive_emb, margin=0.2):
    """
    Batch-all triplet loss using within-batch negatives
    More stable than InfoNCE for similar graphs
    """
    batch_size = anchor_emb.size(0)
    
    # Compute positive similarities
    pos_sim = F.cosine_similarity(anchor_emb, positive_emb, dim=1)  # [B]
    
    # Use other anchors as negatives (batch-all approach)
    # Each anchor compared to all other positives in batch
    anchor_expanded = anchor_emb.unsqueeze(1)  # [B, 1, D]
    positive_expanded = positive_emb.unsqueeze(0)  # [1, B, D]
    all_sim = F.cosine_similarity(anchor_expanded, positive_expanded, dim=2)  # [B, B]
    
    # Create mask to exclude positive pairs (diagonal)
    mask = torch.eye(batch_size, device=anchor_emb.device).bool()
    all_sim_masked = all_sim.masked_fill(mask, float('-inf'))
    
    # Find hardest negative (highest similarity to wrong positive)
    hard_neg_sim = torch.max(all_sim_masked, dim=1)[0]  # [B]
    
    # Triplet loss: pos_sim should be higher than hard_neg_sim by margin
    loss = F.relu(hard_neg_sim - pos_sim + margin)
    
    return loss.mean()

def supervised_contrastive_loss(anchor_emb, positive_emb, temperature=0.1):
    """
    Supervised contrastive loss - normalize embeddings and maximize positive similarity
    """
    # L2 normalize embeddings
    anchor_norm = F.normalize(anchor_emb, p=2, dim=1)
    positive_norm = F.normalize(positive_emb, p=2, dim=1)
    
    # Compute similarities
    pos_sim = torch.sum(anchor_norm * positive_norm, dim=1) / temperature
    
    # Create within-batch negatives 
    batch_size = anchor_emb.size(0)
    
    # Similarity matrix: anchor[i] vs all positives[j]
    sim_matrix = torch.matmul(anchor_norm, positive_norm.T) / temperature  # [B, B]
    
    # Create labels (diagonal elements are positive pairs)
    labels = torch.arange(batch_size, device=anchor_emb.device)
    
    # Cross entropy treats this as classification: which positive belongs to this anchor?
    return F.cross_entropy(sim_matrix, labels)

# OPTIMIZED TRAINING WITH EARLY STOPPING

def train_siamese_model(model, train_loader, val_loader, params, device):
    """Train Siamese network with InfoNCE loss and early stopping"""
    optimizer = torch.optim.Adam(model.parameters(), lr=params['learning_rates'])
    max_epochs = params['epochs']
    patience = params['patience']
    temperature = params['temperature']
    
    print(f"🚂 Training Siamese network: {max_epochs} epochs, patience={patience}")
    print(f"   Training batches: {len(train_loader)}")
    print(f"   Validation batches: {len(val_loader)}")
    
    best_val_score = -float('inf')
    best_train_score = -float('inf')
    best_val_state = None
    best_train_state = None
    patience_counter = 0
    
    start_time = time.time()
    
    for epoch in range(max_epochs):
        epoch_start = time.time()
        
        # Training phase
        model.train()
        total_loss = 0
        batch_count = 0
        
        for batch_idx, batch in enumerate(train_loader):
            # Move to device
            anchor_nodes = batch['anchor_nodes'].to(device)
            anchor_adj = batch['anchor_adj'].to(device)
            anchor_features = batch['anchor_features'].to(device)
            positive_nodes = batch['positive_nodes'].to(device)
            positive_adj = batch['positive_adj'].to(device)
            positive_features = batch['positive_features'].to(device)
            
            # Encode anchor and positive
            anchor_emb = model(anchor_nodes, anchor_adj, anchor_features)
            positive_emb = model(positive_nodes, positive_adj, positive_features)
            
            # Encode negatives if present
            negative_emb = None
            if 'negative_nodes' in batch:
                negative_nodes = batch['negative_nodes'].to(device)
                negative_adj = batch['negative_adj'].to(device)
                negative_features = batch['negative_features'].to(device)
                negative_emb = model(negative_nodes, negative_adj, negative_features)
            
            # Adaptive loss function
            optimizer.zero_grad()
            
            loss_type = params.get('loss_type', 'infonce')
            if loss_type == 'supervised_contrastive':
                loss = supervised_contrastive_loss(anchor_emb, positive_emb, temperature)
            elif loss_type == 'batch_triplet':
                loss = batch_all_triplet_loss(anchor_emb, positive_emb, margin=0.2)
            else:  # infonce (default)
                loss = infonce_loss(anchor_emb, positive_emb, negative_emb, temperature)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            batch_count += 1
            
            # Progress update
            if batch_idx % 50 == 0:
                neg_count = negative_emb.size(0) if negative_emb is not None else 0
                pos_sim = F.cosine_similarity(anchor_emb, positive_emb, dim=1).mean().item()
                print(f"    Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}, Pos_Sim: {pos_sim:.3f} (negatives: {neg_count})")
        
        avg_train_loss = total_loss / batch_count if batch_count > 0 else 0
        
        # Validation phase (compute similarity accuracy)
        model.eval()
        val_similarities = []
        train_similarities = []
        
        with torch.no_grad():
            # Validation similarities
            for batch in val_loader:
                anchor_nodes = batch['anchor_nodes'].to(device)
                anchor_adj = batch['anchor_adj'].to(device)
                anchor_features = batch['anchor_features'].to(device)
                positive_nodes = batch['positive_nodes'].to(device)
                positive_adj = batch['positive_adj'].to(device)
                positive_features = batch['positive_features'].to(device)
                
                anchor_emb = model(anchor_nodes, anchor_adj, anchor_features)
                positive_emb = model(positive_nodes, positive_adj, positive_features)
                
                similarities = F.cosine_similarity(anchor_emb, positive_emb, dim=1)
                val_similarities.extend(similarities.cpu().numpy())
            
            # Training similarities (sample)
            train_sample = next(iter(train_loader))
            anchor_nodes = train_sample['anchor_nodes'].to(device)
            anchor_adj = train_sample['anchor_adj'].to(device)
            anchor_features = train_sample['anchor_features'].to(device)
            positive_nodes = train_sample['positive_nodes'].to(device)
            positive_adj = train_sample['positive_adj'].to(device)
            positive_features = train_sample['positive_features'].to(device)
            
            anchor_emb = model(anchor_nodes, anchor_adj, anchor_features)
            positive_emb = model(positive_nodes, positive_adj, positive_features)
            similarities = F.cosine_similarity(anchor_emb, positive_emb, dim=1)
            train_similarities.extend(similarities.cpu().numpy())
        
        val_score = np.mean(val_similarities) if val_similarities else 0
        train_score = np.mean(train_similarities) if train_similarities else 0
        
        epoch_time = time.time() - epoch_start
        
        print(f"Epoch {epoch+1}/{max_epochs} ({epoch_time:.1f}s)")
        print(f"  Train Loss: {avg_train_loss:.4f}, Train Sim: {train_score:.4f} (range: {np.min(train_similarities):.3f}-{np.max(train_similarities):.3f})")
        print(f"  Val Sim: {val_score:.4f} (range: {np.min(val_similarities):.3f}-{np.max(val_similarities):.3f}, n={len(val_similarities)})")
        
        # Early stopping and checkpoint saving
        if val_score > best_val_score:
            best_val_score = val_score
            best_val_state = model.state_dict().copy()
            patience_counter = 0
            print(f"  ✅ New best validation score: {best_val_score:.4f}")
        else:
            patience_counter += 1
        
        if train_score > best_train_score:
            best_train_score = train_score
            best_train_state = model.state_dict().copy()
            print(f"  🧠 New best training score: {best_train_score:.4f}")
        
        if patience_counter >= patience:
            print(f"  🛑 Early stopping after {patience} epochs without improvement")
            break
    
    total_time = time.time() - start_time
    print(f"🏁 Training completed in {total_time:.1f}s")
    print(f"   Best validation score: {best_val_score:.4f}")
    print(f"   Best training score: {best_train_score:.4f}")
    
    # Restore best validation model
    if best_val_state:
        model.load_state_dict(best_val_state)
    
    return model, best_val_score, best_train_score, best_val_state, best_train_state

# EVALUATION WITH EMBEDDING-BASED RETRIEVAL

def evaluate_siamese_retrieval(model, test_dataset, train_dataset, device, top_k=[5, 10]):
    """Evaluate Siamese network using embedding-based retrieval"""
    print(f"🧪 Evaluating with embedding-based retrieval...")
    print(f"   Test queries: {len(test_dataset)}")
    print(f"   Gallery size: {len(train_dataset)}")
    
    model.eval()
    
    # Pre-compute gallery embeddings (all training originals)
    print("   Computing gallery embeddings...")
    gallery_embeddings = []
    gallery_ids = []
    gallery_types = []
    
    gallery_loader = DataLoader(train_dataset, batch_size=32, shuffle=False, 
                               collate_fn=siamese_collate_fn, num_workers=0)
    
    with torch.no_grad():
        for batch in gallery_loader:
            # Use positive (original) graphs as gallery
            pos_nodes = batch['positive_nodes'].to(device)
            pos_adj = batch['positive_adj'].to(device)
            pos_features = batch['positive_features'].to(device)
            
            embeddings = model(pos_nodes, pos_adj, pos_features)
            gallery_embeddings.append(embeddings.cpu())
            gallery_ids.extend(batch['gallery_ids'])
            gallery_types.extend(batch['graph_types'])
    
    gallery_embeddings = torch.cat(gallery_embeddings, dim=0)  # [gallery_size, embedding_dim]
    print(f"   Gallery embeddings shape: {gallery_embeddings.shape}")
    
    # Evaluate test queries
    print("   Evaluating test queries...")
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False,
                            collate_fn=siamese_collate_fn, num_workers=0)
    
    all_recalls = {k: [] for k in top_k}
    all_mrrs = []
    type_results = defaultdict(lambda: {k: [] for k in top_k + ['mrr']})
    
    with torch.no_grad():
        for batch in test_loader:
            # Use anchor (enigma_2) graphs as queries
            anchor_nodes = batch['anchor_nodes'].to(device)
            anchor_adj = batch['anchor_adj'].to(device) 
            anchor_features = batch['anchor_features'].to(device)
            
            query_embeddings = model(anchor_nodes, anchor_adj, anchor_features)
            
            # Compute similarities with entire gallery
            similarities = torch.mm(query_embeddings.cpu(), gallery_embeddings.t())  # [batch_size, gallery_size]
            
            # Rank gallery items for each query
            for i, (query_sim, true_id, graph_type) in enumerate(zip(
                similarities, batch['gallery_ids'], batch['graph_types']
            )):
                # Get top-K most similar gallery items
                _, top_indices = torch.topk(query_sim, k=max(top_k), largest=True)
                top_gallery_ids = [gallery_ids[idx] for idx in top_indices]
                
                # Compute metrics
                true_rank = None
                for rank, pred_id in enumerate(top_gallery_ids):
                    if pred_id == true_id:
                        true_rank = rank + 1
                        break
                
                # Recall@K
                for k in top_k:
                    recall = 1.0 if true_rank and true_rank <= k else 0.0
                    all_recalls[k].append(recall)
                    type_results[graph_type][k].append(recall)
                
                # MRR
                mrr = 1.0 / true_rank if true_rank else 0.0
                all_mrrs.append(mrr)
                type_results[graph_type]['mrr'].append(mrr)
    
    # Compute overall results
    results = {}
    for k in top_k:
        results[f'recall_{k}'] = np.mean(all_recalls[k])
    results['mrr'] = np.mean(all_mrrs)
    
    # Compute per-type results
    type_summary = {}
    for graph_type in type_results:
        type_summary[graph_type] = {}
        for k in top_k:
            if type_results[graph_type][k]:
                type_summary[graph_type][f'recall_{k}'] = np.mean(type_results[graph_type][k])
            else:
                type_summary[graph_type][f'recall_{k}'] = 0.0
        if type_results[graph_type]['mrr']:
            type_summary[graph_type]['mrr'] = np.mean(type_results[graph_type]['mrr'])
        else:
            type_summary[graph_type]['mrr'] = 0.0
    
    # Print results
    print("\n📊 OVERALL RESULTS:")
    for k in top_k:
        print(f"   Recall@{k}: {results[f'recall_{k}']:.3f}")
    print(f"   MRR: {results['mrr']:.3f}")
    
    print("\n📊 PER-TYPE RESULTS:")
    for graph_type, metrics in type_summary.items():
        recall_str = ", ".join([f"R@{k}={metrics[f'recall_{k}']:.3f}" for k in top_k])
        print(f"   {graph_type}: {recall_str}, MRR={metrics['mrr']:.3f}")
    
    return results, type_summary

# MAIN EXECUTION

def load_prepared_dataset(config):
    """Load prepared dataset with error handling"""
    print("📁 Loading prepared dataset...")
    base_path = config["base_path"]
    files_config = config["files"]
    
    datasets = {}
    for key, filename in files_config.items():
        filepath = os.path.join(base_path, filename)
        if os.path.exists(filepath):
            with open(filepath, "rb") as f:
                data = pickle.load(f)
                datasets[key] = data
            print(f"   ✅ Loaded {key}: {len(datasets[key])} items")
        else:
            datasets[key] = []
            print(f"   ⚠️ Missing {key} ({filename})")
    
    return datasets

def main():
    """Main function with complete Siamese network pipeline"""
    parser = argparse.ArgumentParser(description="Exhaustive with Siamese Networks")
    parser.add_argument("--model", choices=["siamese_gin", "siamese_simgnn", "siamese_nre"], 
                       default="siamese_gin")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--max_cache", type=int, default=None, help="Max graphs to cache")
    
    args = parser.parse_args()
    
    print(" ANALYSIS- SIAMESE NETWORKS")
    
    print(" MODERN APPROACH: Embedding-based retrieval with InfoNCE loss")
    print(" OPTIMIZATIONS: Pre-cached RAM, proper batching, max threading")
    
    
    device = torch.device(args.device)
    print(f"🔧 Using device: {device}")
    print(f"🧵 PyTorch threads: {torch.get_num_threads()}")
    
    # Load data
    datasets = load_prepared_dataset(SERVER_DATA_CONFIGS["prepared_dataset"])
    if not datasets.get("opt2_train") or not datasets.get("opt2_test"):
        print("❌ Required datasets not found")
        return 1
    
    train_records = datasets["opt2_train"]
    test_records = datasets["opt2_test"]
    
    print(f"\n📊 Dataset summary:")
    print(f"   Training: {len(train_records)} pairs")
    print(f"   Testing: {len(test_records)} pairs")
    
    # Create datasets with caching
    print(f"\nCreating optimized datasets with caching...")
    
    # Use 90% for training, 10% for validation during hyperparameter search
    train_split, val_split = train_test_split(train_records, test_size=0.1, random_state=42)
    
    train_dataset = SiameseGraphDataset(train_split, mode='train', 
                                       negative_ratio=1.0, max_cache_size=args.max_cache)
    val_dataset = SiameseGraphDataset(val_split, mode='val', 
                                     negative_ratio=0.5, max_cache_size=args.max_cache)
    test_dataset = SiameseGraphDataset(test_records, mode='test', max_cache_size=args.max_cache)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                             collate_fn=siamese_collate_fn, num_workers=args.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                           collate_fn=siamese_collate_fn, num_workers=args.num_workers)
    
    print(f"✅ DataLoaders created with {args.num_workers} workers")
    
    # Create model
    model_params = HYPERPARAMETER_GRIDS[args.model]
    
    if args.model == "siamese_gin":
        model = SiameseGraphNetwork(
            node_dim=10,
            hidden_dim=model_params["hidden_dims"][0],
            num_layers=model_params["num_layers"][0],
            dropout=model_params["dropout_rates"][0],
            architecture='gin',
            use_features=True
        )
    elif args.model == "siamese_simgnn":
        model = SiameseGraphNetwork(
            node_dim=10,
            hidden_dim=model_params["hidden_dims"][0],
            num_layers=model_params["num_layers"][0],
            dropout=model_params["dropout_rates"][0],
            architecture='gcn',  # SimGNN uses GCN-style layers
            use_features=True
        )
    else:  # siamese_nre
        model = SiameseGraphNetwork(
            node_dim=10,
            hidden_dim=model_params["hidden_dims"][0],
            num_layers=model_params["num_layers"][0],
            dropout=model_params["dropout_rates"][0],
            architecture='gcn',
            use_features=True
        )
    
    model = model.to(device)
    print(f"🤖 Created {args.model} with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Training parameters
    training_params = {
        'learning_rates': model_params["learning_rates"][0],
        'epochs': model_params["epochs"][0],
        'patience': model_params["patience"][0],
        'temperature': model_params["temperature"][0],
        'loss_type': model_params["loss_type"][0]
    }
    
    print(f"\n🏎️ STARTING SIAMESE TRAINING...")
    print(f"   Architecture: {args.model}")
    print(f"   Learning rate: {training_params['learning_rates']}")
    print(f"   Temperature: {training_params['temperature']}")
    print(f"   Loss type: {training_params['loss_type']}")
    
    # Train model
    trained_model, best_val_score, best_train_score, val_state, train_state = train_siamese_model(
        model, train_loader, val_loader, training_params, device
    )
    
    print(f"\n🎉 TRAINING COMPLETED!")
    print(f"   Best validation score: {best_val_score:.4f}")
    print(f"   Best training score: {best_train_score:.4f}")
    
    # Comprehensive evaluation
    print(f"\n🧪 COMPREHENSIVE EVALUATION...")
    
    # Evaluate validation-best model
    results_val, type_results_val = evaluate_siamese_retrieval(
        trained_model, test_dataset, train_dataset, device, top_k=[5, 10]
    )
    
    # Evaluate training-best model
    if train_state:
        trained_model.load_state_dict(train_state)
        print(f"\n🧠 EVALUATING TRAINING-BEST MODEL...")
        results_train, type_results_train = evaluate_siamese_retrieval(
            trained_model, test_dataset, train_dataset, device, top_k=[5, 10]
        )
    
    # Save model
    checkpoint_path = f"{args.model}_siamese_best.pt"
    torch.save(val_state if val_state else trained_model.state_dict(), checkpoint_path)
    print(f"💾 Model saved to {checkpoint_path}")
    
    print(f"\n Analysis complete")
    print(f" Modern Siamese network with InfoNCE successfully trained and evaluated")
    
    return 0

if __name__ == "__main__":
    # Required for multiprocessing
    mp.set_start_method('spawn', force=True)
    exit(main())