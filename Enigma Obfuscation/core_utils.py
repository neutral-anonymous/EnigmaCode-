# -*- coding: utf-8 -*-
"""
Core Utilities for Ising/QUBO Models
Contains helper functions for model generation, conversion, and graph operations.
"""

import numpy as np
import networkx as nx
import pickle
import collections


# ============================================================================
# File I/O
# ============================================================================

def save_pickle(obj, filename):
    """Save object to pickle file."""
    with open(filename, 'wb') as pkl:
        pickle.dump(obj, pkl)


def load_pickle(filename):
    """Load object from pickle file."""
    with open(filename, 'rb') as pkl:
        return pickle.load(pkl)


# ============================================================================
# Ising Model Generation
# ============================================================================

def generate_random_ising(graph_type='er', n=10, graph_param=0.5, weight_mode='uniform'):
    """
    Generate a random IsingModel.
    
    Args:
        graph_type: 'er', 'ba', 'reg', or 'sk'
        graph_param: probability for 'er' (default 0.5), m for 'ba', degree for 'reg'
        weight_mode: 'uniform', 'normal', or 'binary'
    
    Returns:
        h: dict of node biases
        J: dict of edge couplings
    """
    if graph_type == 'er':
        G = nx.erdos_renyi_graph(n, graph_param)
    elif graph_type == 'ba':
        G = nx.barabasi_albert_graph(n, int(graph_param))
    elif graph_type == 'reg':
        G = nx.random_regular_graph(int(graph_param), n)
    elif graph_type == 'sk':
        G = nx.complete_graph(n)
    else:
        raise ValueError(f"Unsupported graph type: {graph_type}")

    def sample_weight():
        if weight_mode == 'uniform':
            return float(np.random.uniform(-1, 1))
        elif weight_mode == 'normal':
            return float(np.random.normal(0, 1))
        elif weight_mode == 'binary':
            return int(np.random.choice([-1, 1]))
        else:
            raise ValueError(f"Unknown weight_mode: {weight_mode}")

    h = {i: sample_weight() for i in G.nodes}
    J = {(i, j): sample_weight() for i, j in G.edges}

    return h, J


# ============================================================================
# Ising/QUBO Conversion
# ============================================================================

def ising_to_qubo(h, J, offset=0.0):
    """Convert Ising model to QUBO formulation."""
    Q = {}
    # off-diagonal
    for (i, j), Jij in J.items():
        Q[(i, j)] = 4.0 * Jij
    # diagonal
    for i, hi in h.items():
        sumJ = sum(J.get((min(i, k), max(i, k)), 0.0) for k in h if k != i)
        Q[(i, i)] = -2.0 * hi - 2.0 * sumJ
    # offset
    offset_x = offset + sum(h.values()) + sum(J.values())
    return Q, offset_x


def qubo_to_ising(Q, offset=0.0):
    """Convert QUBO formulation to Ising model."""
    # Recover J
    J = {}
    for (i, j), qij in Q.items():
        if i != j:
            J[(i, j)] = qij / 4.0
    # Recover h
    h = {}
    for i in set([i for i, _ in Q.keys()]):
        sumJ = sum(J.get((min(i, k), max(i, k)), 0.0) for k in set([n for n, _ in Q.keys()]) if k != i)
        Qii = Q.get((i, i), 0.0)
        h[i] = -(Qii + 2.0 * sumJ) / 2.0
    # Recover offset
    sum_h = sum(h.values())
    sum_J = sum(v for (i, j), v in J.items() if i < j)
    offset_z = offset - sum_h - sum_J
    return h, J, offset_z


# ============================================================================
# Graph Utilities
# ============================================================================

def ising_to_networkx(h=None, J=None, offset=0.0):
    """Convert Ising model to NetworkX graph."""
    if h is None:
        h = {}
    if J is None:
        J = {}

    G = nx.Graph()
    G.graph['offset'] = offset

    for i in set(h.keys()).union(*J.keys() if J else []):
        G.add_node(i, bias=h.get(i, 0.0))

    for (i, j), w in J.items():
        G.add_edge(i, j, weight=w)

    return G


def generate_random_graph(graph_type='er', n=10, graph_param=0.5):
    """Generate a random graph structure."""
    if graph_type == 'er':
        G = nx.erdos_renyi_graph(n, graph_param)
    elif graph_type == 'ba':
        G = nx.barabasi_albert_graph(n, int(graph_param))
    elif graph_type == 'reg':
        G = nx.random_regular_graph(int(graph_param), n)
    elif graph_type == 'sk':
        G = nx.complete_graph(n)
    else:
        raise ValueError(f"Unsupported graph type: {graph_type}")

    return G


def classify_graph(G):
    """Classify graph type based on structure."""
    n = G.number_of_nodes()
    m = G.number_of_edges()
    degrees = np.array([d for _, d in G.degree()])

    # Fully connected
    if m == n * (n - 1) // 2:
        return 'sk'

    # Regular
    if np.std(degrees) < 1e-6:
        return 'reg'

    # BA vs ER based on max degree / mean degree
    mean_deg = np.mean(degrees)
    max_deg = np.max(degrees)
    if max_deg > 2 * mean_deg:
        return 'ba'
    else:
        return 'er'
