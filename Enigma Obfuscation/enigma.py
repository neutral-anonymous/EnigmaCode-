# -*- coding: utf-8 -*-
"""
Enigma: Main Obfuscation Pipeline
Orchestrates the complete obfuscation workflow for Ising models.
"""

import copy
from core_utils import ising_to_networkx, classify_graph
from obfuscation_techniques import (
    topology_trimmer,
    structure_camouflage,
    regularizer,
    value_guard,
    ising_sanitizer
)


def enigma(
    h=None,
    J=None,
    offset=0.0,
    regularize_output_graph=False
):
    """
    Apply the complete Enigma obfuscation pipeline to an Ising model.
    
    Args:
        h: dict of node biases
        J: dict of edge couplings
        offset: energy offset
        regularize_output_graph: whether to apply regularizer step
    
    Returns:
        ising_list: list of obfuscated Ising models
        info: list of transformation metadata
    """
    if h is None:
        h = {}
    if J is None:
        J = {}

    G = ising_to_networkx(h, J)
    graph_type = classify_graph(G)

    ising_list = [{'h': copy.deepcopy(h), 'J': copy.deepcopy(J), 'offset': offset}]
    info = []

    # Default parameters
    max_trim = 10
    num_decoy_nodes = None
    min_num_decoy_node = 0.1
    max_num_decoy_node = 0.25
    edge_prob = 0.5
    preferential_attachment_factor = 0.25
    weight_strategy = 'inverse_roulette_wheel'
    edge_strategy = 'random'

    # Graph-type specific parameters
    if graph_type.lower() in ['sk']:
        trim_strategy = ['hotspots', 'preferential', 'random', 'hybrid'][2]
        min_num_decoy_node = 0.01
        max_num_decoy_node = 0.1
        edge_strategy = 'full'

    elif graph_type.lower() in ['reg']:
        trim_strategy = ['hotspots', 'preferential', 'random', 'hybrid'][2]
        min_num_decoy_node = 0.01
        max_num_decoy_node = 0.05
        edge_strategy = ['random', 'prefer_low_degree', 'prefer_high_degree'][1]
        preferential_attachment_factor = 0.01

    elif graph_type.lower() in ['ba']:
        trim_strategy = ['hotspots', 'preferential', 'random', 'hybrid'][3]
        min_num_decoy_node = 0.05
        max_num_decoy_node = 0.15
        edge_strategy = ['random', 'prefer_low_degree', 'prefer_high_degree'][1]
        preferential_attachment_factor = 0.025

    elif graph_type.lower() in ['er']:
        trim_strategy = ['hotspots', 'preferential', 'random', 'hybrid'][3]
        min_num_decoy_node = 0.1
        max_num_decoy_node = 0.25
        edge_strategy = ['random', 'prefer_low_degree', 'prefer_high_degree'][2]
        preferential_attachment_factor = 0.05

    else:
        raise ValueError(f'graph_type was detected as {graph_type}!')

    # Obfuscation pipeline
    
    # 1) Topology Trimmer
    ising_list, info = topology_trimmer(
        ising_list=ising_list,
        trim_strategy=trim_strategy,
        max_trim=max_trim,
        info=info
    )

    # 2) Structure Camouflage
    ising_list, info = structure_camouflage(
        ising_list=ising_list,
        num_decoy_nodes=num_decoy_nodes,
        min_num_decoy_node=min_num_decoy_node,
        max_num_decoy_node=max_num_decoy_node,
        edge_strategy=edge_strategy,
        edge_prob=edge_prob,
        preferential_attachment_factor=preferential_attachment_factor,
        weight_strategy=weight_strategy,
        info=info
    )

    # 3) Regularizer (optional)
    if regularize_output_graph:
        ising_list, info = regularizer(
            ising_list=ising_list,
            weight_strategy=weight_strategy,
            info=info
        )

    # 4) Value Guard
    ising_list, info = value_guard(
        ising_list=ising_list,
        info=info
    )

    # 5) Ising Sanitizer
    ising_list, info = ising_sanitizer(
        ising_list=ising_list,
        info=info
    )

    return ising_list, info
