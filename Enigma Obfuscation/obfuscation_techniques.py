# -*- coding: utf-8 -*-
"""
Obfuscation Techniques for Ising Models
Contains all obfuscation methods: value_guard, structure_camouflage, regularizer, topology_trimmer, and ising_sanitizer.
"""

import numpy as np
import random
import copy
import inspect
import collections
from core_utils import ising_to_qubo, qubo_to_ising


# ============================================================================
# Helper Functions
# ============================================================================

def generate_random_coeffs(input_list, w_count, dist_type='inverse_roulette_wheel', num_bins=10):
    """
    Generate w_count random positive real numbers based on the specified distribution type.
    
    Args:
        input_list: list of real numbers
        w_count: integer, number of random numbers to generate
        dist_type: 'uniform', 'normal', 'roulette_wheel', 'inverse_roulette_wheel'
        num_bins: number of bins for roulette wheel distributions
    
    Returns:
        list of generated random numbers as Python floats
    """
    if not input_list or w_count <= 0:
        raise ValueError("input_list must be a non-empty list and w_count must be a positive integer.")

    abs_input = [abs(c) for c in input_list]
    generated_coeffs = []

    if dist_type == 'uniform':
        min_val = min(abs_input)
        max_val = max(abs_input)
        if min_val == max_val:
            raise ValueError("All values are identical; cannot define a uniform distribution.")
        generated_coeffs = np.random.uniform(low=min_val, high=max_val, size=w_count)

    elif dist_type == 'normal':
        mean_val = np.mean(abs_input)
        std_val = np.std(abs_input)
        if std_val == 0:
            raise ValueError("Standard deviation is zero; cannot define a normal distribution.")
        while len(generated_coeffs) < w_count:
            sample = np.random.normal(loc=mean_val, scale=std_val)
            if sample > 0:
                generated_coeffs.append(sample)
        generated_coeffs = np.array(generated_coeffs)

    elif dist_type == 'roulette_wheel':
        hist, bin_edges = np.histogram(abs_input, bins=num_bins)
        bin_probabilities = hist / hist.sum()

        for _ in range(w_count):
            selected_bin = random.choices(range(num_bins), weights=bin_probabilities, k=1)[0]
            bin_min = bin_edges[selected_bin]
            bin_max = bin_edges[selected_bin + 1]
            generated_coeffs.append(np.random.uniform(bin_min, bin_max))
        generated_coeffs = np.array(generated_coeffs)

    elif dist_type == 'inverse_roulette_wheel':
        hist, bin_edges = np.histogram(abs_input, bins=num_bins)

        with np.errstate(divide='ignore', invalid='ignore'):
            inverse_hist = np.where(hist == 0, 0, 1.0 / hist)
        total_inverse = np.sum(inverse_hist)
        if total_inverse == 0:
            raise ValueError("All bins are empty; cannot perform inverse roulette wheel selection.")
        bin_probabilities = inverse_hist / total_inverse

        for _ in range(w_count):
            selected_bin = random.choices(range(num_bins), weights=bin_probabilities, k=1)[0]
            bin_min = bin_edges[selected_bin]
            bin_max = bin_edges[selected_bin + 1]
            generated_coeffs.append(np.random.uniform(bin_min, bin_max))
        generated_coeffs = np.array(generated_coeffs)

    else:
        raise ValueError("Invalid dist_type. Choose from 'uniform', 'normal', 'roulette_wheel', 'inverse_roulette_wheel'.")

    return [float(x) for x in generated_coeffs]


# ============================================================================
# Obfuscation Techniques
# ============================================================================

def value_guard(ising_list, target_qubits=None, tau=None, info=None):
    """Apply value guard obfuscation by rescaling and flipping selected qubits."""
    if not ising_list:
        raise ValueError("ising_list cannot be empty!")

    if tau is None:
        tau = 10 ** np.random.uniform(-2, 2)

    J = ising_list[0].get('J', {})

    if target_qubits is None:
        all_qubits = {q for pair in J for q in pair}
        target_qubits = {q for q in all_qubits if random.random() < 0.5}
    else:
        target_qubits = set(target_qubits)

    res_ising_list = []

    for ising_obj in ising_list:
        h = ising_obj.get('h', {})
        J = ising_obj.get('J', {})
        offset = ising_obj.get('offset', 0.0)

        _h = {
            i: tau * (-h[i] if i in target_qubits else h[i])
            for i in h
        }

        _J = {}
        for (i, j), val in J.items():
            sign = -1 if (i in target_qubits) ^ (j in target_qubits) else 1
            _J[(i, j)] = tau * val * sign

        res_ising_list.append({
            'h': _h,
            'J': _J,
            'offset': tau * offset
        })

    info = info if info is not None else []
    info.append({
        'ob_name': inspect.currentframe().f_code.co_name,
        'tau': tau,
        'target_qubits': sorted(target_qubits)
    })

    return res_ising_list, info


def structure_camouflage(
    ising_list,
    num_decoy_nodes=None,
    min_num_decoy_node=0.1,
    max_num_decoy_node=0.25,
    edge_strategy='prefer_low_degree',
    edge_prob=0.5,
    preferential_attachment_factor=0.4,
    weight_strategy='inverse_roulette_wheel',
    info=None
):
    """Add decoy nodes and edges to camouflage the graph structure."""
    if not ising_list:
        raise ValueError("ising_list cannot be empty.")

    h0 = ising_list[0].get('h', {})
    J0 = ising_list[0].get('J', {})
    real_nodes = set(h0.keys())
    for i, j in J0:
        real_nodes.update([i, j])
    real_nodes = list(real_nodes)
    num_vars = len(real_nodes)
    max_node = max(real_nodes)

    if num_decoy_nodes is None:
        num_decoy_nodes = int(np.random.uniform(min_num_decoy_node, max_num_decoy_node) * num_vars)

    decoy_nodes = list(range(max_node + 1, max_node + 1 + num_decoy_nodes))

    result_list = []
    for ising in ising_list:
        h = copy.deepcopy(ising.get('h', {}))
        J = copy.deepcopy(ising.get('J', {}))
        offset = ising.get('offset', 0.0)

        Q, qubo_offset = ising_to_qubo(h, J, offset)

        degrees = {}
        for (i, j) in Q:
            if i != j:
                degrees[i] = degrees.get(i, 0) + 1
                degrees[j] = degrees.get(j, 0) + 1

        new_edges = []
        edge_set = {(i, j) for (i, j) in Q if i != j}

        for u in decoy_nodes:
            candidates = [v for v in real_nodes + decoy_nodes if v != u and (min(u, v), max(u, v)) not in edge_set]

            for v in candidates:
                d = degrees.get(v, 0)
                if edge_strategy == 'random':
                    p = edge_prob
                elif edge_strategy == 'prefer_low_degree':
                    p = 1 / (1 + d)
                    p *= preferential_attachment_factor
                elif edge_strategy == 'prefer_high_degree':
                    p = d / (d + 1)
                    p *= preferential_attachment_factor
                elif edge_strategy == 'full':
                    p = 10**20
                else:
                    raise ValueError(f"Unknown edge strategy: {edge_strategy}")

                if random.random() < p:
                    i, j = min(u, v), max(u, v)
                    new_edges.append((i, j))
                    degrees[i] = degrees.get(i, 0) + 1
                    degrees[j] = degrees.get(j, 0) + 1
                    edge_set.add((i, j))

        # Enforce connectivity constraints
        neighbor_map = {u: set() for u in decoy_nodes}
        for i, j in edge_set:
            if i in decoy_nodes:
                neighbor_map[i].add(j)
            if j in decoy_nodes:
                neighbor_map[j].add(i)

        for u in decoy_nodes:
            neighbors = neighbor_map[u]
            if not any(v in real_nodes for v in neighbors):
                v = random.choice(real_nodes)
                i, j = min(u, v), max(u, v)
                new_edges.append((i, j))
                edge_set.add((i, j))
                neighbor_map[u].add(v)
            if not any(v in decoy_nodes for v in neighbors if v != u) and len(decoy_nodes) > 1:
                v = random.choice([n for n in decoy_nodes if n != u])
                i, j = min(u, v), max(u, v)
                new_edges.append((i, j))
                edge_set.add((i, j))
                neighbor_map[u].add(v)

        # Assign weights
        existing_vals = list(Q.values())
        if len(new_edges) > 0:
            weights = generate_random_coeffs(existing_vals, len(new_edges), dist_type=weight_strategy)
            for (i, j), w in zip(new_edges, weights):
                Q[(i, j)] = w

        linear_vals = [Q.get((i, i), 0.0) for i in real_nodes]
        non_zero_count = sum(1 for val in linear_vals if abs(val) > 1e-8)
        if non_zero_count > 0.5 * len(real_nodes):
            if len(decoy_nodes) > 0:
                linear_weights = generate_random_coeffs(linear_vals, len(decoy_nodes), dist_type=weight_strategy)
                for i, w in zip(decoy_nodes, linear_weights):
                    Q[(i, i)] = w

        obf_h, obf_J, obf_offset = qubo_to_ising(Q, offset=qubo_offset)
        result_list.append({'h': obf_h, 'J': obf_J, 'offset': obf_offset})

    if info is None:
        info = []
    info.append({
        'ob_name': inspect.currentframe().f_code.co_name,
        'num_decoy_nodes': num_decoy_nodes,
        'edge_strategy': edge_strategy,
        'weight_strategy': weight_strategy,
        'decoy_nodes': decoy_nodes
    })

    return result_list, info


def regularizer(
    ising_list,
    weight_strategy='inverse_roulette_wheel',
    info=None
):
    """Regularize the graph to make all nodes have the same degree."""
    if not ising_list:
        raise ValueError("ising_list cannot be empty.")

    result_list = []
    for ising in ising_list:
        h = copy.deepcopy(ising.get('h', {}))
        J = copy.deepcopy(ising.get('J', {}))
        offset = ising.get('offset', 0.0)

        Q, qubo_offset = ising_to_qubo(h, J, offset)

        nodes = sorted(set(i for ij in Q for i in ij))
        n = len(nodes)

        degrees = {i: 0 for i in nodes}
        for (i, j) in Q:
            if i != j:
                degrees[i] += 1
                degrees[j] += 1

        d_list = np.array([degrees[i] for i in nodes])
        d = int(np.max(d_list))
        deficiency_list = d - d_list
        max_def = int(np.max(deficiency_list))
        s = int(np.sum(deficiency_list))

        m = None
        for m_try in range(n + 1):
            if (
                m_try * d >= s and
                m_try**2 - (d + 1) * m_try + s >= 0 and
                m_try >= max_def and
                (n + m_try) * d % 2 == 0
            ):
                m = m_try
                break

        if m is None:
            raise ValueError("Could not find valid m to regularize the graph.")

        decoy_nodes = list(range(max(nodes) + 1, max(nodes) + 1 + m))

        x_list = sorted([(int(deficiency_list[i]), nodes[i]) for i in range(n) if deficiency_list[i] > 0], reverse=True)
        y_list = [[d, j] for j in decoy_nodes]
        random.shuffle(y_list)

        existing_vals = list(Q.values())
        new_weights = []

        for e_i, x_i in x_list:
            for _ in range(e_i):
                if not y_list:
                    break
                y_entry = y_list[0]
                _, y_j = y_entry
                Q[(min(x_i, y_j), max(x_i, y_j))] = 0.0
                new_weights.append((x_i, y_j))
                y_entry[0] -= 1
                if y_entry[0] == 0:
                    y_list.pop(0)
                else:
                    y_list = sorted(y_list, reverse=True)

        y_list = sorted([entry for entry in y_list if entry[0] > 0], reverse=True)
        while len(y_list) > 1:
            e1, y1 = y_list[0]
            for i in range(e1):
                if i + 1 >= len(y_list):
                    break
                e2, y2 = y_list[i + 1]
                Q[(min(y1, y2), max(y1, y2))] = 0.0
                new_weights.append((y1, y2))
                y_list[0][0] -= 1
                y_list[i + 1][0] -= 1
            y_list = sorted([entry for entry in y_list if entry[0] > 0], reverse=True)

        # Add linear terms
        linear_vals = [Q.get((i, i), 0.0) for i in nodes]
        filtered_vals = [val for val in linear_vals if abs(val) > 1e-8]
        if len(filtered_vals) >= 2 and len(decoy_nodes) > 0:
            lin_weights = generate_random_coeffs(filtered_vals, len(decoy_nodes), dist_type=weight_strategy)
            for i, w in zip(decoy_nodes, lin_weights):
                Q[(i, i)] = w

        if new_weights:
            edge_weights = generate_random_coeffs(existing_vals, len(new_weights), dist_type=weight_strategy)
            for (i, j), w in zip(new_weights, edge_weights):
                Q[(i, j)] = w

        obf_h, obf_J, obf_offset = qubo_to_ising(Q, offset=qubo_offset)
        result_list.append({'h': obf_h, 'J': obf_J, 'offset': obf_offset})

    if info is None:
        info = []
    info.append({
        'ob_name': inspect.currentframe().f_code.co_name,
        'target_degree': d,
        'num_decoy_nodes': m,
        'decoy_nodes': decoy_nodes,
        'weight_strategy': weight_strategy
    })

    return result_list, info


def topology_trimmer(
    ising_list,
    trim_strategy='preferential',
    max_trim=10,
    info=None
):
    """Trim variables from the Ising model by fixing them to specific values."""
    only_generate_one_sub_problem = True

    if not ising_list:
        raise ValueError("ising_list cannot be empty.")
    if len(ising_list) != 1:
        raise ValueError("topology_trimmer only accepts a single Ising instance.")
    if info is None:
        info = []

    h = copy.deepcopy(ising_list[0].get('h', {}))
    raw_J = ising_list[0].get('J', {})
    offset = ising_list[0].get('offset', 0.0)

    J = {}
    degrees = collections.defaultdict(int)
    for (i, j), val in raw_J.items():
        a, b = min(i, j), max(i, j)
        J[(a, b)] = J.get((a, b), 0.0) + val
        degrees[a] += 1
        degrees[b] += 1
    for u in h:
        _ = degrees[u]

    nodes = sorted(degrees)

    max_trim = min(max_trim, len(nodes))
    m = random.randint(0, max_trim)

    if m == 0:
        info.append({
            'ob_name': inspect.currentframe().f_code.co_name,
            'trim_strategy': trim_strategy,
            'num_trimmed_vars': 0,
            'trimmed_vars': [],
            'fixing_order': []
        })
        return ising_list, info

    if trim_strategy == 'hotspots':
        trimmed_vars = [v for v, _ in sorted(degrees.items(), key=lambda x: -x[1])[:m]]
    elif trim_strategy == 'preferential':
        weights = [degrees[v] + 1 for v in nodes]
        trimmed_vars = random.choices(nodes, weights=weights, k=m)
        trimmed_vars = list(dict.fromkeys(trimmed_vars))
        while len(trimmed_vars) < m:
            v = random.choices(nodes, weights=weights, k=1)[0]
            if v not in trimmed_vars:
                trimmed_vars.append(v)
    elif trim_strategy == 'random':
        trimmed_vars = random.sample(nodes, m)
    elif trim_strategy == 'hybrid':
        sorted_nodes = sorted(degrees.items(), key=lambda x: -x[1])
        top_vars = [sorted_nodes[0][0]]
        remaining_nodes = [v for v in nodes if v not in top_vars]
        rand_vars = random.sample(remaining_nodes, m - 1) if m > 1 else []
        trimmed_vars = top_vars + rand_vars
    else:
        raise ValueError(f"Unknown trim_strategy: {trim_strategy}")

    fixing_order = list(range(2 ** m))
    random.shuffle(fixing_order)

    new_ising_list = []
    for idx in fixing_order:
        bin_val = list(map(int, bin(idx)[2:].zfill(m)))
        h_new = {k: v for k, v in h.items() if k not in trimmed_vars}
        J_new = {}
        offset_new = offset

        for (i, j), val in J.items():
            if i in trimmed_vars and j in trimmed_vars:
                i_idx = trimmed_vars.index(i)
                j_idx = trimmed_vars.index(j)
                offset_new += val * bin_val[i_idx] * bin_val[j_idx]
            elif i in trimmed_vars:
                i_idx = trimmed_vars.index(i)
                j_val = bin_val[i_idx]
                h_new[j] = h_new.get(j, 0.0) + val * j_val
            elif j in trimmed_vars:
                j_idx = trimmed_vars.index(j)
                i_val = bin_val[j_idx]
                h_new[i] = h_new.get(i, 0.0) + val * i_val
            else:
                J_new[(i, j)] = val

        for i, val in h.items():
            if i in trimmed_vars:
                idx_i = trimmed_vars.index(i)
                offset_new += val * bin_val[idx_i]

        new_ising_list.append({'h': h_new, 'J': J_new, 'offset': offset_new})

        if only_generate_one_sub_problem:
            break

    info.append({
        'ob_name': inspect.currentframe().f_code.co_name,
        'trim_strategy': trim_strategy,
        'num_trimmed_vars': m,
        'trimmed_vars': trimmed_vars,
        'fixing_order': fixing_order
    })

    return new_ising_list, info


def ising_sanitizer(ising_list, info=None, threshold=None):
    """Clean and normalize the Ising model by removing near-zero coefficients and re-indexing."""
    if info is None:
        info = []

    results = []
    mappings = []
    removed_vars_list = []

    for ising in ising_list:
        h = copy.deepcopy(ising.get("h", {}))
        raw_J = copy.deepcopy(ising.get("J", {}))
        offset = ising.get("offset", 0.0)

        # Normalize J keys
        J = {}
        for (i, j), val in raw_J.items():
            a, b = min(i, j), max(i, j)
            J[(a, b)] = J.get((a, b), 0.0) + val

        # Adaptive threshold
        if threshold is None:
            coeffs = [abs(v) for v in h.values()] + [abs(v) for v in J.values()]
            nonzero_coeffs = [c for c in coeffs if c > 0]
            if nonzero_coeffs:
                threshold = np.median(nonzero_coeffs) * 1e-6
            else:
                threshold = 1e-8

        # Remove near-zero coefficients
        h = {i: v for i, v in h.items() if abs(v) >= threshold}
        J = {k: v for k, v in J.items() if abs(v) >= threshold}

        # Compute degrees
        degrees = collections.defaultdict(int)
        for (i, j) in J:
            degrees[i] += 1
            degrees[j] += 1
        for i in h:
            degrees[i] += 0

        # Remove degree-zero nodes
        zero_deg = [i for i, d in degrees.items() if d == 0]
        h = {i: v for i, v in h.items() if i not in zero_deg}

        # Re-index variables
        all_vars = sorted(set(h.keys()) | {i for ij in J for i in ij})
        var_mapping = {old: new for new, old in enumerate(all_vars)}

        h_new = {var_mapping[i]: v for i, v in h.items()}
        J_new = {(var_mapping[i], var_mapping[j]): v for (i, j), v in J.items()}

        results.append({"h": h_new, "J": J_new, "offset": offset})
        mappings.append(var_mapping)
        removed_vars_list.append(zero_deg)

    info.append({
        "ob_name": inspect.currentframe().f_code.co_name,
        "threshold": threshold,
        "var_mappings": mappings,
        "removed_zero_degree_vars": removed_vars_list,
    })

    return results, info
