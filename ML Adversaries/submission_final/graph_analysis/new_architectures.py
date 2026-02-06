#!/usr/bin/env python3
"""
Implementation for:
1. Direct Feature Matching (no training baseline)
2. Graph Attention Networks (structure-aware learning)
3. Metric Learning (proper hard negative mining)
4. SecGraph-style Graph Matching
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool, global_max_pool, global_add_pool
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
import networkx as nx
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import pdist, squareform
import time

# DIRECT FEATURE MATCHING

class DirectFeatureMatcher:
    """
    Direct structural feature matching - no training needed
    Based on classical de-anonymization approaches
    """
    
    def __init__(self, feature_dim=50):
        self.feature_dim = feature_dim
        self.scaler = StandardScaler()
        self.is_fitted = False
    
    def extract_advanced_structural_features(self, qubo_mapping):
        """Extract comprehensive structural features that might be preserved"""
        if not isinstance(qubo_mapping, dict) or len(qubo_mapping) == 0:
            return np.zeros(self.feature_dim, dtype=np.float32)
        
        features = []
        
        # Basic QUBO properties
        values = []
        diagonal_terms = []
        off_diagonal_terms = []
        
        for key, value in qubo_mapping.items():
            try:
                if isinstance(key, str):
                    if ',' in key:
                        i, j = map(int, key.split(','))
                    else:
                        i = j = int(key)
                elif isinstance(key, (tuple, list)):
                    i, j = int(key[0]), int(key[1])
                else:
                    i = j = int(key)
                
                val = float(value) if value is not None else 0.0
                values.append(val)
                
                if i == j:
                    diagonal_terms.append(val)
                else:
                    off_diagonal_terms.append(val)
            except:
                continue
        
        if not values:
            return np.zeros(self.feature_dim, dtype=np.float32)
        
        # Statistical features (preserved under many obfuscations)
        features.extend([
            len(values),                                          # Total terms
            len(diagonal_terms),                                  # Diagonal count
            len(off_diagonal_terms),                             # Off-diagonal count
            np.mean(values),                                     # Mean value
            np.std(values),                                      # Std value
            np.min(values),                                      # Min value
            np.max(values),                                      # Max value
            np.median(values),                                   # Median
            np.percentile(values, 25),                          # Q1
            np.percentile(values, 75),                          # Q3
        ])
        
        # Ratio features (often preserved)
        total_terms = len(values)
        features.extend([
            len(diagonal_terms) / total_terms if total_terms > 0 else 0,     # Diagonal ratio
            len([v for v in values if v > 0]) / total_terms if total_terms > 0 else 0,  # Positive ratio
            len([v for v in values if v < 0]) / total_terms if total_terms > 0 else 0,  # Negative ratio
            len(set(values)) / total_terms if total_terms > 0 else 0,        # Uniqueness ratio
        ])
        
        # Distribution features
        if len(values) > 1:
            features.extend([
                np.var(values),                                  # Variance
                np.sum(np.abs(values)),                         # L1 norm
                np.sqrt(np.sum(np.square(values))),             # L2 norm
                np.max(values) - np.min(values),                # Range
            ])
        else:
            features.extend([0, 0, 0, 0])
        
        # Structure-based features
        if diagonal_terms:
            features.extend([
                np.mean(diagonal_terms),
                np.std(diagonal_terms),
                np.sum(diagonal_terms),
            ])
        else:
            features.extend([0, 0, 0])
            
        if off_diagonal_terms:
            features.extend([
                np.mean(off_diagonal_terms),
                np.std(off_diagonal_terms),
                np.sum(off_diagonal_terms),
            ])
        else:
            features.extend([0, 0, 0])
        
        # Correlation and interaction features
        try:
            # Create adjacency-like structure
            max_idx = max([max(int(str(k).split(',')[0]) if ',' in str(k) else int(k),
                              int(str(k).split(',')[1]) if ',' in str(k) else int(k))
                          for k in qubo_mapping.keys()])
            
            # Graph-theoretic features
            features.extend([
                max_idx + 1,                                    # Graph size
                len(off_diagonal_terms) / ((max_idx + 1) * max_idx / 2) if max_idx > 0 else 0,  # Density
            ])
        except:
            features.extend([0, 0])
        
        # Advanced statistical features
        if len(values) > 2:
            # Moments
            features.extend([
                float(np.mean(np.power(values, 3))),           # Third moment
                float(np.mean(np.power(values, 4))),           # Fourth moment
            ])
            
            # Distribution shape
            from scipy import stats
            try:
                features.extend([
                    float(stats.skew(values)),                  # Skewness
                    float(stats.kurtosis(values)),              # Kurtosis
                ])
            except:
                features.extend([0, 0])
        else:
            features.extend([0, 0, 0, 0])
        
        # Pad to fixed size
        while len(features) < self.feature_dim:
            features.append(0.0)
        
        return np.array(features[:self.feature_dim], dtype=np.float32)
    
    def fit(self, qubo_mappings):
        """Fit the scaler on training QUBOs"""
        features = []
        for qubo in qubo_mappings:
            feat = self.extract_advanced_structural_features(qubo)
            features.append(feat)
        
        features = np.array(features)
        self.scaler.fit(features)
        self.is_fitted = True
    
    def compute_similarity(self, qubo1, qubo2):
        """Compute similarity between two QUBOs"""
        feat1 = self.extract_advanced_structural_features(qubo1)
        feat2 = self.extract_advanced_structural_features(qubo2)
        
        if self.is_fitted:
            feat1 = self.scaler.transform(feat1.reshape(1, -1)).flatten()
            feat2 = self.scaler.transform(feat2.reshape(1, -1)).flatten()
        
        # Multiple similarity metrics
        cosine_sim = cosine_similarity([feat1], [feat2])[0, 0]
        l2_sim = 1.0 / (1.0 + np.linalg.norm(feat1 - feat2))
        
        # Weighted combination
        return 0.7 * cosine_sim + 0.3 * l2_sim
    
    def evaluate_retrieval(self, query_qubos, gallery_qubos, gallery_ids, top_k=5):
        """Evaluate retrieval performance"""
        recalls = []
        
        for i, query_qubo in enumerate(query_qubos):
            similarities = []
            for gallery_qubo in gallery_qubos:
                sim = self.compute_similarity(query_qubo, gallery_qubo)
                similarities.append(sim)
            
            # Find top-k most similar
            similarities = np.array(similarities)
            top_k_indices = np.argsort(similarities)[::-1][:top_k]
            
            # Check if correct match is in top-k
            correct_id = gallery_ids[i]  # Assuming same order
            predicted_ids = [gallery_ids[idx] for idx in top_k_indices]
            
            recalls.append(1 if correct_id in predicted_ids else 0)
        
        return np.mean(recalls)

# GRAPH ATTENTION NETWORKS

class GraphAttentionMatcher(nn.Module):
    """
    Graph Attention Network for structure-aware graph matching
    Learns which structural features are preserved under obfuscation
    """
    
    def __init__(self, node_dim=10, hidden_dim=128, attention_heads=8, num_layers=3, dropout=0.2):
        super().__init__()
        
        self.num_layers = num_layers
        self.dropout = dropout
        
        # Multi-layer GAT
        self.gat_layers = nn.ModuleList()
        self.gat_layers.append(GATConv(node_dim, hidden_dim, heads=attention_heads, dropout=dropout))
        
        for _ in range(num_layers - 1):
            self.gat_layers.append(
                GATConv(hidden_dim * attention_heads, hidden_dim, 
                       heads=attention_heads, dropout=dropout)
            )
        
        # Attention pooling for graph-level representation
        self.attention_pool = nn.Sequential(
            nn.Linear(hidden_dim * attention_heads, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
        
        # Cross-graph attention (key innovation)
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim * attention_heads,
            num_heads=attention_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Final similarity computation
        self.similarity_mlp = nn.Sequential(
            nn.Linear(hidden_dim * attention_heads * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
        # Feature-level attention (for QUBO features)
        self.feature_attention = nn.Sequential(
            nn.Linear(70, hidden_dim),  # 70D QUBO features
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh()
        )
    
    def forward(self, anchor_nodes, anchor_adj, anchor_features, 
                positive_nodes, positive_adj, positive_features):
        
        # Process anchor graph
        anchor_emb = self.process_graph(anchor_nodes, anchor_adj, anchor_features)
        
        # Process positive graph
        positive_emb = self.process_graph(positive_nodes, positive_adj, positive_features)
        
        # Cross-graph attention to find correspondences
        anchor_attended, _ = self.cross_attention(
            anchor_emb.unsqueeze(1), positive_emb.unsqueeze(1), positive_emb.unsqueeze(1)
        )
        positive_attended, _ = self.cross_attention(
            positive_emb.unsqueeze(1), anchor_emb.unsqueeze(1), anchor_emb.unsqueeze(1)
        )
        
        anchor_final = anchor_attended.squeeze(1)
        positive_final = positive_attended.squeeze(1)
        
        # Compute similarity
        combined = torch.cat([anchor_final, positive_final], dim=1)
        similarity = self.similarity_mlp(combined)
        
        return similarity.squeeze(1)
    
    def process_graph(self, nodes, adj, features):
        """Process single graph through GAT layers"""
        batch_size = nodes.size(0)
        
        # Create edge indices from adjacency matrix
        edge_indices = []
        for b in range(batch_size):
            adj_b = adj[b]
            edge_index = torch.nonzero(adj_b, as_tuple=False).t()
            edge_indices.append(edge_index)
        
        # Process each graph in batch
        graph_embeddings = []
        for b in range(batch_size):
            node_feat = nodes[b]  # [num_nodes, node_dim]
            edge_index = edge_indices[b]  # [2, num_edges]
            
            # GAT layers
            x = node_feat
            for layer in self.gat_layers:
                x = F.dropout(x, p=self.dropout, training=self.training)
                x = layer(x, edge_index)
                x = F.elu(x)
            
            # Attention-based pooling
            attention_weights = self.attention_pool(x)  # [num_nodes, 1]
            attention_weights = F.softmax(attention_weights, dim=0)
            
            # Weighted sum of node embeddings
            graph_emb = torch.sum(attention_weights * x, dim=0)  # [hidden_dim * heads]
            
            # Add feature-level information
            feature_emb = self.feature_attention(features[b])
            graph_emb = graph_emb + feature_emb
            
            graph_embeddings.append(graph_emb)
        
        return torch.stack(graph_embeddings)

# METRIC LEARNING WITH HARD NEGATIVES

class QUBOMetricLearning(nn.Module):
    """
    Proper metric learning approach with hard negative mining
    Learns embedding space where similar graphs are close
    """
    
    def __init__(self, node_dim=10, hidden_dim=128, num_layers=3, dropout=0.2):
        super().__init__()
        
        # Graph encoder (using GIN for better representation)
        self.node_encoder = nn.Linear(node_dim, hidden_dim)
        
        self.gin_layers = nn.ModuleList()
        for _ in range(num_layers):
            mlp = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim * 2),
                nn.ReLU(),
                nn.Linear(hidden_dim * 2, hidden_dim)
            )
            self.gin_layers.append(mlp)
        
        # Feature encoder
        self.feature_encoder = nn.Sequential(
            nn.Linear(70, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Final projection layer (key for metric learning)
        self.projection = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim)
        )
        
        # Temperature parameter for contrastive learning
        self.temperature = nn.Parameter(torch.tensor(0.07))
    
    def forward(self, nodes, adj, features):
        """Encode graph to embedding space"""
        batch_size = nodes.size(0)
        
        # Process nodes
        x = self.node_encoder(nodes)  # [batch, nodes, hidden]
        
        # GIN layers with sum aggregation
        for gin_layer in self.gin_layers:
            # Sum aggregation (simple but effective)
            neighbor_sum = torch.bmm(adj, x)  # [batch, nodes, hidden]
            x_new = gin_layer(x + neighbor_sum)
            x = F.relu(x_new)
        
        # Global pooling
        graph_emb = torch.mean(x, dim=1)  # [batch, hidden]
        
        # Feature embedding
        feat_emb = self.feature_encoder(features)  # [batch, hidden]
        
        # Combine and project
        combined = torch.cat([graph_emb, feat_emb], dim=1)
        embedding = self.projection(combined)
        
        # L2 normalize (crucial for metric learning)
        embedding = F.normalize(embedding, p=2, dim=1)
        
        return embedding
    
    def compute_similarity(self, emb1, emb2):
        """Compute similarity in learned metric space"""
        return F.cosine_similarity(emb1, emb2, dim=1)

def hard_negative_triplet_loss(anchor_emb, positive_emb, negative_emb_pool, margin=0.2, temperature=0.07):
    """
    Proper hard negative mining for metric learning
    """
    batch_size = anchor_emb.size(0)
    
    # Positive similarities
    pos_sim = F.cosine_similarity(anchor_emb, positive_emb, dim=1)
    
    # Mine hard negatives from the pool
    hard_neg_sims = []
    for i in range(batch_size):
        anchor_i = anchor_emb[i:i+1]  # [1, dim]
        
        # Compute similarities to all negatives
        neg_sims = F.cosine_similarity(anchor_i, negative_emb_pool, dim=1)
        
        # Find hardest negative (most similar to anchor)
        hard_neg_sim = torch.max(neg_sims)
        hard_neg_sims.append(hard_neg_sim)
    
    hard_neg_sims = torch.stack(hard_neg_sims)
    
    # Triplet loss with temperature scaling
    pos_sim_scaled = pos_sim / temperature
    neg_sim_scaled = hard_neg_sims / temperature
    
    loss = F.relu(neg_sim_scaled - pos_sim_scaled + margin)
    
    return loss.mean()

# SECGRAPH-STYLE GRAPH MATCHING

class SecGraphMatcher:
    """
    SecGraph-style graph matching algorithm
    Based on structural similarity and graph isomorphism
    """
    
    def __init__(self, alpha=0.7, max_iterations=10):
        self.alpha = alpha  # Weight for structural vs feature similarity
        self.max_iterations = max_iterations
    
    def qubo_to_networkx(self, qubo_mapping):
        """Convert QUBO to NetworkX graph"""
        G = nx.Graph()
        
        # Add nodes and edges
        for key, value in qubo_mapping.items():
            try:
                if isinstance(key, str):
                    if ',' in key:
                        i, j = map(int, key.split(','))
                    else:
                        i = j = int(key)
                elif isinstance(key, (tuple, list)):
                    i, j = int(key[0]), int(key[1])
                else:
                    i = j = int(key)
                
                val = float(value) if value is not None else 0.0
                
                if i == j:
                    # Diagonal term - node weight
                    G.add_node(i, weight=val)
                else:
                    # Off-diagonal term - edge weight
                    G.add_edge(i, j, weight=val)
            except:
                continue
        
        return G
    
    def compute_structural_features(self, G):
        """Compute structural features for each node"""
        features = {}
        
        # Degree centrality
        degree_cent = nx.degree_centrality(G)
        
        # Betweenness centrality
        try:
            between_cent = nx.betweenness_centrality(G)
        except:
            between_cent = {node: 0 for node in G.nodes()}
        
        # Clustering coefficient
        try:
            clustering = nx.clustering(G)
        except:
            clustering = {node: 0 for node in G.nodes()}
        
        # Closeness centrality
        try:
            closeness_cent = nx.closeness_centrality(G)
        except:
            closeness_cent = {node: 0 for node in G.nodes()}
        
        for node in G.nodes():
            features[node] = np.array([
                degree_cent.get(node, 0),
                between_cent.get(node, 0),
                clustering.get(node, 0),
                closeness_cent.get(node, 0),
                G.nodes[node].get('weight', 0),  # Node weight from QUBO
                len(list(G.neighbors(node)))     # Degree
            ])
        
        return features
    
    def compute_similarity_matrix(self, G1, G2):
        """Compute similarity matrix between nodes of two graphs"""
        features1 = self.compute_structural_features(G1)
        features2 = self.compute_structural_features(G2)
        
        nodes1 = list(G1.nodes())
        nodes2 = list(G2.nodes())
        
        similarity_matrix = np.zeros((len(nodes1), len(nodes2)))
        
        for i, node1 in enumerate(nodes1):
            for j, node2 in enumerate(nodes2):
                feat1 = features1[node1]
                feat2 = features2[node2]
                
                # Cosine similarity between features
                sim = cosine_similarity([feat1], [feat2])[0, 0]
                similarity_matrix[i, j] = sim
        
        return similarity_matrix, nodes1, nodes2
    
    def iterative_matching(self, G1, G2):
        """Iterative matching algorithm similar to SecGraph"""
        similarity_matrix, nodes1, nodes2 = self.compute_similarity_matrix(G1, G2)
        
        # Initial matching using Hungarian algorithm
        row_ind, col_ind = linear_sum_assignment(-similarity_matrix)
        
        # Iteratively refine matching
        for iteration in range(self.max_iterations):
            new_similarity = similarity_matrix.copy()
            
            # Update similarities based on neighborhood consistency
            for i, node1 in enumerate(nodes1):
                for j, node2 in enumerate(nodes2):
                    
                    # Neighborhood similarity
                    neighbors1 = set(G1.neighbors(node1))
                    neighbors2 = set(G2.neighbors(node2))
                    
                    neighbor_sim = 0
                    if neighbors1 and neighbors2:
                        # Check how many neighbors are also matched
                        matched_neighbors = 0
                        for n1 in neighbors1:
                            if n1 in nodes1:
                                n1_idx = nodes1.index(n1)
                                if n1_idx < len(col_ind) and col_ind[n1_idx] < len(nodes2):
                                    matched_n2 = nodes2[col_ind[n1_idx]]
                                    if matched_n2 in neighbors2:
                                        matched_neighbors += 1
                        
                        neighbor_sim = matched_neighbors / max(len(neighbors1), len(neighbors2))
                    
                    # Combined similarity
                    new_similarity[i, j] = (
                        self.alpha * similarity_matrix[i, j] + 
                        (1 - self.alpha) * neighbor_sim
                    )
            
            # Re-match with updated similarities
            row_ind, col_ind = linear_sum_assignment(-new_similarity)
            similarity_matrix = new_similarity
        
        # Compute final matching score
        total_similarity = similarity_matrix[row_ind, col_ind].sum()
        max_possible = len(row_ind)
        
        matching_score = total_similarity / max_possible if max_possible > 0 else 0
        
        return matching_score
    
    def compute_graph_similarity(self, qubo1, qubo2):
        """Compute similarity between two QUBO graphs"""
        G1 = self.qubo_to_networkx(qubo1)
        G2 = self.qubo_to_networkx(qubo2)
        
        if len(G1.nodes()) == 0 or len(G2.nodes()) == 0:
            return 0.0
        
        # Graph-level similarity
        try:
            matching_score = self.iterative_matching(G1, G2)
        except:
            matching_score = 0.0
        
        # Size similarity penalty
        size_ratio = min(len(G1.nodes()), len(G2.nodes())) / max(len(G1.nodes()), len(G2.nodes()))
        
        return matching_score * size_ratio
    
    def evaluate_retrieval(self, query_qubos, gallery_qubos, gallery_ids, top_k=5):
        """Evaluate retrieval performance"""
        recalls = []
        
        for i, query_qubo in enumerate(query_qubos):
            similarities = []
            for gallery_qubo in gallery_qubos:
                sim = self.compute_graph_similarity(query_qubo, gallery_qubo)
                similarities.append(sim)
            
            # Find top-k most similar
            similarities = np.array(similarities)
            top_k_indices = np.argsort(similarities)[::-1][:top_k]
            
            # Check if correct match is in top-k
            correct_id = gallery_ids[i]
            predicted_ids = [gallery_ids[idx] for idx in top_k_indices]
            
            recalls.append(1 if correct_id in predicted_ids else 0)
        
        return np.mean(recalls)


    

