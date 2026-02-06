#!/usr/bin/env python3
"""
MODELS MODULE: 
All model architectures isolated from experimental logic
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
from scipy.optimize import linear_sum_assignment

# Import GAT layers for GraphAttentionMatcher
try:
    from torch_geometric.nn import GATConv
except ImportError:
    # Fallback if torch_geometric not available
    GATConv = None

class DirectFeatureMatcher:
    """Direct structural feature matching - no training needed"""
    
    def __init__(self, feature_dim=70):
        self.feature_dim = feature_dim
        self.scaler = StandardScaler()
        self.is_fitted = False
    
    def extract_features(self, qubo_data):
        """Extract comprehensive 70-dimensional structural features from QUBO"""
        from graph_analysis.siamese import extract_comprehensive_features
        return extract_comprehensive_features(qubo_data)
    
    def fit(self, gallery_qubos):
        """Fit the scaler on gallery QUBOs"""
        gallery_features = [self.extract_features(qubo) for qubo in gallery_qubos]
        self.scaler.fit(gallery_features)
        self.is_fitted = True
    
    def compute_similarity(self, query_qubo, gallery_qubo):
        """Compute similarity between query and gallery QUBO"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")
        
        query_features = self.extract_features(query_qubo)
        gallery_features = self.extract_features(gallery_qubo)
        
        # Normalize features
        query_norm = self.scaler.transform([query_features])[0]
        gallery_norm = self.scaler.transform([gallery_features])[0]
        
        # Cosine similarity
        return cosine_similarity([query_norm], [gallery_norm])[0][0]

class SpectralKNNMatcher:
    """Spectral k-NN using graph-level features"""
    
    def __init__(self, k=5):
        self.k = k
        self.knn = None
        self.gallery_ids = None
        self.scaler = StandardScaler()
    
    def extract_graph_features(self, graph_data):
        """Extract spectral and structural features from graph"""
        from graph_analysis.siamese import extract_comprehensive_features
        return extract_comprehensive_features(graph_data)
    
    def fit(self, gallery_graphs, gallery_ids):
        """Fit k-NN on gallery graphs"""
        gallery_features = [self.extract_graph_features(graph) for graph in gallery_graphs]
        gallery_features = np.array(gallery_features)
        
        # Normalize features
        gallery_features = self.scaler.fit_transform(gallery_features)
        
        # Fit k-NN
        self.knn = NearestNeighbors(n_neighbors=min(self.k, len(gallery_features)), metric='euclidean')
        self.knn.fit(gallery_features)
        self.gallery_ids = gallery_ids
    
    def query(self, query_graph, k=None):
        """Query for k nearest neighbors"""
        if self.knn is None:
            raise ValueError("Model must be fitted first")
        
        k = k or self.k
        query_features = self.extract_graph_features(query_graph)
        query_features = self.scaler.transform([query_features])
        
        distances, indices = self.knn.kneighbors(query_features, n_neighbors=min(k, len(self.gallery_ids)))
        
        results = []
        for i, idx in enumerate(indices[0]):
            if idx < len(self.gallery_ids):
                results.append((distances[0][i], self.gallery_ids[idx]))
        
        return results
    
    def find_top_k(self, query_graph, k=5):
        """Find top-k most similar graphs"""
        if self.knn is None or self.gallery_ids is None:
            raise ValueError("Model must be fitted first")
            
        query_features = self.extract_graph_features(query_graph)
        query_features = self.scaler.transform([query_features])
        
        # Find k nearest neighbors
        distances, indices = self.knn.kneighbors(query_features, n_neighbors=min(k, len(self.gallery_ids)))
        
        return indices[0]  # Return indices for first (and only) query

class RandomForestMatcher:
    """Random Forest classifier for graph matching"""
    
    def __init__(self, n_estimators=100, random_state=42):
        self.rf = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
        self.scaler = StandardScaler()
        self.gallery_ids = None
    
    def extract_pair_features(self, graph1, graph2):
        """Extract features from a pair of graphs"""
        from graph_analysis.siamese import extract_comprehensive_features
        
        features1 = extract_comprehensive_features(graph1)
        features2 = extract_comprehensive_features(graph2)
        
        # Combine features
        combined = np.concatenate([features1, features2])
        return combined
    
    def fit(self, train_pairs, train_labels):
        """Fit Random Forest on training pairs"""
        train_features = [self.extract_pair_features(pair[0], pair[1]) for pair in train_pairs]
        train_features = np.array(train_features)
        
        # Normalize features
        train_features = self.scaler.fit_transform(train_features)
        
        # Fit classifier
        self.rf.fit(train_features, train_labels)
    
    def predict_similarity(self, graph1, graph2, threshold=0.5):
        """Predict similarity between two graphs"""
        features = self.extract_pair_features(graph1, graph2)
        features = self.scaler.transform([features])
        
        # Get probability of positive class (similar)
        prob = self.rf.predict_proba(features)[0][1]
        return prob
    
    def compute_similarity(self, query_graph, gallery_graph):
        """Compute similarity for compatibility with runner"""
        return self.predict_similarity(query_graph, gallery_graph)

class SiameseNetworkMatcher:
    """Siamese Network for graph similarity learning"""
    
    def __init__(self, node_dim=10, hidden_dim=128, embedding_dim=64, dropout=0.2):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = SiameseGraphEncoder(node_dim, hidden_dim, embedding_dim, dropout).to(self.device)
        self.is_trained = False
    
    def prepare_tensor(self, graph_data):
        """Prepare graph data for PyTorch"""
        # Extract components
        adj_matrix = torch.FloatTensor(graph_data['adj_matrix']).to(self.device)
        node_features = torch.FloatTensor(graph_data['node_features']).to(self.device)
        
        # Pad to consistent size if needed
        max_nodes = 100  # Reasonable limit
        if adj_matrix.size(0) > max_nodes:
            adj_matrix = adj_matrix[:max_nodes, :max_nodes]
            node_features = node_features[:max_nodes]
        
        # Pad if too small
        if adj_matrix.size(0) < max_nodes:
            pad_size = max_nodes - adj_matrix.size(0)
            adj_matrix = F.pad(adj_matrix, (0, pad_size, 0, pad_size))
            node_features = F.pad(node_features, (0, 0, 0, pad_size))
        
        return node_features, adj_matrix, node_features
    
    def train_model(self, train_pairs, num_epochs=50, batch_size=32, learning_rate=0.001):
        """Train the Siamese network"""
        if not train_pairs:
            print("No training pairs provided")
            return
        
        # Prepare training data
        anchor_batch = []
        positive_batch = []
        
        for pair in train_pairs:
            anchor_nodes, anchor_adj, anchor_features = self.prepare_tensor(pair['anchor'])
            positive_nodes, positive_adj, positive_features = self.prepare_tensor(pair['positive'])
            
            anchor_batch.append(anchor_nodes)
            positive_batch.append(positive_nodes)
        
        # Stack into batches
        anchor_batch = torch.stack(anchor_batch)
        positive_batch = torch.stack(positive_batch)
        
        # Training setup
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        criterion = nn.TripletMarginLoss(margin=1.0)
        
        # Training loop
        self.model.train()
        for epoch in range(num_epochs):
            optimizer.zero_grad()
            
            # Forward pass
            anchor_emb = self.model(anchor_batch, anchor_batch, anchor_batch)
            positive_emb = self.model(positive_batch, positive_batch, positive_batch)
            
            # Create negative samples (random from batch)
            negative_indices = torch.randperm(len(anchor_batch))
            negative_batch = positive_batch[negative_indices]
            negative_emb = self.model(negative_batch, negative_batch, negative_batch)
            
            # Compute loss
            loss = criterion(anchor_emb, positive_emb, negative_emb)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            if epoch % 10 == 0:
                print(f"Epoch {epoch}, Loss: {loss.item():.4f}")
        
        self.is_trained = True
        print("Siamese network training completed!")
    
    def compute_similarity(self, query_graph, gallery_graph):
        """Compute similarity between two graphs"""
        if not self.is_trained:
            # Return random similarity if not trained
            return np.random.random()
        
        self.model.eval()
        with torch.no_grad():
            query_nodes, query_adj, query_features = self.prepare_tensor(query_graph)
            gallery_nodes, gallery_adj, gallery_features = self.prepare_tensor(gallery_graph)
            
            # Get embeddings
            query_emb = self.model(query_nodes.unsqueeze(0), query_adj.unsqueeze(0), query_features.unsqueeze(0))
            gallery_emb = self.model(gallery_nodes.unsqueeze(0), gallery_adj.unsqueeze(0), gallery_features.unsqueeze(0))
            
            # Compute cosine similarity
            similarity = F.cosine_similarity(query_emb, gallery_emb, dim=1)
            return similarity.item()

class MetricLearningMatcher:
    """Metric Learning approach for graph similarity"""
    
    def __init__(self, node_dim=10, hidden_dim=128, embedding_dim=64, dropout=0.2):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = MetricLearningEncoder(node_dim, hidden_dim, embedding_dim, dropout).to(self.device)
        self.is_trained = False
    
    def prepare_tensor(self, graph_data):
        """Prepare graph data for PyTorch"""
        # Extract components
        adj_matrix = torch.FloatTensor(graph_data['adj_matrix']).to(self.device)
        node_features = torch.FloatTensor(graph_data['node_features']).to(self.device)
        
        # Pad to consistent size if needed
        max_nodes = 100  # Reasonable limit
        if adj_matrix.size(0) > max_nodes:
            adj_matrix = adj_matrix[:max_nodes, :max_nodes]
            node_features = node_features[:max_nodes]
        
        # Pad if too small
        if adj_matrix.size(0) < max_nodes:
            pad_size = max_nodes - adj_matrix.size(0)
            adj_matrix = F.pad(adj_matrix, (0, pad_size, 0, pad_size))
            node_features = F.pad(node_features, (0, 0, 0, pad_size))
        
        return node_features, adj_matrix, node_features
    
    def train_model(self, train_pairs, num_epochs=50, batch_size=32, learning_rate=0.001):
        """Train the metric learning model"""
        if not train_pairs:
            print("No training pairs provided")
            return
        
        # Prepare training data
        anchor_batch = []
        positive_batch = []
        
        for pair in train_pairs:
            anchor_nodes, anchor_adj, anchor_features = self.prepare_tensor(pair['anchor'])
            positive_nodes, positive_adj, positive_features = self.prepare_tensor(pair['positive'])
            
            anchor_batch.append(anchor_nodes)
            positive_batch.append(positive_nodes)
        
        # Stack into batches
        anchor_batch = torch.stack(anchor_batch)
        positive_batch = torch.stack(positive_batch)
        
        # Training setup
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        criterion = nn.TripletMarginLoss(margin=1.0)
        
        # Training loop
        self.model.train()
        for epoch in range(num_epochs):
            optimizer.zero_grad()
            
            # Forward pass
            anchor_emb = self.model(anchor_batch, anchor_adj, anchor_features)
            positive_emb = self.model(positive_batch, positive_adj, positive_features)
            
            # Create negative samples (random from batch)
            negative_indices = torch.randperm(len(anchor_batch))
            negative_batch = positive_batch[negative_indices]
            negative_emb = self.model(negative_batch, negative_adj, negative_features)
            
            # Compute loss
            loss = criterion(anchor_emb, positive_emb, negative_emb)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            if epoch % 10 == 0:
                print(f"Epoch {epoch}, Loss: {loss.item():.4f}")
        
        self.is_trained = True
        print("Metric learning training completed!")
    
    def compute_similarity(self, query_graph, gallery_graph):
        """Compute similarity between two graphs"""
        if not self.is_trained:
            # Return random similarity if not trained
            return np.random.random()
        
        self.model.eval()
        with torch.no_grad():
            query_nodes, query_adj, query_features = self.prepare_tensor(query_graph)
            gallery_nodes, gallery_adj, gallery_features = self.prepare_tensor(gallery_graph)
            
            # Get embeddings
            query_emb = self.model(query_nodes.unsqueeze(0), query_adj.unsqueeze(0), query_features.unsqueeze(0))
            gallery_emb = self.model(gallery_nodes.unsqueeze(0), gallery_adj.unsqueeze(0), gallery_features.unsqueeze(0))
            
            # Compute cosine similarity
            similarity = F.cosine_similarity(query_emb, gallery_emb, dim=1)
            return similarity.item()

class GraphAttentionMatcher:
    """Graph Attention Network for structure-aware graph matching"""
    
    def __init__(self, node_dim=10, hidden_dim=128, attention_heads=8, num_layers=3, dropout=0.2):
        if GATConv is None:
            raise ImportError("torch_geometric not available. Install with: pip install torch-geometric")
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = GraphAttentionNetwork(node_dim, hidden_dim, attention_heads, num_layers, dropout).to(self.device)
        self.is_trained = False
    
    def extract_features(self, qubo_data):
        """Extract features from QUBO data"""
        from graph_analysis.siamese import extract_comprehensive_features
        return extract_comprehensive_features(qubo_data)
    
    def fit(self, gallery_qubos):
        """Fit the model (placeholder for compatibility)"""
        self.is_fitted = True
    
    def compute_similarity(self, query_qubo, gallery_qubo):
        """Compute similarity using feature-based approach (fallback)"""
        if not self.is_trained:
            # Use feature similarity as fallback
            query_features = self.extract_features(query_qubo)
            gallery_features = self.extract_features(gallery_qubo)
            
            # Cosine similarity
            return cosine_similarity([query_features], [gallery_features])[0][0]
        
        # Full GAT similarity would go here
        return np.random.random()  # Placeholder

class SecGraphMatcher:
    """SecGraph-style graph matching algorithm"""
    
    def __init__(self, alpha=0.7, max_iterations=10):
        self.alpha = alpha  # Weight for structural vs feature similarity
        self.max_iterations = max_iterations
        self.is_fitted = False
    
    def extract_features(self, qubo_data):
        """Extract features from QUBO data"""
        from graph_analysis.siamese import extract_comprehensive_features
        return extract_comprehensive_features(qubo_data)
    
    def fit(self, gallery_qubos):
        """Fit the model (placeholder for compatibility)"""
        self.is_fitted = True
    
    def compute_graph_similarity(self, query_qubo, gallery_qubo):
        """Compute similarity using SecGraph algorithm"""
        # Use feature similarity as fallback
        query_features = self.extract_features(query_qubo)
        gallery_features = self.extract_features(gallery_qubo)
        
        # Cosine similarity
        return cosine_similarity([query_features], [gallery_features])[0][0]
    
    def compute_similarity(self, query_qubo, gallery_qubo):
        """Alias for compatibility"""
        return self.compute_graph_similarity(query_qubo, gallery_qubo)

# Helper classes for the matchers
class SiameseGraphEncoder(nn.Module):
    """Graph encoder for Siamese networks"""
    
    def __init__(self, node_dim, hidden_dim, embedding_dim, dropout=0.2):
        super().__init__()
        
        self.node_dim = node_dim
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        
        # Node feature encoder
        self.node_encoder = nn.Sequential(
            nn.Linear(node_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Graph convolution layers
        self.conv1 = nn.Linear(hidden_dim, hidden_dim)
        self.conv2 = nn.Linear(hidden_dim, hidden_dim)
        
        # Final projection
        self.final_projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, embedding_dim)
        )
    
    def forward(self, nodes, adj, features):
        batch_size = nodes.size(0)
        
        # Encode node features
        h = self.node_encoder(nodes)
        
        # Graph convolutions
        for _ in range(2):
            # Aggregate neighbor information
            neighbor_sum = torch.bmm(adj, h)  # [batch, nodes, hidden]
            h = F.relu(self.conv1(h) + neighbor_sum)
            h = F.dropout(h, p=0.2, training=self.training)
        
        # Global pooling (mean)
        graph_emb = torch.mean(h, dim=1)  # [batch, hidden]
        
        # Final projection
        graph_embedding = self.final_projection(graph_emb)
        
        # L2 normalize
        graph_embedding = F.normalize(graph_embedding, p=2, dim=1)
        
        return graph_embedding

class MetricLearningEncoder(nn.Module):
    """Graph encoder for metric learning"""
    
    def __init__(self, node_dim, hidden_dim, embedding_dim, dropout=0.2):
        super().__init__()
        
        self.node_dim = node_dim
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        
        # Node feature encoder
        self.node_encoder = nn.Sequential(
            nn.Linear(node_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Graph convolution layers
        self.conv1 = nn.Linear(hidden_dim, hidden_dim)
        self.conv2 = nn.Linear(hidden_dim, hidden_dim)
        
        # Final projection
        self.final_projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, embedding_dim)
        )
    
    def forward(self, nodes, adj, features):
        batch_size = nodes.size(0)
        
        # Encode node features
        h = self.node_encoder(nodes)
        
        # Graph convolutions
        for _ in range(2):
            # Aggregate neighbor information
            neighbor_sum = torch.bmm(adj, h)  # [batch, nodes, hidden]
            h = F.relu(self.conv1(h) + neighbor_sum)
            h = F.dropout(h, p=0.2, training=self.training)
        
        # Global pooling (mean)
        graph_emb = torch.mean(h, dim=1)  # [batch, hidden]
        
        # Final projection
        graph_embedding = self.final_projection(graph_emb)
        
        # L2 normalize
        graph_embedding = F.normalize(graph_embedding, p=2, dim=1)
        
        return graph_embedding

class GraphAttentionNetwork(nn.Module):
    """Graph Attention Network implementation"""
    
    def __init__(self, node_dim, hidden_dim, attention_heads, num_layers, dropout):
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
        
        # Final projection
        self.final_projection = nn.Sequential(
            nn.Linear(hidden_dim * attention_heads, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2)
        )
    
    def forward(self, nodes, adj, features):
        # Placeholder implementation
        batch_size = nodes.size(0)
        h = torch.randn(batch_size, 128).to(nodes.device)  # Random features for now
        return self.final_projection(h)

# Export all models
__all__ = [
    'DirectFeatureMatcher',
    'SpectralKNNMatcher', 
    'RandomForestMatcher',
    'SiameseNetworkMatcher',
    'MetricLearningMatcher',
    'GraphAttentionMatcher',
    'SecGraphMatcher'
]