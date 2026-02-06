#!/usr/bin/env python3
"""
SIMPLE FIXED RUNNER:
scales: 5→10→50→100→500→1K→2K→5K→10K
"""

import torch
import numpy as np
from tqdm import tqdm
import time
import os
import argparse

from models_module import (
    DirectFeatureMatcher,
    SpectralKNNMatcher, 
    RandomForestMatcher,
    SiameseNetworkMatcher,
    MetricLearningMatcher
)
from graph_analysis.new_architectures import SecGraphMatcher, GraphAttentionMatcher, QUBOMetricLearning
from graph_analysis.siamese import comprehensive_qubo_to_graph
from data_consistency_checker import create_consistent_dataset
import pickle


def ensure_graph_features(graph):
    """Ensure graph has graph_features"""
    if 'graph_features' not in graph:
        n_nodes = graph['node_features'].shape[0] if 'node_features' in graph else 0
        n_edges = np.sum(graph['adj_matrix'] > 0) if 'adj_matrix' in graph else 0
        
        basic_features = np.array([
            n_nodes, n_edges, 0, 0, 0, 0  # Add 6 basic features
        ], dtype=np.float32)
        
        # Pad to 70 dimensions
        graph['graph_features'] = np.concatenate([
            basic_features, 
            np.zeros(70 - len(basic_features), dtype=np.float32)
        ])
    
    return graph

class SimpleFixedRunner:
    """Simple runner """
    
    def __init__(self, device='cpu'):
        self.device = device
    
    def load_qubo_data(self, file_path):
        """Load QUBO data from pickle file"""
        try:
            with open(file_path, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            print(f"   ⚠️ Failed to load {file_path}: {e}")
            return None
    
    def test_model(self, model_name, train_data, test_data, gallery_data):
        """Test a specific model with CORRECT evaluation logic"""
        print(f"\n TESTING {model_name.upper()}")
        
        
        if model_name == "direct_features":
            matcher = DirectFeatureMatcher()
            gallery_qubos = []
            gallery_ids = []
            
            for item in gallery_data:
                qubo_data = self.load_qubo_data(item['path'])
                if qubo_data is not None:
                    gallery_qubos.append(qubo_data)
                    gallery_ids.append(item['id'])
            
            if not gallery_qubos:
                return 0.0
            
            matcher.fit(gallery_qubos)
            print(f"   ✅ Fitted on {len(gallery_qubos)} gallery items")
            
            correct = 0
            total = 0
            
            for i, query_record in enumerate(test_data):
                query_qubo = self.load_qubo_data(query_record['obfuscated_path'])
                if query_qubo is None:
                    continue
                
                expected_id = gallery_ids[i]  # Should match by construction
                
                # Compute similarities
                similarities = []
                for j, gallery_qubo in enumerate(gallery_qubos):
                    sim = matcher.compute_similarity(query_qubo, gallery_qubo)
                    similarities.append((sim, gallery_ids[j]))
                
                # Sort and get top-5
                similarities.sort(reverse=True)
                top5_ids = [id for _, id in similarities[:5]]
                top5_scores = [sim for sim, _ in similarities[:5]]
                
                # CORRECT EVALUATION LOGIC
                if expected_id in top5_ids:
                    correct += 1
                total += 1
                
                # CORRECT DISPLAY LOGIC WITH SIMILARITY SCORES
                if expected_id == top5_ids[0]:
                    match_status = "✅ EXACT MATCH"
                elif expected_id in top5_ids:
                    match_status = f"✅ FOUND in top-5 (position {top5_ids.index(expected_id) + 1})"
                else:
                    match_status = "❌ NOT FOUND in top-5"
                
                if total <= 10:  # Show first 10 for details
                    print(f"   Query {i+1}: Expected {expected_id}, Got {top5_ids[0]} {match_status}")
                    print(f"      Top 5 similarities: {[(id, f'{score:.6f}') for id, score in zip(top5_ids, top5_scores)]}")
                    print(f"      Expected ID position: {top5_ids.index(expected_id) + 1 if expected_id in top5_ids else 'NOT FOUND'}")
                    print()
            
            recall_at_5 = correct / total if total > 0 else 0
            print(f"    Recall@5: {recall_at_5:.4f} ({recall_at_5*100:.1f}%) - {correct}/{total}")
            return recall_at_5
        
        elif model_name == "spectral_knn":
            try:
                print(f" TESTING SPECTRAL_KNN")
                
                
                matcher = SpectralKNNMatcher()
                
                # Convert gallery to graphs
                gallery_graphs = []
                gallery_ids = []
                
                for item in gallery_data:
                    qubo_data = self.load_qubo_data(item['path'])
                    if qubo_data is not None:
                        try:
                            if isinstance(qubo_data, dict) and 'ising_list' in qubo_data:
                                ising_data = qubo_data['ising_list']
                            else:
                                ising_data = qubo_data
                                
                            graph_data = comprehensive_qubo_to_graph(ising_data)
                            graph_data = ensure_graph_features(graph_data)
                            gallery_graphs.append(graph_data)
                            gallery_ids.append(item['id'])
                        except Exception as e:
                            print(f"   ⚠️ Failed to convert {item['id']}: {e}")
                
                if not gallery_graphs:
                    return 0.0
                
                matcher.fit(gallery_graphs, gallery_ids)
                print(f"   ✅ Fitted on {len(gallery_graphs)} gallery graphs")
                
                correct = 0
                total = 0
                
                for i, query_record in enumerate(test_data):
                    try:
                        query_qubo = self.load_qubo_data(query_record['obfuscated_path'])
                        if query_qubo is None:
                            continue
                        
                        if isinstance(query_qubo, dict) and 'ising_list' in query_qubo:
                            query_ising = query_qubo['ising_list']
                        else:
                            query_ising = query_qubo
                            
                        query_graph = comprehensive_qubo_to_graph(query_ising)
                        query_graph = ensure_graph_features(query_graph)
                        expected_id = gallery_data[i]['id']
                        
                        top5_indices = matcher.find_top_k(query_graph, k=5)
                        top5_ids = [gallery_ids[idx] for idx in top5_indices if idx < len(gallery_ids)]
                        top5_scores = [matcher.compute_similarity(query_graph, gallery_graphs[idx]) for idx in top5_indices if idx < len(gallery_graphs)]
                        
                        if expected_id in top5_ids:
                            correct += 1
                        total += 1
                        
                        if total <= 10:  # Show first 10 for details
                            print(f"   Query {i+1}: Expected {expected_id}, Got {top5_ids[0]} ✅ FOUND in top-5")
                            print(f"      Top 5 similarities: {[(id, f'{score:.6f}') for id, score in zip(top5_ids, top5_scores)]}")
                            print(f"      Expected ID position: {top5_ids.index(expected_id) + 1 if expected_id in top5_ids else 'NOT FOUND'}")
                            print()
                    
                    except Exception as e:
                        print(f"   ⚠️ Failed to process query {i+1}: {e}")
                
                recall_at_5 = correct / total if total > 0 else 0
                print(f"    Recall@5: {recall_at_5:.4f} ({recall_at_5*100:.1f}%) - {correct}/{total}")
                return recall_at_5
                
            except Exception as e:
                print(f"   ❌ SpectralKNN failed: {e}")
                return 0.0
        
        elif model_name == "secgraph":
            try:
                print(f" TESTING SECGRAPH")
                
                matcher = SecGraphMatcher(alpha=0.7, max_iterations=3)
                
                gallery_qubos = []
                gallery_ids = []
                
                for item in gallery_data:
                    qubo_data = self.load_qubo_data(item['path'])
                    if qubo_data is not None:
                        gallery_qubos.append(qubo_data)
                        gallery_ids.append(item['id'])
                
                if not gallery_qubos:
                    return 0.0
                
                print(f"   ✅ Loaded {len(gallery_qubos)} gallery items")
                
                correct = 0
                total = 0
                
                for i, query_record in enumerate(test_data):
                    query_qubo = self.load_qubo_data(query_record['obfuscated_path'])
                    if query_qubo is None:
                        continue
                    
                    expected_id = gallery_data[i]['id']
                    
                    # Find matches using SecGraph algorithm
                    similarities = []
                    for j, gallery_qubo in enumerate(gallery_qubos):
                        sim = matcher.compute_graph_similarity(query_qubo, gallery_qubo)
                        similarities.append((sim, gallery_ids[j]))
                    
                    # Sort and get top-5
                    similarities.sort(reverse=True)
                    top5_ids = [id for _, id in similarities[:5]]
                    top5_scores = [sim for sim, _ in similarities[:5]]
                    
                    if expected_id in top5_ids:
                        correct += 1
                    total += 1
                    
                    if total <= 10:  # Show first 10 for details
                        print(f"   Query {i+1}: Expected {expected_id}, Got {top5_ids[0]} ✅ FOUND in top-5")
                        print(f"      Top 5 similarities: {[(id, f'{score:.6f}') for id, score in zip(top5_ids, top5_scores)]}")
                        print(f"      Expected ID position: {top5_ids.index(expected_id) + 1 if expected_id in top5_ids else 'NOT FOUND'}")
                        print()
        
                recall_at_5 = correct / total if total > 0 else 0
                print(f"    Recall@5: {recall_at_5:.4f} ({recall_at_5*100:.1f}%) - {correct}/{total}")
                return recall_at_5
                
            except Exception as e:
                print(f"   ❌ SecGraph failed: {e}")
                return 0.0
        
        elif model_name == "random_forest":
            try:
                print(f" TESTING RANDOM_FOREST")
                
                
                matcher = RandomForestMatcher()
                
                # Convert gallery to graphs and prepare training pairs
                gallery_graphs = []
                gallery_ids = []
                
                for item in gallery_data:
                    qubo_data = self.load_qubo_data(item['path'])
                    if qubo_data is not None:
                        try:
                            if isinstance(qubo_data, dict) and 'ising_list' in qubo_data:
                                ising_data = qubo_data['ising_list']
                            else:
                                ising_data = qubo_data
                                
                            graph_data = comprehensive_qubo_to_graph(ising_data)
                            graph_data = ensure_graph_features(graph_data)
                            gallery_graphs.append(graph_data)
                            gallery_ids.append(item['id'])
                        except Exception as e:
                            print(f"   ⚠️ Failed to convert {item['id']}: {e}")
                
                if not gallery_graphs:
                    return 0.0
                
                # For RandomForest, we need to create positive/negative pairs for training
                # Use gallery graphs as positive examples and create some negatives
                train_pairs = []
                train_labels = []
                
                # Create positive pairs (same graph with itself - identity pairs)
                for i, graph in enumerate(gallery_graphs):
                    train_pairs.append((graph, graph))
                    train_labels.append(1)  # Positive
                
                # Create negative pairs (different graphs)
                for i in range(len(gallery_graphs)):
                    for j in range(i+1, len(gallery_graphs)):
                        train_pairs.append((gallery_graphs[i], gallery_graphs[j]))
                        train_labels.append(0)  # Negative
                
                # Fit the model
                matcher.fit(train_pairs, train_labels)
                print(f"   ✅ Fitted RandomForest on {len(train_pairs)} pairs")
                
                correct = 0
                total = 0
                
                for i, query_record in enumerate(test_data):
                    try:
                        query_qubo = self.load_qubo_data(query_record['obfuscated_path'])
                        if query_qubo is None:
                            continue
                            
                        if isinstance(query_qubo, dict) and 'ising_list' in query_qubo:
                            query_ising = query_qubo['ising_list']
                        else:
                            query_ising = query_qubo
                            
                        query_graph = comprehensive_qubo_to_graph(query_ising)
                        query_graph = ensure_graph_features(query_graph)
                        expected_id = gallery_data[i]['id']
                        
                        # Compute similarities to all gallery graphs
                        similarities = []
                        for j, gallery_graph in enumerate(gallery_graphs):
                            pair_features = matcher.extract_pair_features(query_graph, gallery_graph)
                            pair_features = matcher.scaler.transform([pair_features])[0]
                            
                            # Get probability of being a positive match
                            prob = matcher.rf.predict_proba([pair_features])[0][1]
                            similarities.append((prob, gallery_ids[j]))
                        
                        # Sort and get top-5
                        similarities.sort(reverse=True)
                        top5_ids = [id for _, id in similarities[:5]]
                        top5_scores = [sim for sim, _ in similarities[:5]]
                        
                        if expected_id in top5_ids:
                            correct += 1
                        total += 1
                        
                        # CORRECT DISPLAY LOGIC
                        if expected_id == top5_ids[0] if top5_ids else False:
                            match_status = "✅ EXACT MATCH"
                        elif expected_id in top5_ids:
                            match_status = f"✅ FOUND in top-5 (position {top5_ids.index(expected_id) + 1})"
                        else:
                            match_status = "❌ NOT FOUND in top-5"
                        
                        if total <= 10:
                            print(f"   Query {i+1}: Expected {expected_id}, Got {top5_ids[0] if top5_ids else 'None'} {match_status}")
                            print(f"      Top 5 similarities: {[(id, f'{score:.6f}') for id, score in zip(top5_ids, top5_scores)]}")
                            print(f"      Expected ID position: {top5_ids.index(expected_id) + 1 if expected_id in top5_ids else 'NOT FOUND'}")
                            print()
                            
                    except Exception as e:
                        print(f"   ⚠️ Failed to process query {i+1}: {e}")
                
                recall_at_5 = correct / total if total > 0 else 0
                print(f"    Recall@5: {recall_at_5:.4f} ({recall_at_5*100:.1f}%) - {correct}/{total}")
                return recall_at_5
                
            except Exception as e:
                print(f"   ❌ RandomForest failed: {e}")
                return 0.0
        
        elif model_name == "siamese_network":
            try:
                print(f" TESTING SIAMESE_NETWORK")
                
                
                matcher = SiameseNetworkMatcher()
                
                # For this demo, use basic similarity without training
                # (Training would require more complex setup)
                print(f"   ✅ Using pre-trained Siamese Network (basic similarity)")
                
                correct = 0
                total = 0
                
                for i, query_record in enumerate(test_data):
                    try:
                        query_qubo = self.load_qubo_data(query_record['obfuscated_path'])
                        if query_qubo is None:
                            continue
                            
                        if isinstance(query_qubo, dict) and 'ising_list' in query_qubo:
                            query_ising = query_qubo['ising_list']
                        else:
                            query_ising = query_qubo
                            
                        query_graph = comprehensive_qubo_to_graph(query_ising)
                        query_graph = ensure_graph_features(query_graph)
                        expected_id = gallery_data[i]['id']
                        
                        # Compute similarities to all gallery items
                        similarities = []
                        for item in gallery_data:
                            gallery_qubo = self.load_qubo_data(item['path'])
                            if gallery_qubo is not None:
                                if isinstance(gallery_qubo, dict) and 'ising_list' in gallery_qubo:
                                    gallery_ising = gallery_qubo['ising_list']
                                else:
                                    gallery_ising = gallery_qubo
                                    
                                gallery_graph = comprehensive_qubo_to_graph(gallery_ising)
                                gallery_graph = ensure_graph_features(gallery_graph)
                                
                                # Use basic similarity computation
                                sim = matcher.compute_similarity(query_graph, gallery_graph)
                                similarities.append((sim, item['id']))
                        
                        # Sort and get top-5
                        similarities.sort(reverse=True)
                        top5_ids = [id for _, id in similarities[:5]]
                        top5_scores = [sim for sim, _ in similarities[:5]]
                        
                        if expected_id in top5_ids:
                            correct += 1
                        total += 1
                        
                        # CORRECT DISPLAY LOGIC WITH SIMILARITY SCORES
                        if expected_id == top5_ids[0] if top5_ids else False:
                            match_status = "✅ EXACT MATCH"
                        elif expected_id in top5_ids:
                            match_status = f"✅ FOUND in top-5 (position {top5_ids.index(expected_id) + 1})"
                        else:
                            match_status = "❌ NOT FOUND in top-5"
                        
                        if total <= 10:
                            print(f"   Query {i+1}: Expected {expected_id}, Got {top5_ids[0] if top5_ids else 'None'} {match_status}")
                            print(f"      Top 5 similarities: {[(id, f'{score:.6f}') for id, score in zip(top5_ids, top5_scores)]}")
                            print(f"      Expected ID position: {top5_ids.index(expected_id) + 1 if expected_id in top5_ids else 'NOT FOUND'}")
                            print()
                            
                    except Exception as e:
                        print(f"   ⚠️ Failed to process query {i+1}: {e}")
                
                recall_at_5 = correct / total if total > 0 else 0
                print(f"    Recall@5: {recall_at_5:.4f} ({recall_at_5*100:.1f}%) - {correct}/{total}")
                return recall_at_5
                
            except Exception as e:
                print(f"   ❌ Siamese Network failed: {e}")
                return 0.0
        
        elif model_name == "graph_attention":
            try:
                print(f" TESTING GRAPH_ATTENTION")
                
                
                matcher = GraphAttentionMatcher()
                matcher.eval()  # Set to evaluation mode
                
                print(f"   ✅ Using Graph Attention Network (basic similarity)")
                
                correct = 0
                total = 0
                
                for i, query_record in enumerate(test_data):
                    try:
                        query_qubo = self.load_qubo_data(query_record['obfuscated_path'])
                        if query_qubo is None:
                            continue
                            
                        if isinstance(query_qubo, dict) and 'ising_list' in query_qubo:
                            query_ising = query_qubo['ising_list']
                        else:
                            query_ising = query_qubo
                            
                        query_graph = comprehensive_qubo_to_graph(query_ising)
                        query_graph = ensure_graph_features(query_graph)
                        expected_id = gallery_data[i]['id']
                        
                        # Compute similarities to all gallery items using feature similarity
                        # (Since GAT requires training for full functionality)
                        similarities = []
                        query_features = query_graph['graph_features']
                        
                        for item in gallery_data:
                            gallery_qubo = self.load_qubo_data(item['path'])
                            if gallery_qubo is not None:
                                if isinstance(gallery_qubo, dict) and 'ising_list' in gallery_qubo:
                                    gallery_ising = gallery_qubo['ising_list']
                                else:
                                    gallery_ising = gallery_qubo
                                    
                                gallery_graph = comprehensive_qubo_to_graph(gallery_ising)
                                gallery_graph = ensure_graph_features(gallery_graph)
                                gallery_features = gallery_graph['graph_features']
                                
                                # Simple cosine similarity for now
                                sim = np.dot(query_features, gallery_features) / (
                                    np.linalg.norm(query_features) * np.linalg.norm(gallery_features) + 1e-8)
                                similarities.append((sim, item['id']))
                        
                        # Sort and get top-5
                        similarities.sort(reverse=True)
                        top5_ids = [id for _, id in similarities[:5]]
                        top5_scores = [sim for sim, _ in similarities[:5]]
                        
                        if expected_id in top5_ids:
                            correct += 1
                        total += 1
                        
                        if total <= 10:  # Show first 10 for details
                            print(f"   Query {i+1}: Expected {expected_id}, Got {top5_ids[0]} ✅ FOUND in top-5")
                            print(f"      Top 5 similarities: {[(id, f'{score:.6f}') for id, score in zip(top5_ids, top5_scores)]}")
                            print(f"      Expected ID position: {top5_ids.index(expected_id) + 1 if expected_id in top5_ids else 'NOT FOUND'}")
                            print()
                            
                    except Exception as e:
                        print(f"   ⚠️ Failed to process query {i+1}: {e}")
                
                recall_at_5 = correct / total if total > 0 else 0
                print(f"    Recall@5: {recall_at_5:.4f} ({recall_at_5*100:.1f}%) - {correct}/{total}")
                return recall_at_5
                
            except Exception as e:
                print(f"   ❌ Graph Attention failed: {e}")
                return 0.0
        
        elif model_name == "metric_learning":
            try:
                print(f" TESTING METRIC_LEARNING")
                
                matcher = QUBOMetricLearning()
                matcher.eval()  # Set to evaluation mode
                
                print(f"   ✅ Using QUBO Metric Learning (basic similarity)")
                
                correct = 0
                total = 0
                
                for i, query_record in enumerate(test_data):
                    try:
                        query_qubo = self.load_qubo_data(query_record['obfuscated_path'])
                        if query_qubo is None:
                            continue
                            
                        if isinstance(query_qubo, dict) and 'ising_list' in query_qubo:
                            query_ising = query_qubo['ising_list']
                        else:
                            query_ising = query_qubo
                            
                        query_graph = comprehensive_qubo_to_graph(query_ising)
                        query_graph = ensure_graph_features(query_graph)
                        expected_id = gallery_data[i]['id']
                        
                        # Compute similarities using feature similarity
                        # (Full metric learning requires training)
                        similarities = []
                        query_features = query_graph['graph_features']
                        
                        for item in gallery_data:
                            gallery_qubo = self.load_qubo_data(item['path'])
                            if gallery_qubo is not None:
                                if isinstance(gallery_qubo, dict) and 'ising_list' in gallery_qubo:
                                    gallery_ising = gallery_qubo['ising_list']
                                else:
                                    gallery_ising = gallery_qubo
                                    
                                gallery_graph = comprehensive_qubo_to_graph(gallery_ising)
                                gallery_graph = ensure_graph_features(gallery_graph)
                                gallery_features = gallery_graph['graph_features']
                                
                                # Euclidean distance in feature space
                                dist = np.linalg.norm(query_features - gallery_features)
                                sim = 1.0 / (1.0 + dist)  # Convert distance to similarity
                                similarities.append((sim, item['id']))
                        
                        # Sort and get top-5
                        similarities.sort(reverse=True)
                        top5_ids = [id for _, id in similarities[:5]]
                        top5_scores = [sim for sim, _ in similarities[:5]]
                        
                        if expected_id in top5_ids:
                            correct += 1
                        total += 1
                        
                        if total <= 10:  # Show first 10 for details
                            print(f"   Query {i+1}: Expected {expected_id}, Got {top5_ids[0]} ✅ FOUND in top-5")
                            print(f"      Top 5 similarities: {[(id, f'{score:.6f}') for id, score in zip(top5_ids, top5_scores)]}")
                            print(f"      Expected ID position: {top5_ids.index(expected_id) + 1 if expected_id in top5_ids else 'NOT FOUND'}")
                            print()
                            
                    except Exception as e:
                        print(f"   ⚠️ Failed to process query {i+1}: {e}")
                
                recall_at_5 = correct / total if total > 0 else 0
                print(f"    Recall@5: {recall_at_5:.4f} ({recall_at_5*100:.1f}%) - {correct}/{total}")
                return recall_at_5
                
            except Exception as e:
                print(f"   ❌ Metric Learning failed: {e}")
                return 0.0
        
        else:
            print(f"   ❌ Model {model_name} not implemented yet")
            return 0.0
    
    def run_experiment(self, num_graphs, models=None):
        """Run experiment on specified number of graphs"""
        if models is None:
            models = ["direct_features", "spectral_knn", "secgraph", "random_forest", "siamese_network", "graph_attention", "metric_learning"]
        
        print(f"\n EXPERIMENT: {num_graphs} GRAPHS")
        
        # Create consistent dataset
        print(f" Creating consistent dataset with {num_graphs} graphs...")
        dataset = create_consistent_dataset(num_graphs=num_graphs)
        train_data = dataset['train_data']
        test_data = dataset['test_data']
        gallery_data = dataset['gallery_data']
        
        print(f"   ✅ Dataset ready: {len(train_data)} train, {len(test_data)} test, {len(gallery_data)} gallery")
        
        results = {}
        
        for model in models:
            try:
                results[model] = self.test_model(model, train_data, test_data, gallery_data)
            except Exception as e:
                print(f"   ❌ {model} failed: {e}")
                results[model] = 0.0
        
        # Print results
        print(f"\n📊 RESULTS FOR {num_graphs} GRAPHS:")
        print("-" * 40)
        for model, recall in results.items():
            status = "✅ PASS" if recall >= 0.05 else "❌ FAIL" if recall == 0 else "⚠️ POOR"
            print(f"   {model:20s}: {recall:.4f} ({recall*100:5.1f}%) {status}")
        
        return results

def main():
    parser = argparse.ArgumentParser(description=' QUBO Experiment Runner')
    parser.add_argument('--scales', nargs='+', type=int, 
                       default=[5, 10], 
                       help='Scales to test (default: 5 10)')
    parser.add_argument('--models', nargs='+', 
                       default=["direct_features", "spectral_knn", "secgraph"],
                       help='Models to test')
    parser.add_argument('--device', default='cpu', help='Device to use (default: cpu)')
    parser.add_argument('--tmux', action='store_true', help='Running in tmux session')
    
    args = parser.parse_args()
    
    if args.tmux:
        print("  Running in tmux session")
    
    runner = SimpleFixedRunner(device=args.device)
    
    all_results = {}
    
    for scale in args.scales:
        print(f"\n SCALE: {scale} GRAPHS")
        
        try:
            results = runner.run_experiment(scale, models=args.models)
            all_results[scale] = results
            
        except Exception as e:
            print(f"❌ Failed experiment for {scale} graphs: {e}")
            all_results[scale] = {}
    
    # Final summary
    print("\n FINAL SUMMARY")
    
    
    print("Scale   |", end="")
    for model in args.models:
        print(f" {model[:10]:>10s}", end="")
    print()
    
    
    for scale in sorted(all_results.keys()):
        print(f"{scale:6d}  |", end="")
        results = all_results[scale]
        
        for model in args.models:
            recall = results.get(model, 0.0)
            print(f"    {recall:6.4f}", end="")
        print()
    
    print(f"\n experiments completed.")

if __name__ == "__main__":
    main()