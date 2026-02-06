#!/usr/bin/env python3

import pickle
import os
from collections import defaultdict

class DataConsistencyChecker:
    """Ensures data consistency between training and testing"""
    
    def __init__(self, data_dir="/data/prepared_dataset"):
        self.data_dir = data_dir
        self.train_data = None
        self.test_data = None
        self.consistency_report = {}
    
    def load_datasets(self):
        """Load training and test datasets"""
        train_path = os.path.join(self.data_dir, "pairs_option2_train.pkl")
        test_path = os.path.join(self.data_dir, "pairs_option2_test.pkl")
        
        with open(train_path, 'rb') as f:
            self.train_data = pickle.load(f)
        with open(test_path, 'rb') as f:
            self.test_data = pickle.load(f)
        
        print(f"📚 Loaded {len(self.train_data)} train records, {len(self.test_data)} test records")
    
    def extract_graph_ids(self):
        """Extract graph IDs from file paths"""
        def path_to_id(path):
            if not path:
                return None
            return os.path.basename(path).replace('.pkl', '')
        
        # Training IDs (original graphs available for training)
        train_original_ids = set()
        train_mapping = {}  # original_id -> train_record
        
        for record in self.train_data:
            if record['original_path']:
                original_id = path_to_id(record['original_path'])
                if original_id:
                    train_original_ids.add(original_id)
                    train_mapping[original_id] = record
        
        # Test IDs (enigma_2 graphs we want to recover)
        test_enigma2_ids = set()
        test_original_ids = set()
        test_mapping = {}  # original_id -> test_record
        
        for record in self.test_data:
            if record['original_path']:
                original_id = path_to_id(record['original_path'])
                if original_id:
                    test_original_ids.add(original_id)
                    test_enigma2_ids.add(original_id)  # Same ID, different variant
                    test_mapping[original_id] = record
        
        return {
            'train_original_ids': train_original_ids,
            'test_original_ids': test_original_ids,
            'test_enigma2_ids': test_enigma2_ids,
            'train_mapping': train_mapping,
            'test_mapping': test_mapping
        }
    
    def check_consistency(self):
        """Check if every test graph has its original in training"""
        if not self.train_data or not self.test_data:
            self.load_datasets()
        
        ids_info = self.extract_graph_ids()
        
        train_ids = ids_info['train_original_ids']
        test_ids = ids_info['test_enigma2_ids']
        
        # Find overlap and missing
        overlap = train_ids.intersection(test_ids)
        missing_from_train = test_ids - train_ids
        extra_in_train = train_ids - test_ids
        
        # Consistency report
        self.consistency_report = {
            'total_train_originals': len(train_ids),
            'total_test_enigma2': len(test_ids),
            'overlap_count': len(overlap),
            'missing_from_train': list(missing_from_train),
            'extra_in_train': list(extra_in_train),
            'overlap_ids': list(overlap),
            'coverage_percentage': len(overlap) / len(test_ids) * 100 if test_ids else 0
        }
        
        print(f"\n🔍 DATA CONSISTENCY ANALYSIS")
        print("=" * 50)
        print(f"📊 Training original graphs: {len(train_ids)}")
        print(f"📊 Test enigma_2 graphs: {len(test_ids)}")
        print(f"🔗 Overlap (recoverable): {len(overlap)} ({self.consistency_report['coverage_percentage']:.1f}%)")
        print(f"❌ Missing from training: {len(missing_from_train)}")
        print(f"➕ Extra in training: {len(extra_in_train)}")
        
        if missing_from_train:
            print(f"\n⚠️  CONSISTENCY ISSUE!")
            print(f"   {len(missing_from_train)} enigma_2 graphs cannot be recovered")
            print(f"   Their originals are not in the training set")
            if len(missing_from_train) <= 10:
                print(f"   Missing IDs: {missing_from_train}")
        
        return self.consistency_report
    
    def create_consistent_subset(self, num_graphs, seed=42):
        """Create a consistent subset for testing"""
        if not self.consistency_report:
            self.check_consistency()
        
        # Use only the overlapping IDs (graphs that exist in both train and test)
        overlap_ids = self.consistency_report['overlap_ids']
        
        if len(overlap_ids) < num_graphs:
            raise ValueError(f"Only {len(overlap_ids)} consistent graphs available, need {num_graphs}")
        
        # MAKE IT DETERMINISTIC
        import random
        random.seed(seed)  # Set seed for reproducibility
        overlap_ids = sorted(overlap_ids)  # Sort first to ensure consistent ordering
        selected_ids = overlap_ids[:num_graphs]  # Take first num_graphs (deterministic)
            
        ids_info = self.extract_graph_ids()
        train_mapping = ids_info['train_mapping']
        test_mapping = ids_info['test_mapping']
        
        # Create consistent subset
        consistent_train = []
        consistent_test = []
        gallery_items = []
        
        for graph_id in selected_ids:
            if graph_id in train_mapping and graph_id in test_mapping:
                train_record = train_mapping[graph_id]
                test_record = test_mapping[graph_id]
                
                consistent_train.append(train_record)
                consistent_test.append(test_record)
                # gallery_items.append({
                #     'id': graph_id,
                #     'path': test_record['original_path']  # Gallery uses original graphs
                # })
                # Fix the gallery item creation to match expected format
                gallery_items.append({
                    'id': f"{test_record['graph_type']}_{test_record['graph_id']}",  # Include type prefix
                    'path': test_record['original_path']
                })
        
        print(f"\n✅ Created consistent subset: {len(consistent_train)} graphs")
        print(f"   Training: original→enigma_1 pairs")
        print(f"   Testing: enigma_2 queries → original gallery")
        print(f"   Guarantee: Every test query has its answer in the gallery")
        
        return {
            'train_data': consistent_train,
            'test_data': consistent_test,
            'gallery_data': gallery_items,
            'selected_ids': selected_ids
        }
    
    def validate_subset(self, subset):
        """Validate that subset maintains consistency"""
        train_ids = set()
        test_ids = set()
        gallery_ids = set()
        
        for record in subset['train_data']:
            if record['original_path']:
                train_id = os.path.basename(record['original_path']).replace('.pkl', '')
                train_ids.add(train_id)
        
        for record in subset['test_data']:
            if record['original_path']:
                test_id = os.path.basename(record['original_path']).replace('.pkl', '')
                test_ids.add(test_id)
        
        for item in subset['gallery_data']:
            gallery_ids.add(item['id'])
        
        # Verify perfect alignment
        assert train_ids == test_ids == gallery_ids, "Subset IDs don't match perfectly!"
        
        print(f"✅ Subset validation passed: {len(train_ids)} graphs perfectly aligned")
        return True

def create_consistent_dataset(num_graphs, data_dir="/data/prepared_dataset", seed=42):
    """
    Create a dataset with recoverable test graph
    
    Args:
        num_graphs: Number of graphs to include
        data_dir: Directory containing the dataset
        seed: Random seed for reproducibility
    
    Returns:
        Dictionary with consistent train/test/gallery data
    """
    checker = DataConsistencyChecker(data_dir)
    subset = checker.create_consistent_subset(num_graphs, seed=seed)  # Pass seed through
    checker.validate_subset(subset)
    
    return subset

def main():
    """Main consistency checking function"""

    
    checker = DataConsistencyChecker()
    report = checker.check_consistency()
    
    print(f"\n SUMMARY:")
    print(f"   Coverage: {report['coverage_percentage']:.1f}%")
    print(f"   Recoverable graphs: {report['overlap_count']}")
    print(f"   Unrecoverable graphs: {len(report['missing_from_train'])}")
    
    if report['coverage_percentage'] < 100:
        print(f"\n  Warning: Not all test graphs are recoverable!")
        print(f"   Consider using create_consistent_subset() for fair evaluation")

if __name__ == "__main__":
    main()