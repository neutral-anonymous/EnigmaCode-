# QUBO Obfuscation Evaluation - Final Experiments

This package contains the code for reproducing the final experiments evaluating feature-based and structure-only baselines for QUBO de-anonymization.

## Methods Implemented

### Feature-Based Baselines (6 methods)

1. **Direct cosine similarity** (`DirectFeatureMatcher`)
2. **k-NN (features)** (`SpectralKNNMatcher`)
3. **Random forest** (`RandomForestMatcher`)
4. **Siamese neural network** (`SiameseNetworkMatcher`)
5. **Contrastive-learning model** (`MetricLearningMatcher`)
6. **Feature-weighted similarity** (`QUBOMetricLearning`)

### Structure-Only Baseline (1 method)

7. **Seedless FGW matcher** (`fixed_test_li4.py`)

## Evaluation Metric

- **Top-5 accuracy**: Ranking-based measure (Recall@5)

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```bash
cd graph_analysis
python simple_fixed_runner.py --scales 5 10 100 --models direct_features spectral_knn random_forest siamese_network metric_learning
```
