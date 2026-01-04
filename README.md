# Movie Recommendation System using LightGCN

A Graph Neural Network (GNN)-based movie recommendation system implementing the LightGCN algorithm on the MovieLens 100K dataset.

## Overview

This project builds a recommendation system using **Light Graph Convolutional Networks (LightGCN)** - a state-of-the-art collaborative filtering approach that models user-movie interactions as a bipartite graph.

### How It Works

- **Nodes**: Users and movies form a bipartite graph
- **Edges**: User-movie interactions (ratings >= 4 are positive interactions)
- **Learning**: Message passing propagates information through 3 layers of neighborhood aggregation
- **Prediction**: Dot product similarity between user and movie embeddings

## Project Structure

```
MovieRecommendationLighGCN/
├── gnn_movie_recommender.py          # Main implementation
├── GNN_Movie_Recommender.ipynb       # Jupyter notebook version
├── claude-explanation.md             # Detailed conceptual explanations
├── documentation/
│   └── README.md                     # This file
├── embeddings_visualization.png      # t-SNE visualization output
├── training_history.png              # Training metrics plots
└── ml-100k/                          # MovieLens 100K dataset
    ├── u.data                        # 100,000 ratings
    ├── u.user                        # 943 users
    └── u.item                        # 1,682 movies
```

## Requirements

### Dependencies

- Python 3.7+
- PyTorch
- PyTorch Geometric
- pandas
- numpy
- scikit-learn
- matplotlib

### Installation

```bash
# Install dependencies
pip install torch torch-geometric pandas numpy scikit-learn matplotlib

# For GPU support with CUDA (optional)
pip install torch --index-url https://download.pytorch.org/whl/cu118
pip install torch-geometric
```

## Quick Start

### Running the Script

```bash
cd MovieRecommendationLighGCN
python gnn_movie_recommender.py
```

### Using the Jupyter Notebook

```bash
jupyter notebook GNN_Movie_Recommender.ipynb
```

### What Happens During Execution

1. **Data Loading** - Automatically downloads MovieLens 100K if not present
2. **Graph Construction** - Builds user-movie bipartite graph
3. **Training** - Runs 50 epochs with BPR loss
4. **Evaluation** - Computes Hit@K and NDCG@K metrics
5. **Visualization** - Generates embedding and training plots

## Model Architecture

### LightGCN

LightGCN simplifies traditional Graph Convolutional Networks by:

- **No feature transformation** - Pure neighborhood aggregation
- **No non-linear activation** - Linear propagation only
- **Layer combination** - Averages embeddings from all layers

```
final_embedding = (layer_0 + layer_1 + layer_2 + layer_3) / 4
```

### Key Components

| Component | Description |
|-----------|-------------|
| `MovieLensGraph` | Constructs bipartite graph from raw data |
| `LightGCNConv` | Single message passing layer with normalization |
| `LightGCN` | Complete model with embedding layers |
| `train_epoch()` | Training loop with BPR loss |
| `compute_metrics()` | Evaluation with Hit@K and NDCG@K |

## Configuration

All hyperparameters are configurable in the `main()` function:

```python
# Model Architecture
embedding_dim = 64          # Embedding dimensionality
num_layers = 3              # Number of GCN layers

# Training
num_epochs = 50             # Total training epochs
learning_rate = 0.001       # Adam optimizer learning rate
batch_size = 1024           # Training batch size
l2_weight = 0.001           # L2 regularization coefficient

# Data
rating_threshold = 4        # Minimum rating for positive interaction
test_ratio = 0.2            # Test set ratio

# Evaluation
top_k = 10                  # Number of recommendations
K_values = [5, 10, 20]      # K values for metrics
```

## Dataset

### MovieLens 100K

| Statistic | Value |
|-----------|-------|
| Users | 943 |
| Movies | 1,682 |
| Ratings | 100,000 |
| Rating Scale | 1-5 |
| Sparsity | 6.3% |

### Data Files

- **u.data**: Tab-separated ratings (user_id, movie_id, rating, timestamp)
- **u.user**: Pipe-separated user info (user_id, age, gender, occupation, zip)
- **u.item**: Pipe-separated movie info (movie_id, title, genres...)

## Performance

### Training Results

| Metric | Initial | Final |
|--------|---------|-------|
| BPR Loss | ~0.69 | ~0.20 |
| Hit@5 | ~45% | ~58% |
| Hit@10 | ~61% | ~71% |
| Hit@20 | ~73% | ~84% |
| NDCG@10 | ~0.15 | ~0.23 |

### Evaluation Metrics

- **Hit@K**: Fraction of users with at least one relevant item in top-K
- **NDCG@K**: Normalized Discounted Cumulative Gain (rewards higher-ranked relevant items)

## Usage Examples

### Generate Recommendations

```python
# After training the model
recommendations = recommend_movies(
    model=model,
    user_idx=0,
    graph=graph,
    edge_index=train_edge_index,
    top_k=10,
    device=device
)

for movie, score in recommendations:
    print(f"{movie}: {score:.4f}")
```

### Explain a Recommendation

```python
explain_recommendation(
    model=model,
    user_idx=0,
    movie_idx=5,
    graph=graph,
    edge_index=train_edge_index,
    device=device
)
```

## Visualizations

### Generated Outputs

1. **embeddings_visualization.png**
   - t-SNE projection of user and movie embeddings
   - Movies colored by genre

2. **training_history.png**
   - BPR loss over epochs
   - Hit@K metrics during training
   - NDCG@K metrics during training

## Algorithm Details

### Bipartite Graph Construction

- Users indexed from 0 to 942
- Movies indexed from 943 to 2624
- Bidirectional edges for message passing

### Message Passing

Each layer aggregates neighbor information:

```
embedding_i = sum(norm * embedding_j for j in neighbors)
norm = 1/sqrt(degree_i) * 1/sqrt(degree_j)
```

### BPR Loss

Bayesian Personalized Ranking optimizes:

```
L = -log(sigmoid(score_pos - score_neg)) + lambda * ||embeddings||^2
```

## Troubleshooting

### Common Issues

**CUDA Out of Memory**
```python
# Reduce batch size
batch_size = 512
```

**Dataset Not Found**
```python
# Script auto-downloads, or manually download from:
# https://grouplens.org/datasets/movielens/100k/
```

**PyTorch Geometric Installation Issues**
```bash
# Install with specific PyTorch version
pip install torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.0.0+cpu.html
pip install torch-geometric
```

## References

- [LightGCN Paper](https://arxiv.org/abs/2002.02126) - He et al., 2020
- [MovieLens Dataset](https://grouplens.org/datasets/movielens/)
- [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/)

## License

This project uses the MovieLens 100K dataset, which is provided by GroupLens Research for non-commercial use.
