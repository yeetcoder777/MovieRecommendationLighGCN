"""
================================================================================
GNN-BASED MOVIE RECOMMENDATION SYSTEM
================================================================================
This script implements a Graph Neural Network recommender using the MovieLens 
100K dataset. We'll build a bipartite graph where users and movies are nodes,
and ratings are edges.

Author: Claude (Anthropic)
Dataset: MovieLens 100K
Model: LightGCN (a simplified but powerful GNN for recommendations)
================================================================================
"""

# ==============================================================================
# PART 1: IMPORTS AND SETUP
# ==============================================================================
"""
We need several libraries:
- torch: The deep learning framework
- torch_geometric: For graph neural networks
- pandas/numpy: For data manipulation
- sklearn: For evaluation metrics and visualization
- matplotlib: For plotting
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import degree
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from collections import defaultdict
import urllib.request
import zipfile
import os
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# ==============================================================================
# PART 2: DOWNLOAD AND LOAD MOVIELENS 100K DATASET
# ==============================================================================
"""
WHAT IS MOVIELENS 100K?
-----------------------
- 100,000 ratings from 943 users on 1,682 movies
- Each user has rated at least 20 movies
- Ratings are on a scale of 1-5
- Contains user demographics (age, gender, occupation)
- Contains movie information (title, genres)

We'll download it and parse the relevant files.
"""

def download_movielens():
    """Download MovieLens 100K dataset if not already present"""
    
    data_dir = 'ml-100k'
    
    if not os.path.exists(data_dir):
        print("Downloading MovieLens 100K dataset...")
        url = 'https://files.grouplens.org/datasets/movielens/ml-100k.zip'
        
        urllib.request.urlretrieve(url, 'ml-100k.zip')
        
        with zipfile.ZipFile('ml-100k.zip', 'r') as zip_ref:
            zip_ref.extractall('.')
        
        os.remove('ml-100k.zip')
        print("Download complete!")
    else:
        print("Dataset already exists.")
    
    return data_dir


def load_movielens_data(data_dir):
    """
    Load and parse the MovieLens 100K data files.
    
    Returns:
        ratings_df: DataFrame with user_id, movie_id, rating, timestamp
        users_df: DataFrame with user demographics
        movies_df: DataFrame with movie information and genres
    """
    
    # Load ratings (user_id, movie_id, rating, timestamp)
    ratings_df = pd.read_csv(
        f'{data_dir}/u.data',
        sep='\t',
        names=['user_id', 'movie_id', 'rating', 'timestamp']
    )
    
    # Load user information (user_id, age, gender, occupation, zip_code)
    users_df = pd.read_csv(
        f'{data_dir}/u.user',
        sep='|',
        names=['user_id', 'age', 'gender', 'occupation', 'zip_code'],
        encoding='latin-1'
    )
    
    # Load movie information
    movies_df = pd.read_csv(
        f'{data_dir}/u.item',
        sep='|',
        names=['movie_id', 'title', 'release_date', 'video_release_date',
               'imdb_url', 'unknown', 'Action', 'Adventure', 'Animation',
               'Children', 'Comedy', 'Crime', 'Documentary', 'Drama',
               'Fantasy', 'Film-Noir', 'Horror', 'Musical', 'Mystery',
               'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western'],
        encoding='latin-1'
    )
    
    print(f"Loaded {len(ratings_df)} ratings")
    print(f"Number of users: {len(users_df)}")
    print(f"Number of movies: {len(movies_df)}")
    
    return ratings_df, users_df, movies_df


# ==============================================================================
# PART 3: CONSTRUCT THE BIPARTITE GRAPH
# ==============================================================================
"""
WHAT IS A BIPARTITE GRAPH?
--------------------------
A bipartite graph has two types of nodes (users and movies in our case),
and edges only connect nodes of DIFFERENT types.

    User_1 -------- Movie_A
         \        /
          \      /
           \    /
    User_2 -------- Movie_B

In this graph:
- Users connect to movies they've rated
- Users NEVER connect directly to other users
- Movies NEVER connect directly to other movies

WHY BIPARTITE FOR RECOMMENDATIONS?
----------------------------------
1. Natural representation: Users rate movies, not other users
2. Captures collaborative filtering: Similar users rate similar movies
3. Message passing enables:
   - User ← information from movies they rated
   - Movie ← information from users who rated it
   - After 2 hops: User ← info from other users with similar taste!
"""

class MovieLensGraph:
    """
    Constructs a bipartite graph from MovieLens data.
    
    Node indexing scheme:
    - Users: indices 0 to (num_users - 1)
    - Movies: indices num_users to (num_users + num_movies - 1)
    
    This allows us to have a single embedding matrix for all nodes.
    """
    
    def __init__(self, ratings_df, users_df, movies_df, rating_threshold=4):
        """
        Args:
            ratings_df: DataFrame with ratings
            users_df: DataFrame with user info
            movies_df: DataFrame with movie info
            rating_threshold: Consider ratings >= this as positive interactions
        """
        self.rating_threshold = rating_threshold
        
        # Create mappings from original IDs to consecutive indices
        self.user_ids = ratings_df['user_id'].unique()
        self.movie_ids = ratings_df['movie_id'].unique()
        
        self.user_to_idx = {uid: idx for idx, uid in enumerate(self.user_ids)}
        self.movie_to_idx = {mid: idx for idx, mid in enumerate(self.movie_ids)}
        self.idx_to_user = {idx: uid for uid, idx in self.user_to_idx.items()}
        self.idx_to_movie = {idx: mid for mid, idx in self.movie_to_idx.items()}
        
        self.num_users = len(self.user_ids)
        self.num_movies = len(self.movie_ids)
        self.num_nodes = self.num_users + self.num_movies
        
        print(f"\nGraph Statistics:")
        print(f"  Number of users: {self.num_users}")
        print(f"  Number of movies: {self.num_movies}")
        print(f"  Total nodes: {self.num_nodes}")
        
        # Store dataframes for later use
        self.users_df = users_df
        self.movies_df = movies_df
        self.ratings_df = ratings_df
        
        # Build edges and features
        self._build_edges(ratings_df)
        self._build_node_features(users_df, movies_df)
        
    def _build_edges(self, ratings_df):
        """
        Build edge index for the bipartite graph.
        
        For recommendation, we typically treat high ratings as "positive" edges.
        We create BIDIRECTIONAL edges (user→movie AND movie→user) because:
        - Information should flow both ways during message passing
        - User learns from movies, movie learns from users
        """
        # Filter to positive interactions (ratings >= threshold)
        positive_ratings = ratings_df[ratings_df['rating'] >= self.rating_threshold]
        
        print(f"  Positive interactions (rating >= {self.rating_threshold}): {len(positive_ratings)}")
        
        # Create edges: user → movie
        user_indices = [self.user_to_idx[uid] for uid in positive_ratings['user_id']]
        # Movie indices are offset by num_users
        movie_indices = [self.num_users + self.movie_to_idx[mid] 
                        for mid in positive_ratings['movie_id']]
        
        # Bidirectional edges
        src_nodes = user_indices + movie_indices  # user→movie, then movie→user
        dst_nodes = movie_indices + user_indices  # movie←user, then user←movie
        
        self.edge_index = torch.tensor([src_nodes, dst_nodes], dtype=torch.long)
        
        print(f"  Number of edges (bidirectional): {self.edge_index.shape[1]}")
        
        # Store user-movie interactions for train/test split
        self.user_movie_interactions = defaultdict(set)
        for _, row in positive_ratings.iterrows():
            user_idx = self.user_to_idx[row['user_id']]
            movie_idx = self.movie_to_idx[row['movie_id']]
            self.user_movie_interactions[user_idx].add(movie_idx)
    
    def _build_node_features(self, users_df, movies_df):
        """
        Build initial node features.
        
        For LightGCN, we don't actually use input features - it learns embeddings
        from scratch. But we'll create features anyway for visualization and
        potential use with other GNN architectures.
        
        User features: [age_normalized, gender_encoded, occupation_one_hot]
        Movie features: [genre_one_hot]
        """
        # === User Features ===
        # Normalize age to [0, 1]
        age_normalized = (users_df['age'] - users_df['age'].min()) / \
                        (users_df['age'].max() - users_df['age'].min())
        
        # Gender: M=1, F=0
        gender_encoded = (users_df['gender'] == 'M').astype(float)
        
        # Occupation one-hot (21 occupations)
        occupation_dummies = pd.get_dummies(users_df['occupation'])
        
        # Combine user features
        user_features = np.column_stack([
            age_normalized.values,
            gender_encoded.values,
            occupation_dummies.values
        ])
        
        # === Movie Features ===
        # Genre columns are already one-hot encoded
        genre_columns = ['unknown', 'Action', 'Adventure', 'Animation', 'Children',
                        'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy',
                        'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance',
                        'Sci-Fi', 'Thriller', 'War', 'Western']
        
        movie_features = movies_df[genre_columns].values
        
        # Store dimensions
        self.user_feature_dim = user_features.shape[1]
        self.movie_feature_dim = movie_features.shape[1]
        
        print(f"  User feature dimension: {self.user_feature_dim}")
        print(f"  Movie feature dimension: {self.movie_feature_dim}")
        
        # For LightGCN, we pad features to the same dimension
        # (not actually used in forward pass, but needed for Data object)
        max_dim = max(self.user_feature_dim, self.movie_feature_dim)
        
        user_features_padded = np.zeros((self.num_users, max_dim))
        user_features_padded[:, :self.user_feature_dim] = user_features[:self.num_users]
        
        movie_features_padded = np.zeros((self.num_movies, max_dim))
        movie_features_padded[:, :self.movie_feature_dim] = movie_features[:self.num_movies]
        
        # Combine: users first, then movies
        all_features = np.vstack([user_features_padded, movie_features_padded])
        self.node_features = torch.tensor(all_features, dtype=torch.float)
        
        # Store genre names for later
        self.genre_columns = genre_columns
        
    def get_pyg_data(self):
        """Return PyTorch Geometric Data object"""
        return Data(
            x=self.node_features,
            edge_index=self.edge_index,
            num_nodes=self.num_nodes
        )
    
    def train_test_split(self, test_ratio=0.2):
        """
        Split interactions into train and test sets.

        FIXED: Now creates separate train_user_items dict
        so evaluation doesn't accidentally mask test items.
        """
        train_edges = []
        test_edges = []

        for user_idx, movie_indices in self.user_movie_interactions.items():
            movies = list(movie_indices)
            
            if len(movies) < 2:
                # If user has only 1 interaction, keep it in training
                train_edges.extend([(user_idx, m) for m in movies])
            else:
                # Split user's movies into train and test
                n_test = max(1, int(len(movies) * test_ratio))
                np.random.shuffle(movies)
                
                test_movies = movies[:n_test]
                train_movies = movies[n_test:]
                
                train_edges.extend([(user_idx, m) for m in train_movies])
                test_edges.extend([(user_idx, m) for m in test_movies])

        # Convert to tensors (add num_users offset to movie indices)
        train_src = [e[0] for e in train_edges]
        train_dst = [e[1] + self.num_users for e in train_edges]

        test_src = [e[0] for e in test_edges]
        test_dst = [e[1] + self.num_users for e in test_edges]

        # Bidirectional
        self.train_edge_index = torch.tensor([
            train_src + train_dst,
            train_dst + train_src
        ], dtype=torch.long)

        self.test_edges = list(zip(test_src, test_dst))

        # === NEW: Store ONLY training items per user ===
        self.train_user_items = defaultdict(set)
        for user_idx, movie_idx in train_edges:
            self.train_user_items[user_idx].add(movie_idx)

        print(f"\nTrain/Test Split:")
        print(f"  Training edges: {len(train_edges)} (bidirectional: {self.train_edge_index.shape[1]})")
        print(f"  Test edges: {len(test_edges)}")

        return self.train_edge_index, self.test_edges


# ==============================================================================
# PART 4: LIGHTGCN MODEL
# ==============================================================================
"""
WHAT IS LIGHTGCN?
-----------------
LightGCN is a simplified Graph Convolutional Network designed specifically
for collaborative filtering (recommendations). It was published in 2020 and
achieved state-of-the-art results while being simpler than previous GNN
recommenders.

KEY SIMPLIFICATIONS:
1. NO feature transformation (no weight matrices in convolution)
2. NO non-linear activation functions
3. ONLY neighborhood aggregation (weighted sum of neighbors)

WHY DOES SIMPLER WORK BETTER?
-----------------------------
In collaborative filtering, we don't have rich node features - we're learning
embeddings from scratch. The complex transformations in standard GCNs can
cause overfitting and make training harder. LightGCN removes this complexity.

THE LIGHTGCN FORMULA:
--------------------
For each layer:
    e_u^(k+1) = Σ (1/√|N_u| × 1/√|N_i|) × e_i^(k)    (for user u)
    e_i^(k+1) = Σ (1/√|N_i| × 1/√|N_u|) × e_u^(k)    (for item i)

Final embedding = average of all layer embeddings:
    e_u = (1/(K+1)) × (e_u^(0) + e_u^(1) + ... + e_u^(K))

WHERE:
- e_u^(k) = user u's embedding at layer k
- N_u = neighbors of user u (movies they rated)
- The 1/√|N| terms normalize by node degree (prevents high-degree nodes
  from dominating)
"""

class LightGCNConv(MessagePassing):
    """
    Single LightGCN convolution layer.
    
    This implements the message passing:
    - Each node receives messages from its neighbors
    - Messages are the neighbor embeddings, normalized by degrees
    - No transformation, no activation - just aggregation!
    """
    
    def __init__(self):
        # aggr='add' means we sum up all messages from neighbors
        super().__init__(aggr='add')
        
    def forward(self, x, edge_index):
        """
        Perform one layer of LightGCN convolution.
        
        Args:
            x: Node embeddings [num_nodes, embedding_dim]
            edge_index: Edge connectivity [2, num_edges]
            
        Returns:
            Updated node embeddings [num_nodes, embedding_dim]
        """
        # Calculate normalization coefficients
        # For edge (i, j): norm = 1 / sqrt(degree_i) / sqrt(degree_j)
        row, col = edge_index
        deg = degree(col, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
        
        # Start message passing
        return self.propagate(edge_index, x=x, norm=norm)
    
    def message(self, x_j, norm):
        """
        Construct messages from source nodes.
        
        This is called for each edge. x_j is the source node embedding,
        and we multiply by the normalization factor.
        
        In plain English: "The message from neighbor j is their embedding,
        scaled by how popular they are (normalization)"
        """
        return norm.view(-1, 1) * x_j


class LightGCN(nn.Module):
    """
    Complete LightGCN model for collaborative filtering.
    
    Architecture:
    1. Embedding layer for users and items (learned from scratch)
    2. K LightGCN convolution layers (neighborhood aggregation)
    3. Final embedding = mean of all layer embeddings
    
    The model learns to place similar users/items close in embedding space.
    """
    
    def __init__(self, num_users, num_items, embedding_dim=64, num_layers=100):
        """
        Args:
            num_users: Number of users in the graph
            num_items: Number of items (movies) in the graph
            embedding_dim: Dimension of learned embeddings
            num_layers: Number of GCN layers (K)
        """
        super().__init__()
        
        self.num_users = num_users
        self.num_items = num_items
        self.num_layers = num_layers
        
        # Initialize embeddings (this is what we learn!)
        # Xavier initialization helps training stability
        self.embedding = nn.Embedding(num_users + num_items, embedding_dim)
        nn.init.xavier_uniform_(self.embedding.weight)
        
        # LightGCN convolution layers (they share the same operation)
        self.convs = nn.ModuleList([LightGCNConv() for _ in range(num_layers)])
        
    def forward(self, edge_index):
        """
        Forward pass: Compute final embeddings for all nodes.
        
        Process:
        1. Get initial embeddings (layer 0)
        2. Apply K convolution layers, collecting outputs
        3. Average all layer outputs for final embedding
        
        Why average all layers?
        - Layer 0: Original embedding (no neighbor info)
        - Layer 1: Info from 1-hop neighbors
        - Layer 2: Info from 2-hop neighbors
        - ...
        Averaging captures both local and global structure!
        """
        # Get initial embeddings
        x = self.embedding.weight
        all_embeddings = [x]
        
        # Apply convolution layers
        for conv in self.convs:
            x = conv(x, edge_index)
            all_embeddings.append(x)
        
        # Stack and average across layers
        all_embeddings = torch.stack(all_embeddings, dim=1)  # [N, K+1, D]
        final_embeddings = all_embeddings.mean(dim=1)        # [N, D]
        
        return final_embeddings
    
    def get_user_item_embeddings(self, edge_index):
        """
        Get separate user and item embeddings.
        
        Returns:
            user_emb: [num_users, embedding_dim]
            item_emb: [num_items, embedding_dim]
        """
        embeddings = self.forward(edge_index)
        user_emb = embeddings[:self.num_users]
        item_emb = embeddings[self.num_users:]
        return user_emb, item_emb
    
    def predict(self, user_indices, item_indices, edge_index):
        """
        Predict scores for user-item pairs.
        
        Score = dot product of user and item embeddings.
        Higher score = model thinks user will like this item more.
        
        Args:
            user_indices: Tensor of user indices
            item_indices: Tensor of item indices (NOT offset by num_users)
            edge_index: Graph connectivity for message passing
            
        Returns:
            scores: Predicted preference scores
        """
        user_emb, item_emb = self.get_user_item_embeddings(edge_index)
        
        user_vectors = user_emb[user_indices]
        item_vectors = item_emb[item_indices]
        
        # Dot product as similarity score
        scores = (user_vectors * item_vectors).sum(dim=1)
        return scores


# ==============================================================================
# PART 5: TRAINING WITH BPR LOSS
# ==============================================================================
"""
WHAT IS BPR LOSS?
-----------------
BPR = Bayesian Personalized Ranking

The idea: For a user, items they've interacted with should be ranked HIGHER
than items they haven't interacted with.

For each training sample, we create a triplet:
    (user, positive_item, negative_item)
    
Where:
- positive_item = something the user actually liked
- negative_item = something the user didn't interact with (assumed not liked)

The loss pushes:
    score(user, positive) > score(user, negative)

Formula:
    BPR_loss = -log(sigmoid(score_pos - score_neg))

This is essentially saying: "Maximize the probability that positive items
are ranked higher than negative items."

WHY NOT JUST USE MSE ON RATINGS?
--------------------------------
1. We care about RANKING, not exact rating prediction
2. BPR directly optimizes what we want: correct ordering
3. Works well with implicit feedback (clicks, views) where we don't have ratings
"""

class BPRLoss(nn.Module):
    """Bayesian Personalized Ranking Loss"""
    
    def __init__(self):
        super().__init__()
        
    def forward(self, pos_scores, neg_scores):
        """
        Args:
            pos_scores: Predicted scores for positive (user, item) pairs
            neg_scores: Predicted scores for negative (user, item) pairs
            
        Returns:
            BPR loss value
        """
        # We want pos_scores > neg_scores
        # Loss = -log(sigmoid(pos - neg))
        return -torch.mean(F.logsigmoid(pos_scores - neg_scores))


def sample_negative_items(user_indices, num_items, user_movie_interactions, num_neg=1):
    """
    Sample negative items for each user.
    
    A negative item is one the user HASN'T interacted with.
    We assume non-interaction means the user doesn't want it
    (not always true, but a common assumption in recommender systems).
    
    Args:
        user_indices: Users in current batch
        num_items: Total number of items
        user_movie_interactions: Dict mapping user_idx → set of movie_idx
        num_neg: Number of negative samples per positive
        
    Returns:
        negative_items: Tensor of negative item indices
    """
    negative_items = []
    
    for user_idx in user_indices.cpu().numpy():
        user_positives = user_movie_interactions.get(user_idx, set())
        
        # Sample items until we get one that's negative
        for _ in range(num_neg):
            while True:
                neg_item = np.random.randint(0, num_items)
                if neg_item not in user_positives:
                    negative_items.append(neg_item)
                    break
    
    return torch.tensor(negative_items, dtype=torch.long)


def train_epoch(model, optimizer, train_edge_index, graph, device, batch_size=1024):
    """
    Train for one epoch.
    
    Process:
    1. Sample batch of positive edges (user, movie they liked)
    2. Sample negative items for each user
    3. Compute scores for positive and negative pairs
    4. Compute BPR loss
    5. Backpropagate and update embeddings
    """
    model.train()
    
    # Get positive edges (user → movie direction only)
    pos_edges = train_edge_index[:, train_edge_index[0] < graph.num_users]
    num_pos_edges = pos_edges.shape[1]
    
    # Shuffle edges
    perm = torch.randperm(num_pos_edges)
    pos_edges = pos_edges[:, perm]
    
    total_loss = 0
    num_batches = 0
    
    criterion = BPRLoss()
    
    for start in range(0, num_pos_edges, batch_size):
        end = min(start + batch_size, num_pos_edges)
        batch_edges = pos_edges[:, start:end]
        
        # User indices and positive item indices
        user_indices = batch_edges[0]
        pos_item_indices = batch_edges[1] - graph.num_users  # Remove offset
        
        # Sample negative items
        neg_item_indices = sample_negative_items(
            user_indices, 
            graph.num_movies,
            graph.user_movie_interactions
        ).to(device)
        
        # Move to device
        user_indices = user_indices.to(device)
        pos_item_indices = pos_item_indices.to(device)
        
        # Compute scores
        pos_scores = model.predict(user_indices, pos_item_indices, train_edge_index.to(device))
        neg_scores = model.predict(user_indices, neg_item_indices, train_edge_index.to(device))
        
        # Compute loss
        loss = criterion(pos_scores, neg_scores)
        
        # Add L2 regularization on embeddings
        reg_loss = 0.001 * (
            model.embedding.weight[user_indices].norm(2).pow(2) +
            model.embedding.weight[graph.num_users + pos_item_indices].norm(2).pow(2) +
            model.embedding.weight[graph.num_users + neg_item_indices].norm(2).pow(2)
        ) / len(user_indices)
        
        total_loss_batch = loss + reg_loss
        
        # Backprop
        optimizer.zero_grad()
        total_loss_batch.backward()
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
    
    return total_loss / num_batches


# ==============================================================================
# PART 6: EVALUATION METRICS
# ==============================================================================
"""
HOW DO WE EVALUATE A RECOMMENDER SYSTEM?
----------------------------------------

We can't just use accuracy because:
1. We're ranking items, not classifying them
2. Position matters: A good item at position 1 is better than at position 100

Common metrics:

1. HIT@K (Hit Rate at K):
   - For each user, recommend top K items
   - If ANY of the test items appear in top K, it's a "hit"
   - Hit@10 = What fraction of users got at least one relevant item in top 10?

2. NDCG@K (Normalized Discounted Cumulative Gain):
   - Rewards relevant items, but MORE for items ranked higher
   - A relevant item at position 1 contributes more than at position 10
   - NDCG@K ranges from 0 to 1 (1 = perfect ranking)

Formula for DCG@K:
    DCG@K = Σ (relevance_i / log2(i + 1))  for i in 1..K
    
    Position 1 has weight 1/log2(2) = 1.0
    Position 2 has weight 1/log2(3) = 0.63
    Position 10 has weight 1/log2(11) = 0.29

NDCG = DCG / Ideal_DCG (normalized to [0, 1])
"""

def compute_metrics(model, edge_index, graph, test_edges, K_values=[5, 10, 20], device='cpu'):
    """
    Compute Hit@K and NDCG@K metrics.
    
    FIXED: Now masks only TRAINING items, not all items.
    """
    model.eval()
    
    with torch.no_grad():
        # Get all embeddings
        user_emb, item_emb = model.get_user_item_embeddings(edge_index.to(device))
        user_emb = user_emb.cpu()
        item_emb = item_emb.cpu()
    
    # Group test edges by user
    user_test_items = defaultdict(set)
    for user_idx, movie_node_idx in test_edges:
        movie_idx = movie_node_idx - graph.num_users
        user_test_items[user_idx].add(movie_idx)
    
    # Initialize metrics
    hits = {k: [] for k in K_values}
    ndcgs = {k: [] for k in K_values}
    
    for user_idx, test_items in user_test_items.items():
        # Get user embedding
        user_vec = user_emb[user_idx].unsqueeze(0)  # [1, D]
        
        # Compute scores for ALL items
        scores = torch.mm(user_vec, item_emb.t()).squeeze()  # [num_items]
        
        # === FIXED: Mask only TRAINING items, not all items ===
        train_items = graph.train_user_items.get(user_idx, set())
        for item_idx in train_items:
            scores[item_idx] = float('-inf')
        
        # Get top-K items
        max_k = max(K_values)
        _, top_k_indices = torch.topk(scores, max_k)
        top_k_indices = top_k_indices.numpy()
        
        # Compute metrics for each K
        for k in K_values:
            top_k = set(top_k_indices[:k])
            
            # Hit@K: Did we recommend ANY test item?
            hit = len(top_k & test_items) > 0
            hits[k].append(1.0 if hit else 0.0)
            
            # NDCG@K
            dcg = 0.0
            for i, item_idx in enumerate(top_k_indices[:k]):
                if item_idx in test_items:
                    dcg += 1.0 / np.log2(i + 2)  # +2 because positions are 1-indexed
            
            # Ideal DCG: all test items at the top
            idcg = sum(1.0 / np.log2(i + 2) for i in range(min(len(test_items), k)))
            
            ndcg = dcg / idcg if idcg > 0 else 0.0
            ndcgs[k].append(ndcg)
    
    # Average across users
    results = {}
    for k in K_values:
        results[f'Hit@{k}'] = np.mean(hits[k])
        results[f'NDCG@{k}'] = np.mean(ndcgs[k])
    
    return results

# ==============================================================================
# PART 7: RECOMMENDATION GENERATION
# ==============================================================================
"""
HOW ARE RECOMMENDATIONS GENERATED?
----------------------------------

After training, generating recommendations is simple:

1. Forward pass through GNN to get all embeddings
2. For target user, get their embedding vector
3. Compute similarity (dot product) with ALL movie embeddings
4. Exclude movies the user has already interacted with
5. Return top-K movies by similarity score

The magic is in the EMBEDDINGS:
- Similar users end up with similar embeddings (they liked similar movies)
- Similar movies end up with similar embeddings (liked by similar users)
- A user's embedding naturally points toward movies they would like!
"""

def recommend_movies(model, user_idx, graph, edge_index, top_k=10, device='cpu'):
    """
    Generate movie recommendations for a user.
    
    Args:
        model: Trained LightGCN model
        user_idx: Index of the user (in our 0-indexed scheme)
        graph: MovieLensGraph object
        edge_index: Graph edges for message passing
        top_k: Number of recommendations to return
        
    Returns:
        List of (movie_title, score) tuples
    """
    model.eval()
    
    with torch.no_grad():
        user_emb, item_emb = model.get_user_item_embeddings(edge_index.to(device))
        user_emb = user_emb.cpu()
        item_emb = item_emb.cpu()
    
    # Get this user's embedding
    user_vec = user_emb[user_idx].unsqueeze(0)
    
    # Compute scores for all movies
    scores = torch.mm(user_vec, item_emb.t()).squeeze()
    
    # Mask out movies the user has already seen
    seen_movies = graph.user_movie_interactions.get(user_idx, set())
    for movie_idx in seen_movies:
        scores[movie_idx] = float('-inf')
    
    # Get top-K
    top_scores, top_indices = torch.topk(scores, top_k)
    
    # Map back to movie titles
    recommendations = []
    for idx, score in zip(top_indices.numpy(), top_scores.numpy()):
        original_movie_id = graph.idx_to_movie[idx]
        movie_info = graph.movies_df[graph.movies_df['movie_id'] == original_movie_id].iloc[0]
        recommendations.append((movie_info['title'], score))
    
    return recommendations


def explain_recommendation(model, user_idx, movie_idx, graph, edge_index, device='cpu'):
    """
    Explain WHY a movie was recommended.
    
    This looks at:
    1. What genres is this movie?
    2. What other movies has this user liked in similar genres?
    3. What similar users liked this movie?
    """
    model.eval()
    
    # Get movie info
    original_movie_id = graph.idx_to_movie[movie_idx]
    movie_info = graph.movies_df[graph.movies_df['movie_id'] == original_movie_id].iloc[0]
    
    # Get genres
    genres = [g for g in graph.genre_columns if movie_info[g] == 1]
    
    # Get user's watched movies
    seen_movies = graph.user_movie_interactions.get(user_idx, set())
    
    # Find similar movies user has watched
    similar_watched = []
    for seen_idx in list(seen_movies)[:10]:  # Check first 10
        seen_movie_id = graph.idx_to_movie[seen_idx]
        seen_info = graph.movies_df[graph.movies_df['movie_id'] == seen_movie_id].iloc[0]
        seen_genres = [g for g in graph.genre_columns if seen_info[g] == 1]
        
        overlap = set(genres) & set(seen_genres)
        if overlap:
            similar_watched.append((seen_info['title'], list(overlap)))
    
    explanation = {
        'movie': movie_info['title'],
        'genres': genres,
        'similar_watched': similar_watched[:3]
    }
    
    return explanation


# ==============================================================================
# PART 8: VISUALIZATION
# ==============================================================================
"""
VISUALIZING LEARNED EMBEDDINGS
------------------------------

A great way to understand what the model learned is to visualize the
embeddings in 2D using t-SNE (t-distributed Stochastic Neighbor Embedding).

If the model learned well:
- Movies of the same genre should cluster together
- Users with similar tastes should cluster together
- The user-movie space should show meaningful structure
"""

def visualize_embeddings(model, graph, edge_index, device='cpu', sample_size=500):
    """
    Visualize user and movie embeddings using t-SNE.
    """
    model.eval()
    
    with torch.no_grad():
        user_emb, item_emb = model.get_user_item_embeddings(edge_index.to(device))
        user_emb = user_emb.cpu().numpy()
        item_emb = item_emb.cpu().numpy()
    
    # Sample for visualization (t-SNE is slow on large datasets)
    n_users = min(sample_size, len(user_emb))
    n_movies = min(sample_size, len(item_emb))
    
    user_sample_idx = np.random.choice(len(user_emb), n_users, replace=False)
    movie_sample_idx = np.random.choice(len(item_emb), n_movies, replace=False)
    
    user_sample = user_emb[user_sample_idx]
    movie_sample = item_emb[movie_sample_idx]
    
    # Combine for t-SNE
    combined = np.vstack([user_sample, movie_sample])
    
    # Apply t-SNE
    print("Running t-SNE... (this may take a minute)")
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    embeddings_2d = tsne.fit_transform(combined)
    
    user_2d = embeddings_2d[:n_users]
    movie_2d = embeddings_2d[n_users:]
    
    # Get movie genres for coloring
    movie_colors = []
    genre_priority = ['Action', 'Comedy', 'Drama', 'Sci-Fi', 'Romance', 'Horror', 'Documentary']
    
    for idx in movie_sample_idx:
        movie_id = graph.idx_to_movie[idx]
        movie_info = graph.movies_df[graph.movies_df['movie_id'] == movie_id].iloc[0]
        
        color = 'gray'
        for i, genre in enumerate(genre_priority):
            if movie_info.get(genre, 0) == 1:
                color = plt.cm.tab10(i)
                break
        movie_colors.append(color)
    
    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot 1: Users and Movies together
    ax1 = axes[0]
    ax1.scatter(user_2d[:, 0], user_2d[:, 1], c='blue', alpha=0.5, s=20, label='Users')
    ax1.scatter(movie_2d[:, 0], movie_2d[:, 1], c='red', alpha=0.5, s=20, label='Movies')
    ax1.set_title('User and Movie Embeddings (t-SNE)')
    ax1.legend()
    ax1.set_xlabel('Dimension 1')
    ax1.set_ylabel('Dimension 2')
    
    # Plot 2: Movies colored by genre
    ax2 = axes[1]
    scatter = ax2.scatter(movie_2d[:, 0], movie_2d[:, 1], c=movie_colors, alpha=0.6, s=30)
    ax2.set_title('Movie Embeddings by Genre')
    ax2.set_xlabel('Dimension 1')
    ax2.set_ylabel('Dimension 2')
    
    # Create legend for genres
    legend_elements = [plt.Line2D([0], [0], marker='o', color='w', 
                                   markerfacecolor=plt.cm.tab10(i), markersize=8, label=genre)
                      for i, genre in enumerate(genre_priority)]
    ax2.legend(handles=legend_elements, loc='upper right', fontsize=8)
    
    plt.tight_layout()
    plt.savefig('embeddings_visualization.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("Saved visualization to 'embeddings_visualization.png'")


def plot_training_history(train_losses, hit_history, ndcg_history, eval_every=5):
    """
    Plot training loss and evaluation metrics over epochs.
    
    FIXED: Correctly computes epoch numbers to match history length.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # Plot 1: Loss
    axes[0].plot(train_losses, 'b-', linewidth=2)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('BPR Loss')
    axes[0].set_title('Training Loss')
    axes[0].grid(True, alpha=0.3)
    
    # === FIXED: Compute epochs based on actual history length ===
    num_evals = len(hit_history[list(hit_history.keys())[0]])
    epochs = list(range(eval_every, eval_every * num_evals + 1, eval_every))
    
    # Plot 2: Hit@K
    for k, values in hit_history.items():
        axes[1].plot(epochs, values, '-o', label=f'Hit@{k}', markersize=4)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Hit Rate')
    axes[1].set_title('Hit@K over Training')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # Plot 3: NDCG@K
    for k, values in ndcg_history.items():
        axes[2].plot(epochs, values, '-o', label=f'NDCG@{k}', markersize=4)
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('NDCG')
    axes[2].set_title('NDCG@K over Training')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('training_history.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("Saved training history to 'training_history.png'")

# ==============================================================================
# PART 9: MAIN EXECUTION
# ==============================================================================

def main():
    """
    Main function to run the entire pipeline.
    """
    print("=" * 70)
    print("GNN-BASED MOVIE RECOMMENDATION SYSTEM")
    print("=" * 70)
    
    # ===== STEP 1: Load Data =====
    print("\n" + "=" * 70)
    print("STEP 1: LOADING MOVIELENS 100K DATASET")
    print("=" * 70)
    
    data_dir = download_movielens()
    ratings_df, users_df, movies_df = load_movielens_data(data_dir)
    
    # ===== STEP 2: Build Graph =====
    print("\n" + "=" * 70)
    print("STEP 2: CONSTRUCTING BIPARTITE GRAPH")
    print("=" * 70)
    
    graph = MovieLensGraph(ratings_df, users_df, movies_df, rating_threshold=4)
    
    # ===== STEP 3: Train/Test Split =====
    print("\n" + "=" * 70)
    print("STEP 3: SPLITTING DATA INTO TRAIN/TEST")
    print("=" * 70)
    
    train_edge_index, test_edges = graph.train_test_split(test_ratio=0.2)
    
    # ===== STEP 4: Initialize Model =====
    print("\n" + "=" * 70)
    print("STEP 4: INITIALIZING LIGHTGCN MODEL")
    print("=" * 70)
    
    model = LightGCN(
        num_users=graph.num_users,
        num_items=graph.num_movies,
        embedding_dim=64,
        num_layers=3
    ).to(device)
    
    print(f"Model architecture:")
    print(f"  - Embedding dimension: 64")
    print(f"  - Number of GCN layers: 3")
    print(f"  - Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # ===== STEP 5: Training Loop =====
    print("\n" + "=" * 70)
    print("STEP 5: TRAINING THE MODEL")
    print("=" * 70)
    
    num_epochs = 50
    eval_every = 5  # Evaluate every 5 epochs
    
    train_losses = []
    hit_history = {5: [], 10: [], 20: []}
    ndcg_history = {5: [], 10: [], 20: []}
    
    print(f"\nTraining for {num_epochs} epochs...")
    print("-" * 50)
    
    for epoch in range(num_epochs):
        # Train
        loss = train_epoch(model, optimizer, train_edge_index, graph, device)
        train_losses.append(loss)
        
        # Evaluate every eval_every epochs
        if (epoch + 1) % eval_every == 0:
            metrics = compute_metrics(
                model, train_edge_index, graph, test_edges,
                K_values=[5, 10, 20], device=device
            )
            
            for k in [5, 10, 20]:
                hit_history[k].append(metrics[f'Hit@{k}'])
                ndcg_history[k].append(metrics[f'NDCG@{k}'])
            
            print(f"Epoch {epoch+1:3d} | Loss: {loss:.4f} | "
                  f"Hit@10: {metrics['Hit@10']:.4f} | NDCG@10: {metrics['NDCG@10']:.4f}")
    
    # ===== STEP 6: Final Evaluation =====
    print("\n" + "=" * 70)
    print("STEP 6: FINAL EVALUATION")
    print("=" * 70)
    
    final_metrics = compute_metrics(
        model, train_edge_index, graph, test_edges,
        K_values=[5, 10, 20], device=device
    )
    
    print("\nFinal Test Metrics:")
    print("-" * 30)
    for metric, value in final_metrics.items():
        print(f"  {metric}: {value:.4f}")
    
    # ===== STEP 7: Generate Recommendations =====
    print("\n" + "=" * 70)
    print("STEP 7: GENERATING SAMPLE RECOMMENDATIONS")
    print("=" * 70)
    
    # Pick a random user
    sample_user_idx = np.random.randint(0, graph.num_users)
    original_user_id = graph.idx_to_user[sample_user_idx]
    
    # Get user info
    user_info = users_df[users_df['user_id'] == original_user_id].iloc[0]
    print(f"\nUser Profile:")
    print(f"  - User ID: {original_user_id}")
    print(f"  - Age: {user_info['age']}")
    print(f"  - Gender: {user_info['gender']}")
    print(f"  - Occupation: {user_info['occupation']}")
    
    # Show what the user has watched
    watched_movies = list(graph.user_movie_interactions[sample_user_idx])[:5]
    print(f"\nMovies this user liked (sample):")
    for movie_idx in watched_movies:
        movie_id = graph.idx_to_movie[movie_idx]
        movie_title = movies_df[movies_df['movie_id'] == movie_id].iloc[0]['title']
        print(f"  - {movie_title}")
    
    # Generate recommendations
    recommendations = recommend_movies(
        model, sample_user_idx, graph, train_edge_index, top_k=10, device=device
    )
    
    print(f"\nTop 10 Recommended Movies:")
    print("-" * 50)
    for i, (title, score) in enumerate(recommendations, 1):
        print(f"  {i:2d}. {title} (score: {score:.3f})")
    
    # Explain first recommendation
    top_movie_idx = list(graph.movie_to_idx.values())[0]
    for movie_idx in graph.movie_to_idx.values():
        movie_id = graph.idx_to_movie[movie_idx]
        if movies_df[movies_df['movie_id'] == movie_id].iloc[0]['title'] == recommendations[0][0]:
            top_movie_idx = movie_idx
            break
    
    explanation = explain_recommendation(
        model, sample_user_idx, top_movie_idx, graph, train_edge_index, device
    )
    
    print(f"\nWhy '{explanation['movie']}' was recommended:")
    print(f"  Genres: {', '.join(explanation['genres'])}")
    if explanation['similar_watched']:
        print(f"  Similar to movies you liked:")
        for title, genres in explanation['similar_watched']:
            print(f"    - {title} ({', '.join(genres)})")
    
    # ===== STEP 8: Visualizations =====
    print("\n" + "=" * 70)
    print("STEP 8: CREATING VISUALIZATIONS")
    print("=" * 70)
    
    # Plot training history
    plot_training_history(train_losses, hit_history, ndcg_history)
    
    # Visualize embeddings
    visualize_embeddings(model, graph, train_edge_index, device, sample_size=300)
    
    print("\n" + "=" * 70)
    print("COMPLETE!")
    print("=" * 70)
    
    return model, graph, train_edge_index


# ==============================================================================
# PART 10: HOW MESSAGE PASSING CAPTURES INFLUENCE (EXPLANATION)
# ==============================================================================
"""
================================================================================
DEEP DIVE: HOW MESSAGE PASSING CAPTURES PEER AND NEIGHBORHOOD INFLUENCE
================================================================================

Let's trace through exactly what happens during message passing in our
recommendation system. This is the KEY insight for understanding GNN recommenders.

INITIAL STATE (Layer 0):
------------------------
Every user and movie starts with a random embedding vector.
At this point, embeddings carry NO information about relationships.

    User_1: [0.2, -0.1, 0.5, ...]  (random)
    User_2: [0.8, 0.3, -0.2, ...]  (random)
    Movie_A: [-0.4, 0.6, 0.1, ...] (random)
    Movie_B: [0.1, -0.5, 0.9, ...] (random)


AFTER LAYER 1 (1-hop neighborhood):
-----------------------------------
Each node receives messages from its DIRECT neighbors.

For User_1 who watched Movie_A and Movie_B:
    User_1_new = normalize(Movie_A_emb + Movie_B_emb)
    
What does this mean?
    User_1's embedding now ENCODES their taste!
    It's literally a combination of the movies they liked.

For Movie_A which was watched by User_1 and User_3:
    Movie_A_new = normalize(User_1_emb + User_3_emb)
    
What does this mean?
    Movie_A's embedding encodes WHO likes it.
    Similar movies (liked by similar users) will have similar embeddings.


AFTER LAYER 2 (2-hop neighborhood):
-----------------------------------
Now things get interesting! Let's trace User_1's embedding:

    User_1 ← Movie_A ← User_3
                      └── User_3 also watched Movie_C, Movie_D
    User_1 ← Movie_B ← User_2
                      └── User_2 also watched Movie_E

User_1's embedding now contains information about:
    - Movies they watched (1-hop)
    - OTHER USERS who watched those movies (2-hop)
    
This is COLLABORATIVE FILTERING emerging naturally!
User_1's embedding is influenced by User_2 and User_3's preferences
BECAUSE they share movie interests.


AFTER LAYER 3 (3-hop neighborhood):
-----------------------------------
    User_1 ← Movie_A ← User_3 ← Movie_C ← User_5
                                         └── User_5 also watched Movie_X

Now User_1 is influenced by:
    - Their watched movies
    - Users who watched same movies (peers)
    - Movies those peers watched (peer preferences)
    - Users who watched those movies (peers of peers!)

The embedding captures an entire COMMUNITY of similar users.


WHY THIS WORKS FOR RECOMMENDATIONS:
-----------------------------------
After message passing:

1. Users who liked similar movies → similar embeddings
   (they received similar messages from their movie neighbors)

2. Movies liked by similar users → similar embeddings
   (they received similar messages from their user neighbors)

3. A user's embedding naturally "points toward" movies they'd like
   because it's built from movies they liked + movies similar users liked

When we compute:
    score(user, movie) = dot_product(user_emb, movie_emb)
    
High scores mean: "This movie's embedding aligns with this user's taste embedding"


CONCRETE EXAMPLE:
-----------------
Scenario: User_A likes [Toy Story, Finding Nemo, The Lion King]
          User_B likes [Toy Story, Finding Nemo, Shrek]
          
After message passing:
- User_A and User_B will have SIMILAR embeddings (shared movies)
- "Monsters Inc" (similar animated movie) will have embedding close to these
- When recommending to User_A, Monsters Inc will score high because:
  - Its embedding came from users who like animated movies
  - User_A's embedding represents "animated movie lover"
  - Dot product = high!

This is the power of GNN recommenders: collaborative filtering emerges
AUTOMATICALLY from the graph structure, without explicit "user similarity"
calculations!
"""


# ==============================================================================
# RUN THE SYSTEM
# ==============================================================================

if __name__ == "__main__":
    model, graph, train_edge_index = main()
    
    # Interactive recommendations
    print("\n" + "=" * 70)
    print("INTERACTIVE MODE")
    print("=" * 70)
    print("You can generate recommendations for any user!")
    print("Example: recommend_movies(model, user_idx=0, graph=graph, ")
    print("         edge_index=train_edge_index, top_k=10, device=device)")
