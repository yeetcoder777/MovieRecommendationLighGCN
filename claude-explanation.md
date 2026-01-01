# 🎬 GNN Movie Recommender - Explained Simply

Let me break this down into digestible pieces using analogies.

---

## The Big Picture Analogy: **A Party Where People Make Friends**

Imagine a party where:
- **Users** = Party guests
- **Movies** = Topics of conversation
- **Edges** = "Person X talked about Topic Y"

**Goal**: Recommend new conversation topics to each guest based on what similar guests enjoy talking about.

---

## Part 1: Loading the Data

**Analogy**: Getting the guest list and conversation logs

```python
# Download party records
ratings_df  → "Guest X talked about Topic Y with enthusiasm level Z"
users_df    → Guest profiles (age, gender, job)
movies_df   → Topic details (genres like Action, Comedy, etc.)
```

**What we have**:
- 943 guests (users)
- 1,682 topics (movies)  
- 100,000 conversations (ratings)

---

## Part 2: Building the Graph

**Analogy**: Drawing a map of who talked about what

```
    Guest_1 -------- Topic_A (Toy Story)
         \        /
          \      /
           \    /
    Guest_2 -------- Topic_B (Star Wars)
```

**Key code concept**:
```python
# Only connect if they REALLY liked it (rating ≥ 4)
positive_ratings = ratings[ratings['rating'] >= 4]

# Bidirectional: Guest↔Topic (info flows both ways)
edges = user→movie + movie→user
```

**Why bidirectional?** 
- Guest learns from topics they discussed
- Topic learns from guests who discussed it

---

## Part 3: The LightGCN Model

**Analogy**: The "Gossip Network"

Imagine each person starts with a **name tag** (embedding) with random words. Then:

### Layer 1: Direct Gossip
```
"Hey, what topics do YOU like?"

Guest_1's new tag = average of (Topic_A's tag + Topic_B's tag)
Topic_A's new tag = average of (Guest_1's tag + Guest_3's tag)
```

### Layer 2: Friend-of-Friend Gossip
```
Guest_1 → Topic_A → Guest_3 → Topic_C

Now Guest_1 learns about Topic_C through Guest_3!
```

### Layer 3: Even Wider Network
```
Information spreads 3 hops away
```

**The Magic Formula** (simplified):
```python
my_new_embedding = average(all my neighbors' embeddings)
```

**Code**:
```python
class LightGCNConv(MessagePassing):
    def message(self, x_j, norm):
        # Message = neighbor's embedding × normalization
        return norm * x_j
```

---

## Part 4: Training with BPR Loss

**Analogy**: Teaching by Comparison

Instead of saying "Rate this 1-5", we ask:

> "Which would Guest_1 prefer: Topic_A (which they liked) or Topic_X (random)?"

**The rule**: Liked topics should score HIGHER than random topics.

```python
# Training triplet
(Guest_1, Toy Story ✓, Random Movie ✗)

# Loss pushes:
score(Guest_1, Toy Story) > score(Guest_1, Random Movie)
```

**Code**:
```python
def bpr_loss(pos_scores, neg_scores):
    # "Make positive scores bigger than negative scores"
    return -log(sigmoid(pos_scores - neg_scores))
```

---

## Part 5: Making Recommendations

**Analogy**: Matchmaking

After training, each guest and topic has a refined name tag (embedding).

```python
# Find compatibility
score = dot_product(guest_embedding, topic_embedding)

# High score = "These two would get along!"
# Low score = "Probably not a match"
```

**Recommendation process**:
```python
1. Get Guest_1's embedding
2. Compare with ALL topic embeddings
3. Remove topics they already discussed
4. Return top 10 matches
```

---

## Part 6: Evaluation

**Analogy**: Testing the Matchmaker

We hide some conversations and see if we can predict them.

| Metric | Question |
|--------|----------|
| **Hit@10** | "Did ANY hidden topic appear in your top 10 guesses?" |
| **NDCG@10** | "How high did you rank the hidden topics?" |

---

## The Complete Flow (Visual)

```
┌─────────────────────────────────────────────────────────┐
│  1. LOAD DATA                                           │
│     "Get guest list & conversation logs"                │
└──────────────────────┬──────────────────────────────────┘
                       ▼
┌─────────────────────────────────────────────────────────┐
│  2. BUILD GRAPH                                         │
│     "Draw map: Guest ←→ Topic connections"              │
└──────────────────────┬──────────────────────────────────┘
                       ▼
┌─────────────────────────────────────────────────────────┐
│  3. MESSAGE PASSING (LightGCN)                          │
│     Layer 1: Learn from direct connections              │
│     Layer 2: Learn from friends-of-friends              │
│     Layer 3: Learn from wider community                 │
│     Final: Average all layers                           │
└──────────────────────┬──────────────────────────────────┘
                       ▼
┌─────────────────────────────────────────────────────────┐
│  4. TRAIN (BPR Loss)                                    │
│     "Liked items should rank higher than random ones"   │
└──────────────────────┬──────────────────────────────────┘
                       ▼
┌─────────────────────────────────────────────────────────┐
│  5. RECOMMEND                                           │
│     score = dot(user_emb, movie_emb)                    │
│     Return top-K highest scores                         │
└─────────────────────────────────────────────────────────┘
```

---

## Why GNNs Beat Traditional Methods

| Traditional | GNN |
|-------------|-----|
| "Find users who rated same movies" | Graph structure captures this automatically |
| Explicit similarity calculation | Similarity emerges from message passing |
| Limited to direct connections | Multi-hop captures community patterns |

---

## Minimal Working Code (50 lines)

Want to see the core without all the extras? Here's the essence:

```python
# 1. Graph: users 0→942, movies 943→2624
edges = torch.tensor([[user_ids], [movie_ids + num_users]])

# 2. Model: just embeddings + aggregation
class LightGCN(nn.Module):
    def __init__(self):
        self.emb = nn.Embedding(num_users + num_movies, 64)
    
    def forward(self, edges):
        x = self.emb.weight
        for layer in range(3):
            x = aggregate_neighbors(x, edges)  # The magic!
        return x

# 3. Train: positive should beat negative
loss = -log(sigmoid(score_pos - score_neg))

# 4. Recommend: highest dot products
scores = user_emb @ all_movie_embs.T
top_10 = scores.topk(10)
```

---

Does this help clarify the concepts? Would you like me to dive deeper into any specific part?