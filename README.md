# Coordination Detection

## Project Overview

This project detects coordinated groups of social media accounts from a single snapshot of Twitter data. Rather than relying on a single signal such as text similarity or graph clustering, the pipeline combines multiple independent behavioural signals to distinguish genuine coordinated campaigns from large organic crowds. The final output is a ranked `results.json` containing every detected community, its coordination score, and a binary coordination decision.

---

## Problem Statement

The objective is to identify coordinated behaviour, not bots, fake accounts, misinformation, or ordinary users discussing the same topic.

The assignment explicitly warns that simple approaches such as:

- embedding clustering,
- hashtag grouping,
- temporal clustering,
- or plain community detection,

primarily identify crowds rather than coordination. Therefore, the solution combines multiple behavioural signals before making a final decision.

---

## Pipeline Architecture

The complete pipeline consists of eight stages.

1. Load `posts.jsonl`, `accounts.jsonl` and `embeddings.parquet`.
2. Generate candidate post pairs using multiple behavioural interaction signals.
3. Compute pairwise behavioural features.
4. Fuse pair evidence into a coordination score.
5. Construct a weighted account graph.
6. Detect communities using the Leiden algorithm.
7. Compute final community coordination scores using both behavioural evidence and structural graph properties.
8. Produce `results.json`.

---

## Repository Structure

```
src/
└── coordination_detection/
    ├── main.py
    ├── __main__.py
    ├── loader.py
    ├── candidate_generation.py
    ├── temporal_features.py
    ├── interaction_features.py
    ├── semantic_features.py
    ├── rarity_features.py
    ├── repeated_behaviour.py
    ├── higher_order_features.py
    ├── pair_coordination_scoring.py
    ├── graph_builder.py
    ├── community_detection.py
    ├── community_scoring.py
    ├── organic_crowd_filter.py
    ├── output.py
    └── config.py

results.json
requirements.txt
Dockerfile
README.md
prompts.txt
```

---

## Installation

```bash
pip install -r requirements.txt
```

---

## Running Locally

```bash
python -m coordination_detection
```

The pipeline reads the configured input files and writes:

```
results.json
```

---

## Running with Docker

Build:

```bash
docker build -t coordination-detection .
```

Run:

```bash
docker run --rm coordination-detection
```

---

## Input Files

| File | Description |
|------|-------------|
| posts.jsonl | Social media posts |
| accounts.jsonl | Account metadata |
| embeddings.parquet | Multilingual post embeddings |

---

## Output

The pipeline produces

```json
{
  "clusters": [
    {
      "post_ids": ["..."],
      "is_coordinated": true,
      "coordination_score": 0.87
    }
  ]
}
```

Every detected community appears exactly once.

Communities are sorted by decreasing `coordination_score`.

---

# Methodology

## Candidate Generation

Rather than comparing every possible post pair, candidate pairs are generated using multiple independent interaction signals:

- shared thread
- shared reply target
- shared quoted post
- shared mentions
- rare shared URLs
- semantic nearest neighbours (FAISS)

Very common mention targets are ignored using an empirically selected frequency threshold to prevent candidate explosion.

---

## Feature Engineering

Each candidate pair receives features from multiple behavioural dimensions.

### Semantic

- cosine similarity
- cross-language similarity

### Temporal

- absolute time difference
- exponential temporal decay

### Interaction

- shared thread
- shared reply target
- shared quoted post
- mention overlap
- URL overlap
- hashtag overlap

### Rarity

Measures whether two accounts coordinate on uncommon interaction targets.

### Repeated Behaviour

Measures repeated interaction with common external targets across multiple posts.

### Higher-Order Features

Captures agreement between multiple independent behavioural signals rather than relying on any individual feature.

---

## Pair Coordination Scoring

Behavioural features are normalised and fused into a pair-level evidence score.

This stage intentionally combines multiple independent signals rather than allowing semantic similarity or graph proximity alone to dominate.

---

## Graph Construction

Strong candidate pairs become weighted graph edges.

Nodes represent accounts.

Edges represent behavioural coordination evidence.

Only sufficiently strong edges are retained to suppress graph noise.

---

## Community Detection

Leiden community detection partitions the weighted graph into candidate coordinated groups.

Leiden was chosen because it provides stable, well-connected communities while scaling efficiently to sparse graphs.

---

## Community Scoring

Each detected community is assigned an evidence score by aggregating:

- average pair coordination
- graph density
- average edge strength
- semantic agreement
- temporal agreement
- repeated behaviour
- interaction strength

Structural modifiers are then applied using:

- community size
- graph density
- average edge weight
- evidence diversity

The resulting value becomes the final `coordination_score`.

The binary decision is obtained directly from this final score:

```
is_coordinated =
coordination_score >= FINAL_COORDINATION_THRESHOLD
```

This ensures that ranking and classification remain consistent.

---

# Why This Approach?

Simple approaches were intentionally avoided.

Embedding clustering groups semantically similar discussions but cannot distinguish coordinated campaigns from ordinary conversations.

Similarly:

- hashtag clustering,
- temporal clustering,
- graph communities,
- or reply networks

identify crowds rather than coordinated behaviour.

Instead, this pipeline requires agreement across several independent behavioural signals before assigning a high coordination score.

---

# Time Complexity

Let

- N = posts
- C = candidate pairs
- E = graph edges
- V = graph nodes

| Stage | Complexity |
|---------|------------|
| Loading | O(N) |
| Candidate generation | O(N + Σk²) + FAISS search |
| Feature engineering | O(C) |
| Pair scoring | O(C) |
| Graph construction | O(C) |
| Community detection | Approximately O(E log V) |
| Community scoring | O(E + V) |
| Output | O(V) |

The dominant computational cost is candidate generation.

---

# Space Complexity

| Stage | Complexity |
|---------|------------|
| Posts | O(N) |
| Candidate pairs | O(C) |
| Feature tables | O(C) |
| Graph | O(E + V) |

Memory usage is dominated by candidate pairs.

---

# Scaling to 10–100× Larger Datasets

The current implementation is intended for a single snapshot of approximately 11k posts.

At significantly larger scales the primary bottleneck becomes candidate generation.

Future improvements would include:

- streaming candidate generation
- partitioning by interaction target
- batch feature computation
- incremental FAISS indexing
- DuckDB or Polars for large intermediate tables
- parallel feature extraction
- graph partitioning for community detection

The overall methodology would remain unchanged.

---

# Threshold Calibration Without Labels

No labelled coordination data was available.

Thresholds were therefore calibrated empirically using the development dataset.

The process consisted of:

1. analysing interaction-frequency distributions;
2. selecting a mention-frequency threshold to suppress candidate explosion;
3. comparing graph edge percentiles (90, 95, 97, 98 and 99);
4. inspecting community score, graph density and edge-weight distributions;
5. selecting conservative thresholds near the lower tail of plausible coordinated communities.

The final coordination threshold was selected only after structural penalties were incorporated into the community score so that ranking and binary classification remained consistent.

---

# Decision Log

### Candidate Explosion

Initial candidate generation produced an excessive number of mention-based pairs because a small number of popular mention targets generated almost complete graphs.

This was resolved by introducing mention-frequency thresholding.

---

### Temporal Features

Several handcrafted temporal features were initially considered.

Exploratory analysis showed most provided little additional information beyond absolute time difference and exponential decay.

The feature set was simplified accordingly.

---

### Community Scoring

The original implementation computed an evidence score and then independently applied hard filtering rules.

This created situations where highly ranked communities were still classified as non-coordinated.

The final implementation incorporates structural evidence directly into the final coordination score.

The binary coordination decision is obtained by thresholding that final score, ensuring consistency between ranking and classification.

---

### Semantic Candidate Generation

Semantic neighbours were retained during candidate generation to maximise recall.

However, semantic similarity alone is never sufficient.

Semantic candidates must accumulate supporting behavioural evidence throughout later stages before receiving high coordination scores.

---

# Current Limitations

- Thresholds remain empirically calibrated rather than learned.
- Candidate generation remains the largest computational bottleneck.
- Extremely sophisticated campaigns may evade behavioural detection.
- Organic communities sharing many behavioural characteristics may still occasionally receive high scores.

---

# Future Improvements

- Learn calibration from labelled campaigns.
- Adaptive threshold selection.
- Distributed graph processing.
- Streaming candidate generation.
- Better community-level explainability.
- Incremental updates for continuously arriving social media data.