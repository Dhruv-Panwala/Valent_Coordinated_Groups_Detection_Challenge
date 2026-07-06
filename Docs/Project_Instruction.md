# Project Instructions

Goal: Detect coordinated groups of accounts from a single snapshot of social media data.

This is NOT:
- Bot detection
- Fake account detection
- Misinformation detection
- Supervised learning

Detect coordination, not crowds.

Important:
- Do NOT rely on DBSCAN, HDBSCAN, KMeans, embedding clustering, hashtag grouping, cosine thresholding or plain community detection.
- Community detection is only a downstream step after building a high-confidence coordination graph.
- Coordination must be inferred from multiple independent behavioural signals (evidence fusion), never a single feature.

Implementation Rules:
- Python 3.11+
- Modular code with single responsibility per module.
- Type hints and docstrings.
- Logging instead of print().
- No duplicated code.
- No notebooks.
- No hardcoded thresholds or weights; keep all configurable values in config.py.
- Keep functions reusable and independently testable.

Data Handling:
- Load posts.jsonl, accounts.jsonl and embeddings.parquet.
- Safely handle nested dictionaries.
- Do not call drop_duplicates() on dict/list columns.
- Validate required columns.
- Handle missing values gracefully.
- Convert timestamps to datetime.

Feature Engineering:
- Build multiple coordination signals (semantic, temporal, interactions, URLs, hashtags, mentions, replies, threads, quotes, repeated behaviour).
- Prefer higher-order combinations of signals over single features.
- Make adding new signals easy.

Graph Pipeline:
- Compute pairwise coordination scores.
- Remove weak edges.
- Build weighted graph from strong evidence only.
- Run community detection on filtered graph.
- Score communities and filter likely organic crowds.

Performance:
- Avoid unnecessary O(N²).
- Use vectorised operations where possible.
- Prefer scalable implementations.

General:
- Only implement the requested module.
- Do not modify unrelated files.
- Reuse existing utilities.
- Keep code clean, maintainable and production-quality.