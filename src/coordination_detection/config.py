"""Configuration for the coordination detection pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
POSTS_INPUT_FILE = "posts.jsonl"
ACCOUNTS_INPUT_FILE = "accounts.jsonl"
EMBEDDINGS_INPUT_FILE = "embeddings.parquet"

INTERACTION_ORDER: tuple[str, ...] = (
    "thread_id",
    "reply_to_post_id",
    "quoted_post_id",
    "mentions",
    "rare_urls",
)

RARE_URL_MAX_POST_COUNT = 2
MENTION_MAX_POST_COUNT = 75
SEMANTIC_TOP_K = 10
MIN_SEMANTIC_SIMILARITY = 0.85
TEMPORAL_DECAY_SECONDS = 1800
MENTION_THRESHOLD_SWEEP: tuple[int, ...] = (10, 20, 30, 40, 50, 75, 100, 150, 200, 300, 400)
ANALYSIS_OUTPUT_DIR = "analysis"
RESULTS_OUTPUT_FILE = "results.json"
MIN_COMMUNITY_SIZE = 3
MIN_GRAPH_DENSITY = 0.36
MIN_AVERAGE_EDGE_WEIGHT = 0.25
MIN_COMMUNITY_COORDINATION_SCORE = 0.28
DOMINANT_SIGNAL_THRESHOLD = 0.70
# Final community confidence is the evidence score after structural penalties.
FINAL_COORDINATION_THRESHOLD = 0.15
COORDINATION_SCORE_RECOMMENDATION_QUANTILES: tuple[float, ...] = (0.90, 0.95, 0.99)


def resolve_project_path(path: str | Path) -> Path:
    """Resolve a configured path relative to the repository root."""

    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return PROJECT_ROOT / candidate


@dataclass(frozen=True, slots=True)
class CandidateGenerationConfig:
    """Configuration for candidate pair generation."""

    embeddings_path: str = EMBEDDINGS_INPUT_FILE
    post_id_column: str = "post_id"
    account_id_column: str = "account_id"
    thread_id_column: str = "thread_id"
    reply_to_post_id_column: str = "reply_to_post_id"
    quoted_post_id_column: str = "quoted_post_id"
    mentions_column: str = "mentions"
    urls_column: str = "urls"
    mention_max_post_count: int = MENTION_MAX_POST_COUNT
    rare_url_max_post_count: int = RARE_URL_MAX_POST_COUNT
    semantic_top_k: int = SEMANTIC_TOP_K
    min_semantic_similarity: float = MIN_SEMANTIC_SIMILARITY
    diagnostic_top_n_keys: int = 20
    interaction_order: tuple[str, ...] = INTERACTION_ORDER


@dataclass(frozen=True, slots=True)
class MentionThresholdAnalysisConfig:
    """Configuration for the mention-threshold analysis workflow."""

    output_dir: str = ANALYSIS_OUTPUT_DIR
    mention_thresholds: tuple[int, ...] = MENTION_THRESHOLD_SWEEP
    mention_column: str = "mentions"
    post_id_column: str = "post_id"


@dataclass(frozen=True, slots=True)
class PairCoordinationScoringConfig:
    """Configuration for pair-level coordination score fusion."""

    candidate_post_id_1_column: str = "post_id_1"
    candidate_post_id_2_column: str = "post_id_2"
    temporal_decay_column: str = "exponential_decay_score"
    semantic_similarity_column: str = "cosine_similarity"
    same_thread_column: str = "same_thread"
    same_reply_target_column: str = "same_reply_target"
    same_quoted_post_column: str = "same_quoted_post"
    mention_jaccard_similarity_column: str = "mention_jaccard_similarity"
    url_jaccard_similarity_column: str = "url_jaccard_similarity"
    hashtag_jaccard_similarity_column: str = "hashtag_jaccard_similarity"
    mention_rarity_column: str = "mention_rarity"
    url_rarity_column: str = "url_rarity"
    hashtag_rarity_column: str = "hashtag_rarity"
    thread_rarity_column: str = "thread_rarity"
    reply_target_rarity_column: str = "reply_target_rarity"
    repeated_coactivity_count_column: str = "repeated_coactivity_count"
    shared_target_count_column: str = "shared_target_count"
    semantic_temporal_agreement_column: str = "semantic_temporal_agreement"
    thread_temporal_agreement_column: str = "thread_temporal_agreement"
    reply_semantic_agreement_column: str = "reply_semantic_agreement"
    interaction_strength_column: str = "interaction_strength"
    repeated_target_agreement_column: str = "repeated_target_agreement"
    temporal_weight: float = 1.0
    semantic_weight: float = 1.0
    interaction_weight: float = 1.0
    rarity_weight: float = 1.0
    repeated_weight: float = 1.0
    semantic_temporal_weight: float = 1.0
    thread_temporal_weight: float = 1.0
    reply_semantic_weight: float = 1.0
    repeated_target_weight: float = 1.0
    recommendation_quantiles: tuple[float, ...] = COORDINATION_SCORE_RECOMMENDATION_QUANTILES
