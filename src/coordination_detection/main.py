"""Command-line entry point for the coordination detection pipeline."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import networkx as nx
import pandas as pd

if __package__ in (None, ""):
    import sys

    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from coordination_detection.candidate_generation import CandidateGenerator
    from coordination_detection.config import CandidateGenerationConfig
    from coordination_detection import config as pipeline_config
    from coordination_detection.community_detection import CommunityDetector, CommunityDetectionConfig
    from coordination_detection.community_scoring import CommunityScorer, CommunityScoringConfig
    from coordination_detection.graph_builder import GraphBuilder, GraphBuilderConfig
    from coordination_detection.organic_crowd_filter import filter_organic_crowds
    from coordination_detection.output import write_results_json
    from coordination_detection.pair_coordination_scoring import PairCoordinationScorer
    from coordination_detection.interaction_features import InteractionFeatureExtractor
    from coordination_detection.loader import load_accounts, load_posts
    from coordination_detection.higher_order_features import HigherOrderFeatureExtractor
    from coordination_detection.repeated_behavior_features import RepeatedBehaviorFeatureExtractor
    from coordination_detection.rarity_features import RarityFeatureExtractor
    from coordination_detection.semantic_features import SemanticFeatureExtractor
    from coordination_detection.temporal_features import TemporalFeatureExtractor
else:
    from .candidate_generation import CandidateGenerator
    from .config import CandidateGenerationConfig
    from . import config as pipeline_config
    from .community_detection import CommunityDetector, CommunityDetectionConfig
    from .community_scoring import CommunityScorer, CommunityScoringConfig
    from .graph_builder import GraphBuilder, GraphBuilderConfig
    from .organic_crowd_filter import filter_organic_crowds
    from .output import write_results_json
    from .pair_coordination_scoring import PairCoordinationScorer
    from .higher_order_features import HigherOrderFeatureExtractor
    from .interaction_features import InteractionFeatureExtractor
    from .loader import load_accounts, load_posts
    from .repeated_behavior_features import RepeatedBehaviorFeatureExtractor
    from .rarity_features import RarityFeatureExtractor
    from .semantic_features import SemanticFeatureExtractor
    from .temporal_features import TemporalFeatureExtractor

LOGGER = logging.getLogger(__name__)


def build_candidate_pairs(
    posts_path: str | Path | None = None,
    accounts_path: str | Path | None = None,
    embeddings_path: str | Path | None = None,
    config: CandidateGenerationConfig | None = None,
) -> tuple[pd.DataFrame, dict[str, Any], dict[str, Any]]:
    """Load inputs and generate the deduplicated candidate-pair table."""

    posts_file = pipeline_config.resolve_project_path(posts_path or pipeline_config.POSTS_INPUT_FILE)
    accounts_file = pipeline_config.resolve_project_path(accounts_path or pipeline_config.ACCOUNTS_INPUT_FILE)
    embeddings_file = pipeline_config.resolve_project_path(embeddings_path or pipeline_config.EMBEDDINGS_INPUT_FILE)

    accounts = load_accounts(str(accounts_file))
    LOGGER.info("Loaded %d accounts from %s", len(accounts), accounts_file)

    posts = load_posts(str(posts_file))
    generator = CandidateGenerator(config)
    candidates = generator.generate_candidates(posts, embeddings_path=embeddings_file)
    return candidates, generator.get_statistics(), generator.get_diagnostics()


def build_temporal_features(
    candidate_pairs: pd.DataFrame,
    posts_path: str | Path | None = None,
) -> pd.DataFrame:
    """Load posts and compute temporal evidence for candidate pairs."""

    posts_file = pipeline_config.resolve_project_path(posts_path or pipeline_config.POSTS_INPUT_FILE)
    posts = load_posts(str(posts_file))
    extractor = TemporalFeatureExtractor()
    return extractor.transform(candidate_pairs, posts)


def build_semantic_features(
    candidate_pairs: pd.DataFrame,
    embeddings_path: str | Path | None = None,
) -> pd.DataFrame:
    """Load posts and embeddings, then compute pairwise semantic evidence."""

    embeddings_file = pipeline_config.resolve_project_path(embeddings_path or pipeline_config.EMBEDDINGS_INPUT_FILE)
    embeddings = pd.read_parquet(embeddings_file)
    extractor = SemanticFeatureExtractor()
    return extractor.transform(candidate_pairs, embeddings)


def build_interaction_features(
    candidate_pairs: pd.DataFrame,
    posts_path: str | Path | None = None,
) -> pd.DataFrame:
    """Load posts and compute raw interaction evidence for candidate pairs."""

    posts_file = pipeline_config.resolve_project_path(posts_path or pipeline_config.POSTS_INPUT_FILE)
    posts = load_posts(str(posts_file))
    extractor = InteractionFeatureExtractor()
    return extractor.transform(candidate_pairs, posts)


def build_rarity_features(
    candidate_pairs: pd.DataFrame,
    posts_path: str | Path | None = None,
) -> pd.DataFrame:
    """Load posts and compute rarity evidence for candidate pairs."""

    posts_file = pipeline_config.resolve_project_path(posts_path or pipeline_config.POSTS_INPUT_FILE)
    posts = load_posts(str(posts_file))
    extractor = RarityFeatureExtractor()
    return extractor.transform(candidate_pairs, posts)


def build_repeated_behavior_features(
    candidate_pairs: pd.DataFrame,
    posts_path: str | Path | None = None,
) -> pd.DataFrame:
    """Load posts and compute repeated-behaviour evidence for candidate pairs."""

    posts_file = pipeline_config.resolve_project_path(posts_path or pipeline_config.POSTS_INPUT_FILE)
    posts = load_posts(str(posts_file))
    extractor = RepeatedBehaviorFeatureExtractor()
    return extractor.transform(candidate_pairs, posts)


def build_higher_order_features(
    temporal_features: pd.DataFrame,
    interaction_features: pd.DataFrame,
    semantic_features: pd.DataFrame,
    repeated_behavior_features: pd.DataFrame,
) -> pd.DataFrame:
    """Combine previously computed evidence into higher-order behavioural features."""

    extractor = HigherOrderFeatureExtractor()
    return extractor.transform(
        temporal_features,
        interaction_features,
        semantic_features,
        repeated_behavior_features,
    )


def build_pair_coordination_score(
    temporal_features: pd.DataFrame,
    interaction_features: pd.DataFrame,
    semantic_features: pd.DataFrame,
    rarity_features: pd.DataFrame,
    repeated_behavior_features: pd.DataFrame,
    higher_order_features: pd.DataFrame,
) -> pd.DataFrame:
    """Fuse evidence tables into a pair-level coordination score."""

    scorer = PairCoordinationScorer()
    return scorer.transform(
        temporal_features,
        interaction_features,
        semantic_features,
        rarity_features,
        repeated_behavior_features,
        higher_order_features,
    )


def build_account_graph(
    candidate_pairs: pd.DataFrame,
    coordination_scores: pd.DataFrame,
    *,
    edge_percentile: float | None = None,
    edge_threshold: float | None = None,
) -> tuple[Any, pd.DataFrame]:
    """Build a weighted account graph from scored post pairs."""

    base_config = GraphBuilderConfig()
    config = GraphBuilderConfig(
        edge_percentile=edge_percentile if edge_percentile is not None else base_config.edge_percentile,
        edge_threshold=edge_threshold if edge_threshold is not None else base_config.edge_threshold,
    )
    builder = GraphBuilder(config)
    return builder.transform(candidate_pairs, coordination_scores)


def build_communities(graph: Any, config: CommunityDetectionConfig | None = None) -> tuple[pd.DataFrame, dict[int, list[Any]]]:
    """Run Leiden community detection on a weighted account graph."""

    detector = CommunityDetector(config)
    return detector.transform(graph)


def build_community_scores(
    graph: Any,
    community_memberships: pd.DataFrame | dict[int, list[Any]],
    coordination_scores: pd.DataFrame,
    higher_order_features: pd.DataFrame,
    semantic_features: pd.DataFrame | None = None,
    candidate_pairs: pd.DataFrame | None = None,
    config: CommunityScoringConfig | None = None,
) -> pd.DataFrame:
    """Score detected communities using the existing pairwise evidence."""

    scorer = CommunityScorer(config)
    return scorer.transform(
        graph,
        community_memberships,
        coordination_scores,
        higher_order_features,
        semantic_features=semantic_features,
        candidate_pairs=candidate_pairs,
    )


def _log_graph_diagnostics(graph: Any, memberships: pd.DataFrame) -> None:
    """Log graph-shape diagnostics for quick test verification."""

    LOGGER.info("type(graph): %s", type(graph))
    LOGGER.info("isinstance(graph, nx.Graph): %s", isinstance(graph, nx.Graph))
    LOGGER.info("isinstance(graph, nx.MultiGraph): %s", isinstance(graph, nx.MultiGraph))

    if memberships.empty or not {"community_id", "account_id"}.issubset(memberships.columns):
        return

    for community_id, group in memberships.groupby("community_id", sort=True):
        if len(group) != 2:
            continue
        community_nodes = group["account_id"].dropna().tolist()
        subgraph = graph.subgraph(community_nodes)
        LOGGER.info("first size-2 community: %s", community_id)
        LOGGER.info("subgraph.nodes: %s", list(subgraph.nodes()))
        LOGGER.info("subgraph.edges: %s", list(subgraph.edges(data=True)))
        LOGGER.info("subgraph.number_of_nodes: %d", subgraph.number_of_nodes())
        LOGGER.info("subgraph.number_of_edges: %d", subgraph.number_of_edges())
        LOGGER.info("nx.density(subgraph): %.6f", nx.density(subgraph))
        break


def main() -> None:
    """Run candidate generation and feature extraction on repo data."""

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    root = pipeline_config.PROJECT_ROOT
    posts_path = pipeline_config.resolve_project_path(pipeline_config.POSTS_INPUT_FILE)
    candidates, _statistics, _diagnostics = build_candidate_pairs()
    posts = load_posts(str(posts_path))
    interaction_features = build_interaction_features(candidates)
    rarity_features = build_rarity_features(candidates)
    repeated_behavior_features = build_repeated_behavior_features(candidates)
    semantic_features = build_semantic_features(candidates)
    temporal_features = build_temporal_features(candidates)
    higher_order_features = build_higher_order_features(
        temporal_features,
        interaction_features,
        semantic_features,
        repeated_behavior_features,
    )
    coordination_scores = build_pair_coordination_score(
        temporal_features,
        interaction_features,
        semantic_features,
        rarity_features,
        repeated_behavior_features,
        higher_order_features,
    )
    graph_thresholds = (95.0,)
    analysis_dir = root / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    for percentile in graph_thresholds:
        graph, edge_df = build_account_graph(
            candidates,
            coordination_scores,
            edge_percentile=percentile,
        )
        membership_frame, community_map = build_communities(graph)
        _log_graph_diagnostics(graph, membership_frame)
        community_scores = build_community_scores(
            graph,
            membership_frame,
            coordination_scores,
            higher_order_features,
            semantic_features,
            candidates,
        )
        filtered_communities = filter_organic_crowds(community_scores, membership_frame)
        output_path = analysis_dir / f"community_scores_{percentile:.0f}.csv"
        community_scores.to_csv(output_path, index=False)
        LOGGER.info("Saved community scores to %s", output_path)
        write_results_json(
            filtered_communities,
            membership_frame,
            posts,
            output_path=root / pipeline_config.RESULTS_OUTPUT_FILE,
        )
        LOGGER.info(
            "Graph percentile %.0f: nodes=%d, edges=%d, unique_edges=%d",
            percentile,
            graph.number_of_nodes(),
            graph.number_of_edges(),
            len(edge_df),
        )


if __name__ == "__main__":
    main()
