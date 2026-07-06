"""Score detected communities using weighted coordination evidence."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from time import perf_counter
from typing import Any

import networkx as nx
import numpy as np
import pandas as pd

if __package__ in (None, ""):
    import sys
    from pathlib import Path

    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from coordination_detection import config as pipeline_config
else:
    from . import config as pipeline_config

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class CommunityScoringConfig:
    """Configuration for community-level coordination scoring."""

    community_id_column: str = "community_id"
    account_id_column: str = "account_id"
    account_id_1_column: str = "account_id_1"
    account_id_2_column: str = "account_id_2"
    graph_weight_column: str = "weight"
    graph_support_column: str = "support"
    coordination_score_column: str = "coordination_score"
    semantic_similarity_column: str = "mean_semantic_similarity"
    temporal_agreement_column: str = "mean_temporal_agreement"
    interaction_strength_column: str = "mean_interaction_strength"
    repeated_target_agreement_column: str = "mean_repeated_target_agreement"
    community_size_weight: float = 1.0
    edge_count_weight: float = 1.0
    graph_density_weight: float = 1.0
    average_edge_weight_weight: float = 1.0
    average_coordination_score_weight: float = 1.0
    semantic_similarity_weight: float = 1.0
    temporal_agreement_weight: float = 1.0
    interaction_strength_weight: float = 1.0
    repeated_target_agreement_weight: float = 1.0
    evidence_score_column: str = "evidence_score"
    final_coordination_score_column: str = "community_coordination_score"


class CommunityScorer:
    """Score each detected community using weighted behavioural evidence."""

    def __init__(self, config: CommunityScoringConfig | None = None) -> None:
        self.config = config or CommunityScoringConfig()

    def transform(
        self,
        graph: nx.Graph,
        community_memberships: pd.DataFrame | dict[int, list[Any]],
        coordination_scores: pd.DataFrame,
        higher_order_features: pd.DataFrame,
        semantic_features: pd.DataFrame | None = None,
        candidate_pairs: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        """Return one scored row per detected community."""

        start = perf_counter()
        memberships = self._normalize_memberships(community_memberships)
        if memberships.empty:
            LOGGER.info("Number of communities: 0")
            LOGGER.info("Average community size: 0.00")
            LOGGER.info("Runtime: %.2fs", perf_counter() - start)
            return self._empty_output()

        node_to_community = memberships.set_index(self.config.account_id_column)[self.config.community_id_column].to_dict()
        community_sizes = (
            memberships.groupby(self.config.community_id_column, sort=True)[self.config.account_id_column]
            .nunique()
            .rename("community_size")
            .reset_index()
        )
        community_ids = pd.Index(sorted(community_sizes[self.config.community_id_column].unique()), name=self.config.community_id_column)

        graph_edges = self._graph_edge_frame(graph)
        scored_edges = self._attach_graph_communities(graph_edges, node_to_community)

        coordination_edge_values = self._community_edge_metric(
            scored_edges,
            coordination_scores,
            node_to_community,
            value_column=self.config.coordination_score_column,
            fallback_value_column=self.config.graph_weight_column,
            result_column="average_coordination_score",
        )
        pair_feature_rows = self._community_pair_feature_rows(
            scored_edges,
            candidate_pairs=candidate_pairs,
            semantic_features=semantic_features,
            higher_order_features=higher_order_features,
        )
        community_metrics, diagnostics = self._community_pair_metrics(
            scored_edges,
            pair_feature_rows,
            community_ids=community_ids,
        )

        community_frame = community_sizes.merge(coordination_edge_values, on=self.config.community_id_column, how="left")
        community_frame = community_frame.merge(community_metrics, on=self.config.community_id_column, how="left")
        community_frame = community_frame.merge(diagnostics, on=self.config.community_id_column, how="left")
        community_frame = community_frame.merge(
            self._graph_density_frame(graph, memberships),
            on=self.config.community_id_column,
            how="left",
        )

        edge_metrics = (
            scored_edges.groupby(self.config.community_id_column, sort=True)
            .agg(
                edge_count=(self.config.graph_weight_column, "size"),
                average_edge_weight=(self.config.graph_weight_column, "mean"),
            )
            .reset_index()
        )
        community_frame = community_frame.merge(edge_metrics, on=self.config.community_id_column, how="left")
        community_frame["graph_density"] = community_frame["graph_density"].fillna(0.0)

        for column in ("edge_count", "average_edge_weight", "average_coordination_score", "graph_density"):
            if column in community_frame.columns:
                community_frame[column] = community_frame[column].fillna(0.0)

        feature_columns = (
            self.config.semantic_similarity_column,
            self.config.temporal_agreement_column,
            self.config.interaction_strength_column,
            self.config.repeated_target_agreement_column,
        )
        for column in feature_columns:
            if column not in community_frame.columns:
                community_frame[column] = np.nan

        zero_feature_rows = community_frame["matched_feature_rows"].fillna(0.0) <= 0.0
        if zero_feature_rows.any():
            for column in feature_columns:
                community_frame.loc[zero_feature_rows, column] = 0.0

        scored = self._score_communities(community_frame)
        scored = self._apply_structural_adjustment(scored)
        scored = scored.sort_values(
            [self.config.final_coordination_score_column, self.config.community_id_column],
            ascending=[False, True],
            kind="stable",
        ).reset_index(drop=True)
        self._log_diagnostics(scored, start, community_frame)
        return scored[
            [
                self.config.community_id_column,
                "community_size",
                "edge_count",
                "graph_density",
                "average_edge_weight",
                "average_coordination_score",
                self.config.evidence_score_column,
                self.config.semantic_similarity_column,
                self.config.temporal_agreement_column,
                self.config.interaction_strength_column,
                self.config.repeated_target_agreement_column,
                self.config.final_coordination_score_column,
            ]
        ]

    def _normalize_memberships(self, community_memberships: pd.DataFrame | dict[int, list[Any]]) -> pd.DataFrame:
        """Return a standardized community-membership table."""

        if isinstance(community_memberships, dict):
            rows = [
                {self.config.community_id_column: community_id, self.config.account_id_column: account_id}
                for community_id, accounts in community_memberships.items()
                for account_id in accounts
            ]
            frame = pd.DataFrame(rows)
        else:
            required = {self.config.community_id_column, self.config.account_id_column}
            missing = sorted(required.difference(community_memberships.columns))
            if missing:
                raise ValueError(f"Missing required membership columns: {', '.join(missing)}")
            frame = community_memberships[[self.config.community_id_column, self.config.account_id_column]].copy()

        return frame.dropna(subset=[self.config.community_id_column, self.config.account_id_column]).drop_duplicates().reset_index(
            drop=True
        )

    def _graph_edge_frame(self, graph: nx.Graph) -> pd.DataFrame:
        """Convert graph edges into a deterministic DataFrame."""

        rows = [
            {
                self.config.account_id_1_column: left,
                self.config.account_id_2_column: right,
                self.config.graph_weight_column: float(data.get(self.config.graph_weight_column, 1.0)),
                self.config.graph_support_column: int(data.get(self.config.graph_support_column, 1)),
            }
            for left, right, data in graph.edges(data=True)
        ]
        return pd.DataFrame(
            rows,
            columns=[
                self.config.account_id_1_column,
                self.config.account_id_2_column,
                self.config.graph_weight_column,
                self.config.graph_support_column,
            ],
        )

    def _attach_graph_communities(
        self,
        edge_frame: pd.DataFrame,
        node_to_community: dict[Any, Any],
    ) -> pd.DataFrame:
        """Annotate graph edges with community ids and keep intra-community edges only."""

        if edge_frame.empty:
            return pd.DataFrame(columns=[*edge_frame.columns, self.config.community_id_column])

        frame = edge_frame.copy()
        frame["community_left"] = frame[self.config.account_id_1_column].map(node_to_community)
        frame["community_right"] = frame[self.config.account_id_2_column].map(node_to_community)
        frame = frame.dropna(subset=["community_left", "community_right"])
        frame = frame[frame["community_left"] == frame["community_right"]].copy()
        frame[self.config.community_id_column] = frame["community_left"]
        return frame.drop(columns=["community_left", "community_right"]).reset_index(drop=True)

    def _community_edge_metric(
        self,
        scored_edges: pd.DataFrame,
        metric_frame: pd.DataFrame,
        node_to_community: dict[Any, Any],
        *,
        value_column: str,
        fallback_value_column: str,
        result_column: str,
    ) -> pd.DataFrame:
        """Compute the mean edge metric for each community."""

        if scored_edges.empty:
            return pd.DataFrame(columns=[self.config.community_id_column, result_column])

        if metric_frame.empty:
            metric_frame = scored_edges[[self.config.community_id_column, fallback_value_column]].rename(
                columns={fallback_value_column: value_column}
            )
        else:
            metric_frame = self._prepare_metric_frame(metric_frame, node_to_community)

        if metric_frame.empty or self.config.community_id_column not in metric_frame.columns or value_column not in metric_frame.columns:
            LOGGER.info("Falling back to graph-derived %s because no join keys were available.", result_column)
            metric_frame = scored_edges[[self.config.community_id_column, fallback_value_column]].rename(
                columns={fallback_value_column: value_column}
            )

        return (
            metric_frame.groupby(self.config.community_id_column, sort=True)[value_column]
            .mean()
            .rename(result_column)
            .reset_index()
        )

    def _community_pair_feature_rows(
        self,
        scored_edges: pd.DataFrame,
        *,
        candidate_pairs: pd.DataFrame | None,
        semantic_features: pd.DataFrame | None,
        higher_order_features: pd.DataFrame,
    ) -> pd.DataFrame:
        """Join graph edges back to the underlying pairwise feature rows."""

        if scored_edges.empty or candidate_pairs is None or candidate_pairs.empty:
            return pd.DataFrame()

        required_pairs = {
            "post_id_1",
            "post_id_2",
            self.config.account_id_1_column,
            self.config.account_id_2_column,
        }
        missing_pairs = sorted(required_pairs.difference(candidate_pairs.columns))
        if missing_pairs:
            raise ValueError(f"Missing required candidate pair columns: {', '.join(missing_pairs)}")

        pair_frame = candidate_pairs[
            [
                "post_id_1",
                "post_id_2",
                self.config.account_id_1_column,
                self.config.account_id_2_column,
            ]
        ].copy()
        pair_frame = pair_frame.dropna(subset=["post_id_1", "post_id_2", self.config.account_id_1_column, self.config.account_id_2_column])
        pair_frame = pair_frame.drop_duplicates(
            subset=[
                "post_id_1",
                "post_id_2",
                self.config.account_id_1_column,
                self.config.account_id_2_column,
            ],
            keep="first",
        )

        pair_frame = self._add_pair_keys(pair_frame, "post_id_1", "post_id_2", "_post_key")
        pair_frame = self._add_pair_keys(pair_frame, self.config.account_id_1_column, self.config.account_id_2_column, "_account_key")

        edge_pairs = scored_edges[
            [
                self.config.community_id_column,
                self.config.account_id_1_column,
                self.config.account_id_2_column,
            ]
        ].copy()
        edge_pairs = self._add_pair_keys(edge_pairs, self.config.account_id_1_column, self.config.account_id_2_column, "_account_key")

        merged = edge_pairs.merge(pair_frame, on=["_account_key_1", "_account_key_2"], how="inner", sort=False)
        if merged.empty:
            return merged

        merged = self._attach_pair_features(merged, semantic_features, higher_order_features)
        return merged.reset_index(drop=True)

    def _attach_pair_features(
        self,
        frame: pd.DataFrame,
        semantic_features: pd.DataFrame | None,
        higher_order_features: pd.DataFrame,
    ) -> pd.DataFrame:
        """Attach semantic and higher-order pair features onto matched graph edges."""

        merged = frame.copy()
        merged = self._merge_feature_frame(
            merged,
            semantic_features,
            required_columns=("cosine_similarity",),
            pair_key_prefix="_post_key",
        )
        merged = self._merge_feature_frame(
            merged,
            higher_order_features,
            required_columns=(
                "semantic_temporal_agreement",
                "thread_temporal_agreement",
                "reply_semantic_agreement",
                "interaction_strength",
                "repeated_target_agreement",
            ),
            pair_key_prefix="_post_key",
        )
        merged[self.config.semantic_similarity_column] = merged.get("cosine_similarity", np.nan)
        merged[self.config.temporal_agreement_column] = merged.get("semantic_temporal_agreement", np.nan)
        merged[self.config.interaction_strength_column] = merged.get("interaction_strength", np.nan)
        merged[self.config.repeated_target_agreement_column] = merged.get("repeated_target_agreement", np.nan)
        return merged

    def _merge_feature_frame(
        self,
        base: pd.DataFrame,
        features: pd.DataFrame | None,
        *,
        required_columns: tuple[str, ...],
        pair_key_prefix: str,
    ) -> pd.DataFrame:
        """Left-join one feature frame onto the pair/edge table."""

        if features is None or features.empty:
            for column in required_columns:
                if column not in base.columns:
                    base[column] = np.nan
            return base

        required = {"post_id_1", "post_id_2", *required_columns}
        missing = sorted(required.difference(features.columns))
        if missing:
            raise ValueError(f"Missing required feature columns: {', '.join(missing)}")

        frame = features[["post_id_1", "post_id_2", *required_columns]].copy()
        frame = frame.dropna(subset=["post_id_1", "post_id_2"])
        frame = frame.drop_duplicates(subset=["post_id_1", "post_id_2"], keep="first")
        frame = self._add_pair_keys(frame, "post_id_1", "post_id_2", pair_key_prefix)
        merged = base.merge(frame, on=[f"{pair_key_prefix}_1", f"{pair_key_prefix}_2"], how="left", sort=False)
        return merged

    def _add_pair_keys(self, frame: pd.DataFrame, left_column: str, right_column: str, prefix: str) -> pd.DataFrame:
        """Add canonical pair keys for order-insensitive joins."""

        working = frame.copy()
        left = working[left_column].to_numpy(dtype=object)
        right = working[right_column].to_numpy(dtype=object)
        left_key = np.asarray(left, dtype=str)
        right_key = np.asarray(right, dtype=str)
        swap_mask = left_key > right_key
        working[f"{prefix}_1"] = np.where(swap_mask, right, left)
        working[f"{prefix}_2"] = np.where(swap_mask, left, right)
        return working

    def _prepare_metric_frame(self, frame: pd.DataFrame, node_to_community: dict[Any, Any]) -> pd.DataFrame:
        """Attach communities to a metric frame when possible."""

        if self.config.community_id_column in frame.columns:
            return frame.copy()

        if {self.config.account_id_1_column, self.config.account_id_2_column}.issubset(frame.columns):
            working = frame.copy()
            working["community_left"] = working[self.config.account_id_1_column].map(node_to_community)
            working["community_right"] = working[self.config.account_id_2_column].map(node_to_community)
            working = working.dropna(subset=["community_left", "community_right"])
            working = working[working["community_left"] == working["community_right"]].copy()
            working[self.config.community_id_column] = working["community_left"]
            return working.drop(columns=["community_left", "community_right"])

        return pd.DataFrame()

    def _community_pair_metrics(
        self,
        scored_edges: pd.DataFrame,
        pair_feature_rows: pd.DataFrame,
        *,
        community_ids: pd.Index,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Aggregate pairwise features to community-level metrics."""

        columns = [
            self.config.semantic_similarity_column,
            self.config.temporal_agreement_column,
            self.config.interaction_strength_column,
            self.config.repeated_target_agreement_column,
        ]
        if pair_feature_rows.empty:
            metrics = pd.DataFrame({self.config.community_id_column: community_ids, **{column: 0.0 for column in columns}})
            diagnostics = self._empty_feature_diagnostics(scored_edges, community_ids)
            LOGGER.warning("No pairwise feature rows matched any community.")
            return metrics, diagnostics

        matched_feature_rows = (
            pair_feature_rows.groupby(self.config.community_id_column, sort=True)
            .size()
            .rename("matched_feature_rows")
            .reset_index()
        )
        matched_graph_edges = (
            pair_feature_rows.drop_duplicates(
                subset=[self.config.community_id_column, "_account_key_1", "_account_key_2"],
                keep="first",
            )
            .groupby(self.config.community_id_column, sort=True)
            .size()
            .rename("matched_graph_edges")
            .reset_index()
        )
        edge_counts = (
            scored_edges.groupby(self.config.community_id_column, sort=True)
            .size()
            .rename("community_edge_count")
            .reset_index()
        )

        diagnostics = pd.DataFrame({self.config.community_id_column: community_ids})
        diagnostics = diagnostics.merge(matched_feature_rows, on=self.config.community_id_column, how="left")
        diagnostics = diagnostics.merge(matched_graph_edges, on=self.config.community_id_column, how="left")
        diagnostics = diagnostics.merge(edge_counts, on=self.config.community_id_column, how="left")
        diagnostics = diagnostics.fillna(0.0)
        diagnostics["matched_edge_percentage"] = np.divide(
            diagnostics["matched_graph_edges"].to_numpy(dtype=np.float64) * 100.0,
            diagnostics["community_edge_count"].to_numpy(dtype=np.float64),
            out=np.zeros(len(diagnostics), dtype=np.float64),
            where=diagnostics["community_edge_count"].to_numpy(dtype=np.float64) > 0.0,
        )

        metrics = (
            pair_feature_rows.groupby(self.config.community_id_column, sort=True)[columns]
            .mean()
            .reset_index()
        )
        metrics = pd.DataFrame({self.config.community_id_column: community_ids}).merge(metrics, on=self.config.community_id_column, how="left")
        metrics = metrics.reindex(columns=[self.config.community_id_column, *columns])
        zero_feature_rows = diagnostics["matched_feature_rows"].fillna(0.0) <= 0.0
        if zero_feature_rows.any():
            for column in columns:
                metrics.loc[metrics[self.config.community_id_column].isin(diagnostics.loc[zero_feature_rows, self.config.community_id_column]), column] = 0.0
        return metrics, diagnostics

    def _empty_feature_diagnostics(self, scored_edges: pd.DataFrame, community_ids: pd.Index) -> pd.DataFrame:
        """Return zero-valued feature diagnostics."""

        diagnostics = pd.DataFrame({self.config.community_id_column: community_ids})
        edge_counts = (
            scored_edges.groupby(self.config.community_id_column, sort=True)
            .size()
            .rename("community_edge_count")
            .reset_index()
        )
        diagnostics = diagnostics.merge(edge_counts, on=self.config.community_id_column, how="left")
        diagnostics["community_edge_count"] = diagnostics["community_edge_count"].fillna(0.0)
        diagnostics["matched_feature_rows"] = 0.0
        diagnostics["matched_graph_edges"] = 0.0
        diagnostics["matched_edge_percentage"] = 0.0
        return diagnostics

    def _graph_density_frame(self, graph: nx.Graph, memberships: pd.DataFrame) -> pd.DataFrame:
        """Compute induced-subgraph density for each community."""

        rows: list[dict[str, Any]] = []
        for community_id, group in memberships.groupby(self.config.community_id_column, sort=True):
            community_nodes = group[self.config.account_id_column].dropna().unique().tolist()
            subgraph = graph.subgraph(community_nodes)
            rows.append(
                {
                    self.config.community_id_column: community_id,
                    "graph_density": float(nx.density(subgraph)),
                }
            )
        return pd.DataFrame(rows, columns=[self.config.community_id_column, "graph_density"])

    def _score_communities(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Normalize score inputs and combine them into one evidence score."""

        metrics = pd.DataFrame(
            {
                "community_size": self._normalize_count(frame["community_size"].to_numpy(dtype=np.float64)),
                "edge_count": self._normalize_count(frame["edge_count"].to_numpy(dtype=np.float64)),
                "graph_density": frame["graph_density"].to_numpy(dtype=np.float64),
                "average_edge_weight": self._normalize_unit(frame["average_edge_weight"].to_numpy(dtype=np.float64)),
                "average_coordination_score": self._normalize_unit(
                    frame["average_coordination_score"].to_numpy(dtype=np.float64)
                ),
                self.config.semantic_similarity_column: self._normalize_unit(
                    frame[self.config.semantic_similarity_column].to_numpy(dtype=np.float64)
                ),
                self.config.temporal_agreement_column: self._normalize_unit(
                    frame[self.config.temporal_agreement_column].to_numpy(dtype=np.float64)
                ),
                self.config.interaction_strength_column: self._normalize_unit(
                    frame[self.config.interaction_strength_column].to_numpy(dtype=np.float64)
                ),
                self.config.repeated_target_agreement_column: self._normalize_unit(
                    frame[self.config.repeated_target_agreement_column].to_numpy(dtype=np.float64)
                ),
            }
        )
        weights = self._weights()
        score = np.zeros(len(metrics), dtype=np.float64)
        total_weight = 0.0
        for column, weight in weights.items():
            if weight <= 0:
                continue
            score += metrics[column].to_numpy(dtype=np.float64) * weight
            total_weight += weight

        frame = frame.copy()
        frame[self.config.evidence_score_column] = (
            np.divide(score, total_weight, out=np.zeros_like(score), where=total_weight > 0.0)
            if total_weight > 0.0
            else np.zeros(len(frame), dtype=np.float64)
        )
        return frame

    def _apply_structural_adjustment(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Apply structural penalties to the evidence score."""

        adjusted = frame.copy()
        evidence_score = adjusted[self.config.evidence_score_column].to_numpy(dtype=np.float64)
        density_modifier = self._density_modifier(adjusted["graph_density"].to_numpy(dtype=np.float64))
        edge_strength_modifier = self._edge_strength_modifier(adjusted["average_edge_weight"].to_numpy(dtype=np.float64))
        size_modifier = self._size_modifier(adjusted["community_size"].to_numpy(dtype=np.float64))
        diversity_modifier = self._diversity_modifier(adjusted)

        adjusted[self.config.final_coordination_score_column] = np.clip(
            evidence_score * density_modifier * edge_strength_modifier * diversity_modifier * size_modifier,
            0.0,
            1.0,
        )
        return adjusted

    def _size_modifier(self, values: np.ndarray) -> np.ndarray:
        """Return a size penalty anchored at the minimum community size."""

        minimum = max(float(getattr(pipeline_config, "MIN_COMMUNITY_SIZE", 1)), 1.0)
        if minimum <= 1.0:
            return np.ones_like(values, dtype=np.float64)
        return np.clip((values - 1.0) / (minimum - 1.0), 0.0, 1.0)

    def _density_modifier(self, values: np.ndarray) -> np.ndarray:
        """Return a density penalty anchored at the configured minimum density."""

        minimum = float(getattr(pipeline_config, "MIN_GRAPH_DENSITY", 0.0))
        if minimum <= 0.0:
            return np.ones_like(values, dtype=np.float64)
        return np.clip(values / minimum, 0.0, 1.0)

    def _edge_strength_modifier(self, values: np.ndarray) -> np.ndarray:
        """Return an edge-strength penalty anchored at the configured minimum."""

        minimum = float(getattr(pipeline_config, "MIN_AVERAGE_EDGE_WEIGHT", 0.0))
        if minimum <= 0.0:
            return np.ones_like(values, dtype=np.float64)
        return np.clip(values / minimum, 0.0, 1.0)

    def _diversity_modifier(self, frame: pd.DataFrame) -> np.ndarray:
        """Return a dominance penalty derived from the existing evidence columns."""

        threshold = float(getattr(pipeline_config, "DOMINANT_SIGNAL_THRESHOLD", 0.70))
        if threshold <= 0.0:
            return np.ones(len(frame), dtype=np.float64)

        values = np.column_stack(
            [
                frame["average_edge_weight"].to_numpy(dtype=np.float64),
                frame[self.config.semantic_similarity_column].to_numpy(dtype=np.float64),
                frame[self.config.temporal_agreement_column].to_numpy(dtype=np.float64),
                frame[self.config.interaction_strength_column].to_numpy(dtype=np.float64),
                frame[self.config.repeated_target_agreement_column].to_numpy(dtype=np.float64),
            ]
        )
        values = np.clip(values, 0.0, None)
        totals = values.sum(axis=1)
        dominant = np.divide(
            values.max(axis=1),
            totals,
            out=np.zeros(len(frame), dtype=np.float64),
            where=totals > 0.0,
        )
        return np.clip(1.0 - (dominant / threshold), 0.0, 1.0)

    def _weights(self) -> dict[str, float]:
        """Return the active community-score weights."""

        overrides = getattr(pipeline_config, "COMMUNITY_SCORING_WEIGHTS", None)
        if isinstance(overrides, dict):
            return {
                "community_size": float(overrides.get("community_size", self.config.community_size_weight)),
                "edge_count": float(overrides.get("edge_count", self.config.edge_count_weight)),
                "graph_density": float(overrides.get("graph_density", self.config.graph_density_weight)),
                "average_edge_weight": float(overrides.get("average_edge_weight", self.config.average_edge_weight_weight)),
                "average_coordination_score": float(
                    overrides.get("average_coordination_score", self.config.average_coordination_score_weight)
                ),
                self.config.semantic_similarity_column: float(
                    overrides.get("mean_semantic_similarity", self.config.semantic_similarity_weight)
                ),
                self.config.temporal_agreement_column: float(
                    overrides.get("mean_temporal_agreement", self.config.temporal_agreement_weight)
                ),
                self.config.interaction_strength_column: float(
                    overrides.get("mean_interaction_strength", self.config.interaction_strength_weight)
                ),
                self.config.repeated_target_agreement_column: float(
                    overrides.get("mean_repeated_target_agreement", self.config.repeated_target_agreement_weight)
                ),
            }

        return {
            "community_size": float(getattr(pipeline_config, "COMMUNITY_SIZE_WEIGHT", self.config.community_size_weight)),
            "edge_count": float(getattr(pipeline_config, "EDGE_COUNT_WEIGHT", self.config.edge_count_weight)),
            "graph_density": float(getattr(pipeline_config, "GRAPH_DENSITY_WEIGHT", self.config.graph_density_weight)),
            "average_edge_weight": float(
                getattr(pipeline_config, "AVERAGE_EDGE_WEIGHT_WEIGHT", self.config.average_edge_weight_weight)
            ),
            "average_coordination_score": float(
                getattr(pipeline_config, "AVERAGE_COORDINATION_SCORE_WEIGHT", self.config.average_coordination_score_weight)
            ),
            self.config.semantic_similarity_column: float(
                getattr(pipeline_config, "MEAN_SEMANTIC_SIMILARITY_WEIGHT", self.config.semantic_similarity_weight)
            ),
            self.config.temporal_agreement_column: float(
                getattr(pipeline_config, "MEAN_TEMPORAL_AGREEMENT_WEIGHT", self.config.temporal_agreement_weight)
            ),
            self.config.interaction_strength_column: float(
                getattr(pipeline_config, "MEAN_INTERACTION_STRENGTH_WEIGHT", self.config.interaction_strength_weight)
            ),
            self.config.repeated_target_agreement_column: float(
                getattr(
                    pipeline_config,
                    "MEAN_REPEATED_TARGET_AGREEMENT_WEIGHT",
                    self.config.repeated_target_agreement_weight,
                )
            ),
        }

    def _normalize_count(self, values: np.ndarray) -> np.ndarray:
        """Normalize count-like values using log1p scaling."""

        values = np.clip(values, 0.0, None)
        logged = np.log1p(values)
        maximum = float(logged.max(initial=0.0))
        if maximum <= 0.0:
            return np.zeros_like(logged, dtype=np.float64)
        return logged / maximum

    def _normalize_unit(self, values: np.ndarray) -> np.ndarray:
        """Normalize bounded positive values into [0, 1] when needed."""

        values = np.clip(values, 0.0, None)
        maximum = float(values.max(initial=0.0))
        if maximum <= 0.0:
            return np.zeros_like(values, dtype=np.float64)
        if maximum <= 1.0:
            return values.astype(np.float64)
        return values / maximum

    def _log_diagnostics(self, frame: pd.DataFrame, start: float, diagnostics: pd.DataFrame | None = None) -> None:
        """Log score distribution and community diagnostics."""

        score = frame[self.config.final_coordination_score_column].astype(np.float64)
        LOGGER.info("Runtime: %.2fs", perf_counter() - start)
        LOGGER.info("Number of communities: %d", len(frame))
        LOGGER.info("Average community size: %.2f", float(frame["community_size"].mean()) if not frame.empty else 0.0)
        LOGGER.info("Score distribution: %s", score.describe(percentiles=(0.25, 0.5, 0.75, 0.9, 0.95)).to_dict())
        LOGGER.info("Top 20 highest-scoring communities:\n%s", frame.head(20).to_string(index=False))
        LOGGER.info(
            "Largest communities:\n%s",
            frame.sort_values(
                ["community_size", self.config.final_coordination_score_column],
                ascending=[False, False],
                kind="stable",
            )
            .head(20)
            .to_string(index=False),
        )
        if diagnostics is not None and not diagnostics.empty:
            LOGGER.info(
                "Matched feature rows per community:\n%s",
                diagnostics[[self.config.community_id_column, "matched_feature_rows"]].to_string(index=False),
            )
            LOGGER.info(
                "Matched graph edges per community:\n%s",
                diagnostics[
                    [
                        self.config.community_id_column,
                        "community_edge_count",
                        "matched_graph_edges",
                        "matched_edge_percentage",
                    ]
                ].to_string(index=False),
            )
            zero_rows = diagnostics.loc[diagnostics["matched_feature_rows"].fillna(0.0) <= 0.0, self.config.community_id_column].tolist()
            if zero_rows:
                LOGGER.warning("Communities with zero matched feature rows: %s", zero_rows)

    def _empty_output(self) -> pd.DataFrame:
        """Return an empty scored-community table."""

        return pd.DataFrame(
            columns=[
                self.config.community_id_column,
                "community_size",
                "edge_count",
                "graph_density",
                "average_edge_weight",
                "average_coordination_score",
                self.config.evidence_score_column,
                self.config.semantic_similarity_column,
                self.config.temporal_agreement_column,
                self.config.interaction_strength_column,
                self.config.repeated_target_agreement_column,
                self.config.final_coordination_score_column,
            ]
        )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    graph = nx.Graph()
    graph.add_edge("a1", "a2", weight=0.9)
    graph.add_edge("a2", "a3", weight=0.8)
    graph.add_edge("b1", "b2", weight=0.7)
    memberships = pd.DataFrame(
        [
            {"community_id": 0, "account_id": "a1"},
            {"community_id": 0, "account_id": "a2"},
            {"community_id": 0, "account_id": "a3"},
            {"community_id": 1, "account_id": "b1"},
            {"community_id": 1, "account_id": "b2"},
        ]
    )
    coordination_scores = pd.DataFrame(
        [
            {"account_id_1": "a1", "account_id_2": "a2", "coordination_score": 0.9},
            {"account_id_1": "a2", "account_id_2": "a3", "coordination_score": 0.8},
            {"account_id_1": "b1", "account_id_2": "b2", "coordination_score": 0.7},
        ]
    )
    candidate_pairs = pd.DataFrame(
        [
            {"post_id_1": "p1", "post_id_2": "p2", "account_id_1": "a1", "account_id_2": "a2"},
            {"post_id_1": "p3", "post_id_2": "p4", "account_id_1": "a2", "account_id_2": "a3"},
            {"post_id_1": "p5", "post_id_2": "p6", "account_id_1": "b1", "account_id_2": "b2"},
        ]
    )
    semantic_features = pd.DataFrame(
        [
            {"post_id_1": "p1", "post_id_2": "p2", "cosine_similarity": 0.4},
            {"post_id_1": "p3", "post_id_2": "p4", "cosine_similarity": 0.2},
            {"post_id_1": "p5", "post_id_2": "p6", "cosine_similarity": 0.1},
        ]
    )
    higher_order_features = pd.DataFrame(
        [
            {
                "post_id_1": "p1",
                "post_id_2": "p2",
                "semantic_temporal_agreement": 0.5,
                "thread_temporal_agreement": 0.1,
                "reply_semantic_agreement": 0.0,
                "interaction_strength": 0.3,
                "repeated_target_agreement": 0.2,
            },
            {
                "post_id_1": "p3",
                "post_id_2": "p4",
                "semantic_temporal_agreement": 0.4,
                "thread_temporal_agreement": 0.0,
                "reply_semantic_agreement": 0.2,
                "interaction_strength": 0.6,
                "repeated_target_agreement": 0.1,
            },
            {
                "post_id_1": "p5",
                "post_id_2": "p6",
                "semantic_temporal_agreement": 0.2,
                "thread_temporal_agreement": 0.0,
                "reply_semantic_agreement": 0.1,
                "interaction_strength": 0.1,
                "repeated_target_agreement": 0.4,
            },
        ]
    )
    result = CommunityScorer().transform(
        graph,
        memberships,
        coordination_scores,
        higher_order_features,
        semantic_features=semantic_features,
        candidate_pairs=candidate_pairs,
    )
    assert list(result.columns) == [
        "community_id",
        "community_size",
        "edge_count",
        "graph_density",
        "average_edge_weight",
        "average_coordination_score",
        "evidence_score",
        "mean_semantic_similarity",
        "mean_temporal_agreement",
        "mean_interaction_strength",
        "mean_repeated_target_agreement",
        "community_coordination_score",
    ]
    assert float(result.loc[result["community_id"] == 0, "mean_semantic_similarity"].iloc[0]) > 0.0
