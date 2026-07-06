"""Build a filtered weighted account graph from pair coordination scores."""

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
class GraphBuilderConfig:
    """Configuration for graph construction from scored candidate pairs."""

    candidate_post_id_1_column: str = "post_id_1"
    candidate_post_id_2_column: str = "post_id_2"
    candidate_account_id_1_column: str = "account_id_1"
    candidate_account_id_2_column: str = "account_id_2"
    score_column: str = "coordination_score"
    edge_percentile: float = float(getattr(pipeline_config, "EDGE_PERCENTILE", 95.0))
    edge_threshold: float | None = getattr(pipeline_config, "EDGE_THRESHOLD", None)


class GraphBuilder:
    """Construct a weighted undirected account graph from scored post pairs."""

    def __init__(self, config: GraphBuilderConfig | None = None) -> None:
        self.config = config or GraphBuilderConfig()

    def transform(
        self,
        candidate_pairs: pd.DataFrame,
        coordination_scores: pd.DataFrame,
    ) -> tuple[nx.Graph, pd.DataFrame]:
        """Return a weighted account graph and the aggregated edge table."""

        start = perf_counter()
        merged = self._merge_inputs(candidate_pairs, coordination_scores)
        if merged.empty:
            LOGGER.info("Candidate post-pair edges: 0")
            LOGGER.info("Filtered edges: 0")
            LOGGER.info("Unique account edges: 0")
            LOGGER.info("Node count: 0")
            LOGGER.info("Edge count: 0")
            LOGGER.info("Average edge weight: 0.000000")
            LOGGER.info("Average support: 0.000000")
            LOGGER.info("Connected components: 0")
            LOGGER.info("Largest connected component: 0")
            LOGGER.info("Graph density: 0.000000")
            LOGGER.info("Runtime: %.2fs", perf_counter() - start)
            return nx.Graph(), self._empty_edge_frame()

        threshold, threshold_label = self._resolve_threshold(merged[self.config.score_column])
        filtered = merged[merged[self.config.score_column] >= threshold].copy()
        filtered_edges = len(filtered)

        edge_df = self._aggregate_account_edges(filtered)
        graph = self._build_graph(edge_df)

        component_sizes = [len(component) for component in nx.connected_components(graph)] if graph.number_of_nodes() else []
        largest_component = max(component_sizes, default=0)
        density = float(nx.density(graph)) if graph.number_of_nodes() > 1 else 0.0

        LOGGER.info("Candidate post-pair edges: %d", len(merged))
        LOGGER.info("Edge filter: %s = %.6f", threshold_label, threshold)
        LOGGER.info("Filtered edges: %d", filtered_edges)
        LOGGER.info("Unique account edges: %d", len(edge_df))
        LOGGER.info("Node count: %d", graph.number_of_nodes())
        LOGGER.info("Edge count: %d", graph.number_of_edges())
        LOGGER.info("Average edge weight: %.6f", float(edge_df["weight"].mean()) if not edge_df.empty else 0.0)
        LOGGER.info("Average support: %.6f", float(edge_df["support"].mean()) if not edge_df.empty else 0.0)
        LOGGER.info("Connected components: %d", len(component_sizes))
        LOGGER.info("Largest connected component: %d", largest_component)
        LOGGER.info("Graph density: %.6f", density)
        LOGGER.info("Runtime: %.2fs", perf_counter() - start)
        return graph, edge_df

    def _merge_inputs(
        self,
        candidate_pairs: pd.DataFrame,
        coordination_scores: pd.DataFrame,
    ) -> pd.DataFrame:
        """Validate inputs and merge post-pair metadata with coordination scores."""

        candidate_required = {
            self.config.candidate_post_id_1_column,
            self.config.candidate_post_id_2_column,
            self.config.candidate_account_id_1_column,
            self.config.candidate_account_id_2_column,
        }
        score_required = {
            self.config.candidate_post_id_1_column,
            self.config.candidate_post_id_2_column,
            self.config.score_column,
        }

        missing_candidate = sorted(candidate_required.difference(candidate_pairs.columns))
        missing_scores = sorted(score_required.difference(coordination_scores.columns))
        if missing_candidate:
            raise ValueError(f"Missing required columns in candidate pairs: {', '.join(missing_candidate)}")
        if missing_scores:
            raise ValueError(f"Missing required columns in coordination scores: {', '.join(missing_scores)}")

        candidate_frame = candidate_pairs[
            [
                self.config.candidate_post_id_1_column,
                self.config.candidate_post_id_2_column,
                self.config.candidate_account_id_1_column,
                self.config.candidate_account_id_2_column,
            ]
        ].copy()
        score_frame = coordination_scores[
            [
                self.config.candidate_post_id_1_column,
                self.config.candidate_post_id_2_column,
                self.config.score_column,
            ]
        ].copy()
        merged = candidate_frame.merge(
            score_frame,
            on=[self.config.candidate_post_id_1_column, self.config.candidate_post_id_2_column],
            how="inner",
            sort=False,
        )
        return merged.dropna(
            subset=[
                self.config.candidate_account_id_1_column,
                self.config.candidate_account_id_2_column,
                self.config.score_column,
            ]
        ).reset_index(drop=True)

    def _resolve_threshold(self, scores: pd.Series) -> tuple[float, str]:
        """Return the score cutoff used to keep graph edges."""

        threshold = self.config.edge_threshold
        if threshold is not None:
            return float(threshold), "EDGE_THRESHOLD"

        percentile = float(self.config.edge_percentile)
        return float(np.percentile(scores.to_numpy(dtype=np.float64), percentile)), f"EDGE_PERCENTILE({percentile:g})"

    def _aggregate_account_edges(self, filtered: pd.DataFrame) -> pd.DataFrame:
        """Aggregate multiple post-pair edges into account-level edges."""

        if filtered.empty:
            return self._empty_edge_frame()

        account_1, account_2 = self._normalized_account_pair_columns(filtered)
        working = filtered.copy()
        working[self.config.candidate_account_id_1_column] = account_1
        working[self.config.candidate_account_id_2_column] = account_2
        aggregated = (
            working.groupby(
                [
                    self.config.candidate_account_id_1_column,
                    self.config.candidate_account_id_2_column,
                ],
                sort=True,
                as_index=False,
            )
            .agg(
                weight=(self.config.score_column, "max"),
                support=(self.config.score_column, "size"),
            )
            .sort_values(
                [
                    "weight",
                    "support",
                    self.config.candidate_account_id_1_column,
                    self.config.candidate_account_id_2_column,
                ],
                ascending=[False, False, True, True],
                kind="stable",
            )
            .reset_index(drop=True)
        )
        return aggregated

    def _normalized_account_pair_columns(self, frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """Return order-independent account-pair columns."""

        left = frame[self.config.candidate_account_id_1_column].to_numpy(dtype=object)
        right = frame[self.config.candidate_account_id_2_column].to_numpy(dtype=object)
        left_key = np.asarray(left, dtype=str)
        right_key = np.asarray(right, dtype=str)
        swap_mask = left_key > right_key
        normalized_left = np.where(swap_mask, right, left)
        normalized_right = np.where(swap_mask, left, right)
        return normalized_left, normalized_right

    def _build_graph(self, edge_df: pd.DataFrame) -> nx.Graph:
        """Create an undirected weighted NetworkX graph from account edges."""

        graph = nx.Graph()
        for row in edge_df.itertuples(index=False):
            graph.add_edge(
                getattr(row, self.config.candidate_account_id_1_column),
                getattr(row, self.config.candidate_account_id_2_column),
                weight=float(row.weight),
                support=int(row.support),
            )
        return graph

    def _empty_edge_frame(self) -> pd.DataFrame:
        """Return an empty aggregated edge table."""

        return pd.DataFrame(
            columns=[
                self.config.candidate_account_id_1_column,
                self.config.candidate_account_id_2_column,
                "weight",
                "support",
            ]
        )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    candidate_pairs = pd.DataFrame(
        [
            {"post_id_1": "p1", "post_id_2": "p2", "account_id_1": "a1", "account_id_2": "a2"},
            {"post_id_1": "p3", "post_id_2": "p4", "account_id_1": "a1", "account_id_2": "a2"},
            {"post_id_1": "p5", "post_id_2": "p6", "account_id_1": "a2", "account_id_2": "a3"},
        ]
    )
    coordination_scores = pd.DataFrame(
        [
            {"post_id_1": "p1", "post_id_2": "p2", "coordination_score": 0.7},
            {"post_id_1": "p3", "post_id_2": "p4", "coordination_score": 0.9},
            {"post_id_1": "p5", "post_id_2": "p6", "coordination_score": 0.2},
        ]
    )
    graph, edges = GraphBuilder().transform(candidate_pairs, coordination_scores)
    assert graph.number_of_nodes() == 2
    assert graph.number_of_edges() == 1
    assert list(edges.columns) == ["account_id_1", "account_id_2", "weight", "support"]
