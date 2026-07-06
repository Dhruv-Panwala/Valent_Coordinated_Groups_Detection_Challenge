"""Filter scored communities to separate coordinated groups from organic crowds."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from time import perf_counter
from typing import Any

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


def _config_value(name: str, default: Any) -> Any:
    """Return a pipeline config value when available, otherwise a default."""

    return getattr(pipeline_config, name, default)


@dataclass(frozen=True, slots=True)
class OrganicCrowdFilterConfig:
    """Configuration for the organic-crowd filter."""

    community_id_column: str = "community_id"
    account_id_column: str = "account_id"
    community_size_column: str = "community_size"
    community_coordination_score_column: str = "community_coordination_score"
    final_coordination_threshold: float = float(_config_value("FINAL_COORDINATION_THRESHOLD", 0.15))


class OrganicCrowdFilter:
    """Filter already-scored communities to keep likely coordinated groups only."""

    def __init__(self, config: OrganicCrowdFilterConfig | None = None) -> None:
        self.config = config or OrganicCrowdFilterConfig()

    def transform(
        self,
        community_scores: pd.DataFrame,
        community_memberships: pd.DataFrame | dict[int, list[Any]],
    ) -> pd.DataFrame:
        """Return coordinated communities using the final score threshold."""

        start = perf_counter()
        memberships = self._normalize_memberships(community_memberships)
        if community_scores.empty or memberships.empty:
            LOGGER.info("Communities before filtering: %d", len(community_scores))
            LOGGER.info("Communities after filtering: 0")
            LOGGER.info("Runtime: %.2fs", perf_counter() - start)
            return self._empty_output()

        frame = self._prepare_community_frame(community_scores, memberships)
        if frame.empty:
            LOGGER.info("Communities before filtering: 0")
            LOGGER.info("Communities after filtering: 0")
            LOGGER.info("Runtime: %.2fs", perf_counter() - start)
            return self._empty_output()

        frame = frame.sort_values(self.config.community_id_column, kind="stable").reset_index(drop=True)
        frame["is_coordinated"] = (
            frame[self.config.community_coordination_score_column]
            .astype(float)
            >= float(self.config.final_coordination_threshold)
        )

        output = frame[
            [
                self.config.community_id_column,
                self.config.community_size_column,
                self.config.community_coordination_score_column,
                "is_coordinated",
            ]
        ].copy()

        retained = int(output["is_coordinated"].sum())
        total = int(len(output))
        LOGGER.info("Runtime: %.2fs", perf_counter() - start)
        LOGGER.info("Communities before filtering: %d", total)
        LOGGER.info("Communities after filtering: %d", retained)
        LOGGER.info("Final coordination threshold: %.6f", float(self.config.final_coordination_threshold))
        LOGGER.info("Percentage retained: %.2f", (retained / total * 100.0) if total else 0.0)
        return output

    def _normalize_memberships(self, community_memberships: pd.DataFrame | dict[int, list[Any]]) -> pd.DataFrame:
        """Return a normalized community-membership table."""

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

    def _prepare_community_frame(self, community_scores: pd.DataFrame, memberships: pd.DataFrame) -> pd.DataFrame:
        """Attach computed community sizes and validate required score columns."""

        required = {
            self.config.community_id_column,
            self.config.community_coordination_score_column,
        }
        missing = sorted(required.difference(community_scores.columns))
        if missing:
            raise ValueError(f"Missing required community-score columns: {', '.join(missing)}")

        size_frame = (
            memberships.groupby(self.config.community_id_column, sort=True)[self.config.account_id_column]
            .nunique()
            .rename(self.config.community_size_column)
            .reset_index()
        )
        frame = community_scores.copy()
        if self.config.community_size_column in frame.columns:
            frame = frame.drop(columns=[self.config.community_size_column])
        frame = frame.merge(size_frame, on=self.config.community_id_column, how="left")
        frame[self.config.community_size_column] = frame[self.config.community_size_column].fillna(0).astype(int)
        frame[self.config.community_coordination_score_column] = pd.to_numeric(
            frame[self.config.community_coordination_score_column],
            errors="coerce",
        ).fillna(0.0)
        return frame

    def _empty_output(self) -> pd.DataFrame:
        """Return an empty output frame with the expected schema."""

        return pd.DataFrame(
            columns=[
                self.config.community_id_column,
                self.config.community_size_column,
                self.config.community_coordination_score_column,
                "is_coordinated",
            ]
        )


def filter_organic_crowds(
    community_scores: pd.DataFrame,
    community_memberships: pd.DataFrame | dict[int, list[Any]],
    config: OrganicCrowdFilterConfig | None = None,
) -> pd.DataFrame:
    """Convenience wrapper for filtering scored communities."""

    return OrganicCrowdFilter(config).transform(community_scores, community_memberships)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    community_scores = pd.DataFrame(
        [
            {
                "community_id": 1,
                "graph_density": 0.8,
                "average_edge_weight": 0.7,
                "community_coordination_score": 0.75,
                "mean_semantic_similarity": 0.2,
                "mean_temporal_agreement": 0.2,
                "mean_interaction_strength": 0.2,
                "mean_repeated_target_agreement": 0.2,
            },
            {
                "community_id": 2,
                "graph_density": 0.1,
                "average_edge_weight": 0.2,
                "community_coordination_score": 0.2,
                "mean_semantic_similarity": 0.8,
                "mean_temporal_agreement": 0.05,
                "mean_interaction_strength": 0.05,
                "mean_repeated_target_agreement": 0.05,
            },
        ]
    )
    memberships = pd.DataFrame(
        [
            {"community_id": 1, "account_id": "a1"},
            {"community_id": 1, "account_id": "a2"},
            {"community_id": 2, "account_id": "b1"},
            {"community_id": 2, "account_id": "b2"},
            {"community_id": 2, "account_id": "b3"},
        ]
    )
    result = filter_organic_crowds(community_scores, memberships)
    assert list(result.columns) == [
        "community_id",
        "community_size",
        "community_coordination_score",
        "is_coordinated",
    ]
