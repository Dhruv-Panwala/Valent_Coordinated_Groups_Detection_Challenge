"""Write coordination clusters to JSON output."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, Iterable

import pandas as pd

if __package__ in (None, ""):
    import sys

    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from coordination_detection import config as pipeline_config
else:
    from . import config as pipeline_config

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class OutputConfig:
    """Configuration for the final JSON results file."""

    community_id_column: str = "community_id"
    account_id_column: str = "account_id"
    post_id_column: str = "post_id"
    community_coordination_score_column: str = "community_coordination_score"
    is_coordinated_column: str = "is_coordinated"
    output_file: str = pipeline_config.RESULTS_OUTPUT_FILE


class ResultsWriter:
    """Convert filtered community decisions into the required JSON cluster format."""

    def __init__(self, config: OutputConfig | None = None) -> None:
        self.config = config or OutputConfig()

    def transform(
        self,
        community_scores: pd.DataFrame,
        community_memberships: pd.DataFrame | dict[int, list[Any]],
        posts: pd.DataFrame,
        output_path: str | Path | None = None,
    ) -> Path:
        """Write a `results.json` file and return its path."""

        start = perf_counter()
        memberships = self._normalize_memberships(community_memberships)
        posts_frame = self._validate_posts(posts)
        communities = self._prepare_communities(community_scores)
        output_file = Path(output_path or self.config.output_file)

        payload = {"clusters": []}
        if not communities.empty:
            cluster_frame = self._build_cluster_frame(communities, memberships, posts_frame)
            payload["clusters"] = [
                {
                    "post_ids": row.post_ids,
                    "is_coordinated": bool(row.is_coordinated),
                    "coordination_score": float(row.community_coordination_score),
                }
                for row in cluster_frame.itertuples(index=False)
            ]

        output_file.parent.mkdir(parents=True, exist_ok=True)
        output_file.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        LOGGER.info("Saved results to %s", output_file)
        LOGGER.info("Runtime: %.2fs", perf_counter() - start)
        return output_file

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

    def _validate_posts(self, posts: pd.DataFrame) -> pd.DataFrame:
        """Validate that the post table contains the columns needed for output."""

        required = {self.config.account_id_column, self.config.post_id_column}
        missing = sorted(required.difference(posts.columns))
        if missing:
            raise ValueError(f"Missing required post columns: {', '.join(missing)}")
        return posts[[self.config.account_id_column, self.config.post_id_column]].copy()

    def _prepare_communities(self, community_scores: pd.DataFrame) -> pd.DataFrame:
        """Return every community with its filter decision and raw score."""

        required = {
            self.config.community_id_column,
            self.config.community_coordination_score_column,
            self.config.is_coordinated_column,
        }
        missing = sorted(required.difference(community_scores.columns))
        if missing:
            raise ValueError(f"Missing required community-score columns: {', '.join(missing)}")

        frame = community_scores.copy()
        frame[self.config.is_coordinated_column] = frame[self.config.is_coordinated_column].fillna(False).astype(bool)
        return frame[
            [
                self.config.community_id_column,
                self.config.community_coordination_score_column,
                self.config.is_coordinated_column,
            ]
        ].reset_index(drop=True)

    def _build_cluster_frame(
        self,
        retained: pd.DataFrame,
        memberships: pd.DataFrame,
        posts: pd.DataFrame,
    ) -> pd.DataFrame:
        """Expand communities to unique post identifiers."""

        joined = memberships.merge(posts, on=self.config.account_id_column, how="inner")
        if joined.empty:
            return retained.assign(post_ids=[[] for _ in range(len(retained))]).sort_values(
                [self.config.community_coordination_score_column, self.config.community_id_column],
                ascending=[False, True],
                kind="stable",
            )

        post_lists = (
            joined.groupby(self.config.community_id_column, sort=True)[self.config.post_id_column]
            .agg(lambda values: sorted(dict.fromkeys(value for value in values.dropna().tolist()), key=str))
            .rename("post_ids")
            .reset_index()
        )
        cluster_frame = retained.merge(post_lists, on=self.config.community_id_column, how="left")
        cluster_frame["post_ids"] = cluster_frame["post_ids"].apply(self._post_id_list)
        return cluster_frame.sort_values(
            [self.config.community_coordination_score_column, self.config.community_id_column],
            ascending=[False, True],
            kind="stable",
        ).reset_index(drop=True)

    def _post_id_list(self, value: Any) -> list[Any]:
        """Return JSON-serializable post ids from pandas/object list values."""

        if value is None or pd.isna(value) is True:
            return []
        if isinstance(value, str):
            return [value]
        if isinstance(value, Iterable):
            return list(value)
        return []


def write_results_json(
    community_scores: pd.DataFrame,
    community_memberships: pd.DataFrame | dict[int, list[Any]],
    posts: pd.DataFrame,
    output_path: str | Path | None = None,
    config: OutputConfig | None = None,
) -> Path:
    """Write the final results JSON file."""

    return ResultsWriter(config).transform(community_scores, community_memberships, posts, output_path=output_path)
