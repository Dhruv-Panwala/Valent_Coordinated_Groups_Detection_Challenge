"""Temporal feature extraction for candidate pairs."""

from __future__ import annotations

import logging
from time import perf_counter

import numpy as np
import pandas as pd

from .config import TEMPORAL_DECAY_SECONDS

LOGGER = logging.getLogger(__name__)


class TemporalFeatureExtractor:
    """Compute reusable temporal evidence for candidate post pairs."""

    def __init__(self, decay_seconds: int = TEMPORAL_DECAY_SECONDS) -> None:
        if decay_seconds <= 0:
            raise ValueError("decay_seconds must be positive")
        self.decay_seconds = decay_seconds

    def transform(self, candidate_pairs: pd.DataFrame, posts: pd.DataFrame) -> pd.DataFrame:
        """Return temporal features for each candidate pair."""

        start = perf_counter()
        pairs = self._validate_candidate_pairs(candidate_pairs)
        timestamps = self._prepare_timestamps(posts)

        if pairs.empty or timestamps.empty:
            LOGGER.info("Candidate pairs processed: %d", len(pairs))
            LOGGER.info("Runtime: %.2fs", perf_counter() - start)
            return self._empty_output()

        merged = self._attach_timestamps(pairs, timestamps)
        processed = len(merged)
        if processed == 0:
            LOGGER.info("Candidate pairs processed: %d", len(pairs))
            LOGGER.info("Runtime: %.2fs", perf_counter() - start)
            return self._empty_output()

        absolute_diff_seconds = (
            merged["created_at_1"] - merged["created_at_2"]
        ).abs().dt.total_seconds().astype(np.float64)
        decay_scores = np.exp(-absolute_diff_seconds / float(self.decay_seconds))

        output = pd.DataFrame(
            {
                "post_id_1": merged["post_id_1"],
                "post_id_2": merged["post_id_2"],
                "absolute_time_difference_seconds": absolute_diff_seconds,
                "exponential_decay_score": decay_scores,
            }
        ).reset_index(drop=True)

        self._log_summary(output, processed, start)
        return output

    def _validate_candidate_pairs(self, candidate_pairs: pd.DataFrame) -> pd.DataFrame:
        """Validate the candidate-pair input and keep only the needed columns."""

        required = {"post_id_1", "post_id_2"}
        missing = sorted(required.difference(candidate_pairs.columns))
        if missing:
            raise ValueError(f"Missing required columns: {', '.join(missing)}")
        return candidate_pairs[["post_id_1", "post_id_2"]].copy()

    def _prepare_timestamps(self, posts: pd.DataFrame) -> pd.DataFrame:
        """Validate and normalize timestamps for post lookup."""

        required = {"post_id", "created_at"}
        missing = sorted(required.difference(posts.columns))
        if missing:
            raise ValueError(f"Missing required columns: {', '.join(missing)}")

        frame = posts[["post_id", "created_at"]].copy()
        frame["created_at"] = pd.to_datetime(frame["created_at"], errors="coerce")
        frame = frame.dropna(subset=["post_id", "created_at"])
        frame = frame.drop_duplicates(subset=["post_id"], keep="first")
        return frame

    def _attach_timestamps(self, pairs: pd.DataFrame, timestamps: pd.DataFrame) -> pd.DataFrame:
        """Join timestamps onto both sides of each candidate pair."""

        ordered_pairs = pairs.reset_index(drop=False).rename(columns={"index": "_row_order"})
        left = ordered_pairs.merge(
            timestamps.rename(columns={"post_id": "post_id_1", "created_at": "created_at_1"}),
            on="post_id_1",
            how="inner",
            sort=False,
        )
        merged = left.merge(
            timestamps.rename(columns={"post_id": "post_id_2", "created_at": "created_at_2"}),
            on="post_id_2",
            how="inner",
            sort=False,
        )
        return merged.sort_values("_row_order", kind="stable").drop(columns=["_row_order"]).reset_index(drop=True)

    def _log_summary(self, output: pd.DataFrame, processed: int, start: float) -> None:
        """Log summary statistics for the computed temporal evidence."""

        diff = output["absolute_time_difference_seconds"]
        score = output["exponential_decay_score"]
        LOGGER.info("Candidate pairs processed: %d", processed)
        LOGGER.info("Runtime: %.2fs", perf_counter() - start)
        LOGGER.info(
            "Time difference summary: mean=%.2f, median=%.2f, max=%.2f seconds",
            float(diff.mean()),
            float(diff.median()),
            float(diff.max()),
        )
        summary = score.describe(percentiles=(0.25, 0.5, 0.75)).to_dict()
        LOGGER.info("Exponential decay score summary: %s", summary)

    def _empty_output(self) -> pd.DataFrame:
        """Return an empty output frame with the expected schema."""

        return pd.DataFrame(
            columns=[
                "post_id_1",
                "post_id_2",
                "absolute_time_difference_seconds",
                "exponential_decay_score",
            ]
        )


if __name__ == "__main__":
    sample_pairs = pd.DataFrame(
        [
            {"post_id_1": "p1", "post_id_2": "p2"},
            {"post_id_1": "p1", "post_id_2": "p3"},
        ]
    )
    sample_posts = pd.DataFrame(
        [
            {"post_id": "p1", "created_at": "2026-01-01T00:00:00"},
            {"post_id": "p2", "created_at": "2026-01-01T00:10:00"},
            {"post_id": "p3", "created_at": "2026-01-01T01:00:00"},
        ]
    )
    result = TemporalFeatureExtractor(decay_seconds=600).transform(sample_pairs, sample_posts)
    assert list(result.columns) == [
        "post_id_1",
        "post_id_2",
        "absolute_time_difference_seconds",
        "exponential_decay_score",
    ]
    assert result.loc[0, "absolute_time_difference_seconds"] == 600.0
