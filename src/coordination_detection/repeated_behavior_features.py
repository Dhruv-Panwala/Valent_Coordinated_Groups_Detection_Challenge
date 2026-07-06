"""Repeated behavioural feature extraction for candidate pairs."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class _TargetSpec:
    """Configuration for a shared interaction target field."""

    column_name: str
    label: str


class RepeatedBehaviorFeatureExtractor:
    """Compute repeated behavioural evidence for candidate account pairs."""

    def __init__(
        self,
        *,
        post_id_column: str = "post_id",
        account_id_column: str = "account_id",
        candidate_post_id_1_column: str = "post_id_1",
        candidate_post_id_2_column: str = "post_id_2",
        candidate_account_id_1_column: str = "account_id_1",
        candidate_account_id_2_column: str = "account_id_2",
        thread_id_column: str = "thread_id",
        reply_to_post_id_column: str = "reply_to_post_id",
        quoted_post_id_column: str = "quoted_post_id",
    ) -> None:
        self.post_id_column = post_id_column
        self.account_id_column = account_id_column
        self.candidate_post_id_1_column = candidate_post_id_1_column
        self.candidate_post_id_2_column = candidate_post_id_2_column
        self.candidate_account_id_1_column = candidate_account_id_1_column
        self.candidate_account_id_2_column = candidate_account_id_2_column
        self._target_specs = (
            _TargetSpec(thread_id_column, "thread_id"),
            _TargetSpec(reply_to_post_id_column, "reply_to_post_id"),
            _TargetSpec(quoted_post_id_column, "quoted_post_id"),
        )

    def transform(self, candidate_pairs: pd.DataFrame, posts: pd.DataFrame) -> pd.DataFrame:
        """Return repeated behaviour features for each candidate pair."""

        start = perf_counter()
        pairs = self._prepare_candidate_pairs(candidate_pairs)
        if pairs.empty:
            LOGGER.info("Repeated behaviour features processed: 0")
            LOGGER.info("Runtime: %.2fs", perf_counter() - start)
            return self._empty_output()

        lookups = self._build_account_target_lookups(posts)
        repeated_counts = self._repeated_coactivity_counts(pairs)
        shared_target_counts = self._shared_target_counts(pairs, lookups)

        output = pd.DataFrame(
            {
                self.candidate_post_id_1_column: pairs[self.candidate_post_id_1_column],
                self.candidate_post_id_2_column: pairs[self.candidate_post_id_2_column],
                "repeated_coactivity_count": repeated_counts,
                "shared_target_count": shared_target_counts,
            }
        ).reset_index(drop=True)

        self._log_summary(output, start)
        return output

    def _prepare_candidate_pairs(self, candidate_pairs: pd.DataFrame) -> pd.DataFrame:
        """Validate and normalize the candidate-pair frame."""

        required = {
            self.candidate_post_id_1_column,
            self.candidate_post_id_2_column,
            self.candidate_account_id_1_column,
            self.candidate_account_id_2_column,
        }
        missing = sorted(required.difference(candidate_pairs.columns))
        if missing:
            raise ValueError(f"Missing required columns: {', '.join(missing)}")

        frame = candidate_pairs[
            [
                self.candidate_post_id_1_column,
                self.candidate_post_id_2_column,
                self.candidate_account_id_1_column,
                self.candidate_account_id_2_column,
            ]
        ].copy()
        frame = frame.dropna(
            subset=[
                self.candidate_post_id_1_column,
                self.candidate_post_id_2_column,
                self.candidate_account_id_1_column,
                self.candidate_account_id_2_column,
            ]
        )
        frame = frame.reset_index(drop=True)
        frame["_row_order"] = np.arange(len(frame), dtype=np.int64)
        frame["_account_pair_key"] = [
            self._canonical_pair(left, right)
            for left, right in zip(
                frame[self.candidate_account_id_1_column],
                frame[self.candidate_account_id_2_column],
            )
        ]
        return frame

    def _build_account_target_lookups(self, posts: pd.DataFrame) -> dict[Any, frozenset[Any]]:
        """Precompute per-account target sets across thread, reply, and quote targets."""

        required = {self.post_id_column, self.account_id_column}
        missing = sorted(required.difference(posts.columns))
        if missing:
            raise ValueError(f"Missing required columns: {', '.join(missing)}")

        target_frames: list[pd.DataFrame] = []
        for spec in self._target_specs:
            if spec.column_name not in posts.columns:
                continue
            frame = posts[[self.account_id_column, spec.column_name]].copy()
            frame = frame.dropna(subset=[self.account_id_column, spec.column_name])
            if frame.empty:
                continue
            frame = frame.drop_duplicates(subset=[self.account_id_column, spec.column_name], keep="first")
            target_frames.append(
                frame.rename(columns={spec.column_name: "target_value"})[[self.account_id_column, "target_value"]]
            )
        if not target_frames:
            return {}

        combined = pd.concat(target_frames, ignore_index=True)
        combined = combined.drop_duplicates(subset=[self.account_id_column, "target_value"], keep="first")
        grouped = combined.groupby(self.account_id_column, sort=True)["target_value"].agg(lambda series: frozenset(series))
        return grouped.to_dict()

    def _repeated_coactivity_counts(self, pairs: pd.DataFrame) -> pd.Series:
        """Count how many times each account pair appears in the candidate set."""

        counts = pairs.groupby("_account_pair_key", sort=True).size().rename("repeated_coactivity_count")
        return pairs["_account_pair_key"].map(counts).astype(np.int64)

    def _shared_target_counts(
        self,
        pairs: pd.DataFrame,
        lookups: dict[Any, frozenset[Any]],
    ) -> pd.Series:
        """Count shared interaction targets across thread, reply, and quote targets."""

        if not lookups:
            return pd.Series(np.zeros(len(pairs), dtype=np.int64), index=pairs.index)

        empty: frozenset[Any] = frozenset()
        account_1 = pairs[self.candidate_account_id_1_column]
        account_2 = pairs[self.candidate_account_id_2_column]

        left_sets = account_1.map(lambda value, lookup=lookups: lookup.get(value, empty))
        right_sets = account_2.map(lambda value, lookup=lookups: lookup.get(value, empty))
        counts = np.fromiter(
            (len(left_set.intersection(right_set)) for left_set, right_set in zip(left_sets, right_sets)),
            dtype=np.int64,
            count=len(pairs),
        )
        return pd.Series(counts, index=pairs.index)

    def _canonical_pair(self, left: Any, right: Any) -> tuple[Any, Any]:
        """Return a stable ordering for an unordered pair."""

        left_key = str(left)
        right_key = str(right)
        if left_key <= right_key:
            return left, right
        return right, left

    def _log_summary(self, output: pd.DataFrame, start: float) -> None:
        """Log repeated-behaviour summary statistics."""

        LOGGER.info("Repeated behaviour features processed: %d", len(output))
        LOGGER.info("Runtime: %.2fs", perf_counter() - start)
        LOGGER.info(
            "Repeated coactivity count summary: %s",
            output["repeated_coactivity_count"].describe().to_dict(),
        )
        LOGGER.info(
            "Shared target count summary: %s",
            output["shared_target_count"].describe().to_dict(),
        )

    def _empty_output(self) -> pd.DataFrame:
        """Return an empty output frame with the expected schema."""

        return pd.DataFrame(
            columns=[
                self.candidate_post_id_1_column,
                self.candidate_post_id_2_column,
                "repeated_coactivity_count",
                "shared_target_count",
            ]
        )


if __name__ == "__main__":
    sample_candidates = pd.DataFrame(
        [
            {"post_id_1": "p1", "post_id_2": "p2", "account_id_1": "a1", "account_id_2": "a2"},
            {"post_id_1": "p3", "post_id_2": "p4", "account_id_1": "a1", "account_id_2": "a2"},
            {"post_id_1": "p5", "post_id_2": "p6", "account_id_1": "a1", "account_id_2": "a3"},
        ]
    )
    sample_posts = pd.DataFrame(
        [
            {"post_id": "p1", "account_id": "a1", "thread_id": "t1", "reply_to_post_id": "r1", "quoted_post_id": "q1"},
            {"post_id": "p2", "account_id": "a2", "thread_id": "t1", "reply_to_post_id": "r2", "quoted_post_id": "q1"},
            {"post_id": "p3", "account_id": "a1", "thread_id": "t2", "reply_to_post_id": "r1", "quoted_post_id": "q2"},
            {"post_id": "p4", "account_id": "a2", "thread_id": "t3", "reply_to_post_id": "r3", "quoted_post_id": "q2"},
            {"post_id": "p5", "account_id": "a1", "thread_id": "t4", "reply_to_post_id": "r4", "quoted_post_id": "q3"},
            {"post_id": "p6", "account_id": "a3", "thread_id": "t4", "reply_to_post_id": "r4", "quoted_post_id": "q4"},
        ]
    )
    result = RepeatedBehaviorFeatureExtractor().transform(sample_candidates, sample_posts)
    assert list(result.columns) == [
        "post_id_1",
        "post_id_2",
        "repeated_coactivity_count",
        "shared_target_count",
    ]
    assert result.loc[0, "repeated_coactivity_count"] == 2
