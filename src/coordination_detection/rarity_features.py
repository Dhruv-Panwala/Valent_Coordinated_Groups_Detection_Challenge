"""Rarity feature extraction for candidate pairs."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from math import log
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class _FieldSpec:
    """Configuration for one rarity-bearing interaction field."""

    source_column: str
    output_column: str
    is_list_field: bool


class RarityFeatureExtractor:
    """Compute rarity evidence from global interaction frequencies."""

    def __init__(
        self,
        *,
        post_id_column: str = "post_id",
        candidate_post_id_1_column: str = "post_id_1",
        candidate_post_id_2_column: str = "post_id_2",
        thread_id_column: str = "thread_id",
        reply_to_post_id_column: str = "reply_to_post_id",
        mentions_column: str = "mentions",
        urls_column: str = "urls",
        hashtags_column: str = "hashtags",
    ) -> None:
        self.post_id_column = post_id_column
        self.candidate_post_id_1_column = candidate_post_id_1_column
        self.candidate_post_id_2_column = candidate_post_id_2_column
        self._fields = (
            _FieldSpec(thread_id_column, "thread_rarity", False),
            _FieldSpec(reply_to_post_id_column, "reply_target_rarity", False),
            _FieldSpec(mentions_column, "mention_rarity", True),
            _FieldSpec(urls_column, "url_rarity", True),
            _FieldSpec(hashtags_column, "hashtag_rarity", True),
        )

    def transform(self, candidate_pairs: pd.DataFrame, posts: pd.DataFrame) -> pd.DataFrame:
        """Return rarity features for each candidate pair."""

        start = perf_counter()
        pairs = self._prepare_candidate_pairs(candidate_pairs)
        if pairs.empty:
            LOGGER.info("Rarity features processed: 0")
            LOGGER.info("Runtime: %.2fs", perf_counter() - start)
            return self._empty_output()

        post_features, frequency_stats, total_posts = self._build_post_features(posts)
        if not post_features:
            LOGGER.info("Rarity features processed: 0")
            LOGGER.info("Runtime: %.2fs", perf_counter() - start)
            return self._empty_output()

        merged = self._attach_post_features(pairs, post_features)
        if merged.empty:
            LOGGER.info("Rarity features processed: 0")
            LOGGER.info("Runtime: %.2fs", perf_counter() - start)
            return self._empty_output()

        output = pd.DataFrame(
            {
                self.candidate_post_id_1_column: merged[self.candidate_post_id_1_column],
                self.candidate_post_id_2_column: merged[self.candidate_post_id_2_column],
            }
        )

        rarity_outputs: dict[str, np.ndarray] = {}
        for spec in self._fields:
            rarity_outputs[spec.output_column] = self._pair_rarity_series(merged, spec, total_posts=total_posts)
            output[spec.output_column] = rarity_outputs[spec.output_column]

        output = output[
            [
                self.candidate_post_id_1_column,
                self.candidate_post_id_2_column,
                "mention_rarity",
                "url_rarity",
                "hashtag_rarity",
                "thread_rarity",
                "reply_target_rarity",
            ]
        ]
        self._log_summary(output, frequency_stats, start)
        return output.reset_index(drop=True)

    def _prepare_candidate_pairs(self, candidate_pairs: pd.DataFrame) -> pd.DataFrame:
        """Validate and normalize candidate-pair input."""

        required = {self.candidate_post_id_1_column, self.candidate_post_id_2_column}
        missing = sorted(required.difference(candidate_pairs.columns))
        if missing:
            raise ValueError(f"Missing required columns: {', '.join(missing)}")

        frame = candidate_pairs[[self.candidate_post_id_1_column, self.candidate_post_id_2_column]].copy()
        frame = frame.dropna(subset=[self.candidate_post_id_1_column, self.candidate_post_id_2_column])
        frame = frame.reset_index(drop=True)
        frame["_row_order"] = np.arange(len(frame), dtype=np.int64)
        return frame

    def _build_post_features(
        self,
        posts: pd.DataFrame,
    ) -> tuple[dict[str, pd.DataFrame], dict[str, dict[str, Any]], int]:
        """Build per-post lookup tables and global frequency statistics."""

        if self.post_id_column not in posts.columns:
            raise ValueError(f"Missing required column: {self.post_id_column}")

        total_posts = int(posts[self.post_id_column].dropna().nunique())
        post_features: dict[str, pd.DataFrame] = {}
        frequency_stats: dict[str, dict[str, Any]] = {}

        for spec in self._fields:
            if spec.source_column not in posts.columns:
                continue
            feature_frame, stats = self._build_feature_frame(posts, spec, total_posts=total_posts)
            post_features[spec.output_column] = feature_frame
            frequency_stats[spec.output_column] = stats

        return post_features, frequency_stats, total_posts

    def _build_feature_frame(
        self,
        posts: pd.DataFrame,
        spec: _FieldSpec,
        *,
        total_posts: int,
    ) -> tuple[pd.DataFrame, dict[str, Any]]:
        """Compute global frequency maps and per-post feature sets for one field."""

        frame = posts[[self.post_id_column, spec.source_column]].copy()
        frame = frame.dropna(subset=[self.post_id_column])
        if spec.is_list_field:
            frame[spec.source_column] = frame[spec.source_column].map(self._to_value_list)
            frame = frame.explode(spec.source_column, ignore_index=True)
            frame = frame.dropna(subset=[spec.source_column])
            frame = frame.drop_duplicates(subset=[self.post_id_column, spec.source_column], keep="first")
        else:
            frame = frame.drop_duplicates(subset=[self.post_id_column], keep="first")

        if frame.empty:
            empty = pd.DataFrame(columns=[self.post_id_column, "values", "frequencies"])
            return empty, {
                "total_posts": total_posts,
                "unique_items": 0,
                "min_frequency": 0,
                "max_frequency": 0,
                "mean_frequency": 0.0,
            }

        if spec.is_list_field:
            frequency = frame.groupby(spec.source_column, sort=True)[self.post_id_column].nunique()
            values = (
                frame.groupby(self.post_id_column, sort=True)[spec.source_column]
                .agg(lambda series: tuple(sorted(set(series), key=str)))
                .rename("values")
                .reset_index()
            )
            values["frequencies"] = values["values"].map(
                lambda tokens: tuple(int(frequency.get(token, 0)) for token in tokens)
            )
            stats = self._frequency_stats(frequency, total_posts=total_posts)
            return values, stats

        frequency = frame.groupby(spec.source_column, sort=True)[self.post_id_column].nunique()
        values = frame.rename(columns={spec.source_column: "values"})[[self.post_id_column, "values"]]
        values["frequencies"] = values["values"].map(lambda token: (int(frequency.get(token, 0)),))
        stats = self._frequency_stats(frequency, total_posts=total_posts)
        return values, stats

    def _attach_post_features(
        self,
        pairs: pd.DataFrame,
        post_features: dict[str, pd.DataFrame],
    ) -> pd.DataFrame:
        """Join per-post feature sets onto both sides of each candidate pair."""

        merged = pairs.copy()
        for spec in self._fields:
            feature_frame = post_features.get(spec.output_column)
            if feature_frame is None or feature_frame.empty:
                continue
            left = feature_frame.rename(
                columns={
                    self.post_id_column: self.candidate_post_id_1_column,
                    "values": f"{spec.output_column}_values_1",
                    "frequencies": f"{spec.output_column}_frequencies_1",
                }
            )
            right = feature_frame.rename(
                columns={
                    self.post_id_column: self.candidate_post_id_2_column,
                    "values": f"{spec.output_column}_values_2",
                    "frequencies": f"{spec.output_column}_frequencies_2",
                }
            )
            merged = merged.merge(left, on=self.candidate_post_id_1_column, how="left", sort=False)
            merged = merged.merge(right, on=self.candidate_post_id_2_column, how="left", sort=False)

        return merged.sort_values("_row_order", kind="stable").reset_index(drop=True)

    def _pair_rarity_series(self, merged: pd.DataFrame, spec: _FieldSpec, *, total_posts: int) -> np.ndarray:
        """Compute mean rarity for shared items on one field."""

        left_values = f"{spec.output_column}_values_1"
        right_values = f"{spec.output_column}_values_2"
        left_freqs = f"{spec.output_column}_frequencies_1"
        right_freqs = f"{spec.output_column}_frequencies_2"

        if left_values not in merged.columns or right_values not in merged.columns:
            return np.zeros(len(merged), dtype=np.float64)

        return np.array(
            [
                self._shared_mean_rarity(
                    left_tokens,
                    right_tokens,
                    left_token_freqs,
                    right_token_freqs,
                    total_posts=total_posts,
                )
                for left_tokens, right_tokens, left_token_freqs, right_token_freqs in zip(
                    merged[left_values],
                    merged[right_values],
                    merged[left_freqs],
                    merged[right_freqs],
                )
            ],
            dtype=np.float64,
        )

    def _shared_mean_rarity(
        self,
        left_tokens: Any,
        right_tokens: Any,
        left_freqs: Any,
        right_freqs: Any,
        *,
        total_posts: int,
    ) -> float:
        """Return the mean rarity for shared interaction values."""

        if self._is_missing(left_tokens) or self._is_missing(right_tokens):
            return 0.0

        left_token_list = tuple(left_tokens) if isinstance(left_tokens, tuple) else (left_tokens,)
        right_token_list = tuple(right_tokens) if isinstance(right_tokens, tuple) else (right_tokens,)
        left_freq_list = tuple(left_freqs) if isinstance(left_freqs, tuple) else (left_freqs,)
        right_freq_list = tuple(right_freqs) if isinstance(right_freqs, tuple) else (right_freqs,)

        left_map = dict(zip(left_token_list, left_freq_list, strict=False))
        right_map = dict(zip(right_token_list, right_freq_list, strict=False))
        shared_tokens = sorted(set(left_map).intersection(right_map), key=str)
        if not shared_tokens:
            return 0.0

        rarities = [log((total_posts + 1.0) / (float(max(left_map[token], right_map[token])) + 1.0)) for token in shared_tokens]
        return float(np.mean(rarities)) if rarities else 0.0

    def _frequency_stats(self, frequency: pd.Series, *, total_posts: int) -> dict[str, Any]:
        """Summarize global frequency values for logging."""

        return {
            "total_posts": total_posts,
            "unique_items": int(frequency.shape[0]),
            "min_frequency": int(frequency.min()),
            "max_frequency": int(frequency.max()),
            "mean_frequency": float(frequency.mean()),
        }

    def _log_summary(
        self,
        output: pd.DataFrame,
        frequency_stats: dict[str, dict[str, Any]],
        start: float,
    ) -> None:
        """Log frequency and rarity summaries."""

        LOGGER.info("Rarity features processed: %d", len(output))
        LOGGER.info("Runtime: %.2fs", perf_counter() - start)
        LOGGER.info("Frequency statistics: %s", frequency_stats)
        LOGGER.info("Rarity statistics: %s", output.drop(columns=[self.candidate_post_id_1_column, self.candidate_post_id_2_column]).describe().to_dict())

    def _to_value_list(self, value: Any) -> list[Any]:
        """Normalize list-like values into a clean list."""

        if self._is_missing(value):
            return []
        if isinstance(value, list):
            values = value
        elif isinstance(value, (tuple, set)):
            values = list(value)
        else:
            return [value]
        return [item for item in values if not self._is_missing(item)]

    def _is_missing(self, value: Any) -> bool:
        """Return True when the value should be ignored."""

        if value is None:
            return True
        try:
            return bool(pd.isna(value))
        except (TypeError, ValueError):
            return False

    def _empty_output(self) -> pd.DataFrame:
        """Return an empty output frame with the expected schema."""

        return pd.DataFrame(
            columns=[
                self.candidate_post_id_1_column,
                self.candidate_post_id_2_column,
                "mention_rarity",
                "url_rarity",
                "hashtag_rarity",
                "thread_rarity",
                "reply_target_rarity",
            ]
        )


if __name__ == "__main__":
    sample_candidates = pd.DataFrame(
        [
            {"post_id_1": "p1", "post_id_2": "p2"},
            {"post_id_1": "p1", "post_id_2": "p3"},
        ]
    )
    sample_posts = pd.DataFrame(
        [
            {
                "post_id": "p1",
                "thread_id": "t1",
                "reply_to_post_id": "r1",
                "mentions": ["@a", "@b"],
                "urls": ["u1"],
                "hashtags": ["#x", "#y"],
            },
            {
                "post_id": "p2",
                "thread_id": "t1",
                "reply_to_post_id": "r2",
                "mentions": ["@b"],
                "urls": ["u1", "u2"],
                "hashtags": [],
            },
            {
                "post_id": "p3",
                "thread_id": "t2",
                "reply_to_post_id": "r1",
                "mentions": [],
                "urls": [],
                "hashtags": ["#y"],
            },
        ]
    )
    result = RarityFeatureExtractor().transform(sample_candidates, sample_posts)
    assert list(result.columns) == [
        "post_id_1",
        "post_id_2",
        "mention_rarity",
        "url_rarity",
        "hashtag_rarity",
        "thread_rarity",
        "reply_target_rarity",
    ]
    assert result.loc[0, "thread_rarity"] > 0
