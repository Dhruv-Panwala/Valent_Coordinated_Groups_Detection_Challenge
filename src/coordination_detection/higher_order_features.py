"""Higher-order feature extraction for candidate pairs."""

from __future__ import annotations

import logging
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd

LOGGER = logging.getLogger(__name__)


class HigherOrderFeatureExtractor:
    """Combine previously computed evidence into higher-order behavioural features."""

    def __init__(
        self,
        *,
        post_id_column: str = "post_id",
        candidate_post_id_1_column: str = "post_id_1",
        candidate_post_id_2_column: str = "post_id_2",
    ) -> None:
        self.post_id_column = post_id_column
        self.candidate_post_id_1_column = candidate_post_id_1_column
        self.candidate_post_id_2_column = candidate_post_id_2_column

    def transform(
        self,
        temporal_features: pd.DataFrame,
        interaction_features: pd.DataFrame,
        semantic_features: pd.DataFrame,
        repeated_behavior_features: pd.DataFrame,
    ) -> pd.DataFrame:
        """Return higher-order features for each candidate pair."""

        start = perf_counter()
        base = self._prepare_base(temporal_features)
        if base.empty:
            LOGGER.info("Higher-order features processed: 0")
            LOGGER.info("Runtime: %.2fs", perf_counter() - start)
            return self._empty_output()

        merged = self._merge_feature_frames(
            base,
            interaction_features=interaction_features,
            semantic_features=semantic_features,
            repeated_behavior_features=repeated_behavior_features,
        )

        if merged.empty:
            LOGGER.info("Higher-order features processed: 0")
            LOGGER.info("Runtime: %.2fs", perf_counter() - start)
            return self._empty_output()

        output = pd.DataFrame(
            {
                self.candidate_post_id_1_column: merged[self.candidate_post_id_1_column],
                self.candidate_post_id_2_column: merged[self.candidate_post_id_2_column],
            }
        )

        cosine_similarity = merged["cosine_similarity"].fillna(0.0).astype(np.float64)
        cosine_similarity = np.maximum(cosine_similarity.to_numpy(dtype=np.float64), 0.0)
        exponential_decay_score = merged["exponential_decay_score"].fillna(0.0).astype(np.float64).to_numpy(
            dtype=np.float64
        )
        output["semantic_temporal_agreement"] = cosine_similarity * exponential_decay_score
        output["thread_temporal_agreement"] = (
            merged["same_thread"].fillna(False).astype(np.float64)
            * merged["exponential_decay_score"].fillna(0.0).astype(np.float64)
        )
        output["reply_semantic_agreement"] = merged["same_reply_target"].fillna(False).astype(np.float64) * cosine_similarity
        output["interaction_strength"] = merged[
            [
                "mention_jaccard_similarity",
                "url_jaccard_similarity",
                "hashtag_jaccard_similarity",
            ]
        ].fillna(0.0).astype(np.float64).mean(axis=1)
        output["repeated_target_agreement"] = (
            np.log1p(merged["repeated_coactivity_count"].fillna(0.0).astype(np.float64))
            * np.log1p(merged["shared_target_count"].fillna(0.0).astype(np.float64))
        )

        output = output[
            [
                self.candidate_post_id_1_column,
                self.candidate_post_id_2_column,
                "semantic_temporal_agreement",
                "thread_temporal_agreement",
                "reply_semantic_agreement",
                "interaction_strength",
                "repeated_target_agreement",
            ]
        ].reset_index(drop=True)

        self._log_summary(output, start)
        return output

    def _prepare_base(self, temporal_features: pd.DataFrame) -> pd.DataFrame:
        """Validate and normalize the base temporal feature frame."""

        required = {
            self.candidate_post_id_1_column,
            self.candidate_post_id_2_column,
            "exponential_decay_score",
        }
        missing = sorted(required.difference(temporal_features.columns))
        if missing:
            raise ValueError(f"Missing required columns: {', '.join(missing)}")

        frame = temporal_features[
            [
                self.candidate_post_id_1_column,
                self.candidate_post_id_2_column,
                "exponential_decay_score",
            ]
        ].copy()
        frame = frame.dropna(subset=[self.candidate_post_id_1_column, self.candidate_post_id_2_column])
        frame = frame.reset_index(drop=True)
        frame["_row_order"] = np.arange(len(frame), dtype=np.int64)
        return frame

    def _merge_feature_frames(
        self,
        base: pd.DataFrame,
        *,
        interaction_features: pd.DataFrame,
        semantic_features: pd.DataFrame,
        repeated_behavior_features: pd.DataFrame,
    ) -> pd.DataFrame:
        """Join the previously computed evidence tables by pair key."""

        merged = base.copy()
        merged = self._merge_frame(
            merged,
            interaction_features,
            required_columns=(
                "same_thread",
                "same_reply_target",
                "same_quoted_post",
                "mention_jaccard_similarity",
                "url_jaccard_similarity",
                "hashtag_jaccard_similarity",
            ),
        )
        merged = self._merge_frame(
            merged,
            semantic_features,
            required_columns=("cosine_similarity",),
        )
        merged = self._merge_frame(
            merged,
            repeated_behavior_features,
            required_columns=("repeated_coactivity_count", "shared_target_count"),
        )
        return merged.sort_values("_row_order", kind="stable").reset_index(drop=True)

    def _merge_frame(
        self,
        base: pd.DataFrame,
        features: pd.DataFrame,
        *,
        required_columns: tuple[str, ...],
    ) -> pd.DataFrame:
        """Left-join one feature frame onto the base pair table."""

        if features.empty:
            for column in required_columns:
                if column not in base.columns:
                    base[column] = np.nan
            return base

        required = {self.candidate_post_id_1_column, self.candidate_post_id_2_column, *required_columns}
        missing = sorted(required.difference(features.columns))
        if missing:
            raise ValueError(f"Missing required columns: {', '.join(missing)}")

        frame = features[[self.candidate_post_id_1_column, self.candidate_post_id_2_column, *required_columns]].copy()
        frame = frame.dropna(subset=[self.candidate_post_id_1_column, self.candidate_post_id_2_column])
        return base.merge(
            frame,
            on=[self.candidate_post_id_1_column, self.candidate_post_id_2_column],
            how="left",
            sort=False,
        )

    def _log_summary(self, output: pd.DataFrame, start: float) -> None:
        """Log summary statistics for each higher-order feature."""

        LOGGER.info("Higher-order features processed: %d", len(output))
        LOGGER.info("Runtime: %.2fs", perf_counter() - start)
        for column in (
            "semantic_temporal_agreement",
            "thread_temporal_agreement",
            "reply_semantic_agreement",
            "interaction_strength",
            "repeated_target_agreement",
        ):
            LOGGER.info("%s summary: %s", column, output[column].describe().to_dict())

    def _empty_output(self) -> pd.DataFrame:
        """Return an empty output frame with the expected schema."""

        return pd.DataFrame(
            columns=[
                self.candidate_post_id_1_column,
                self.candidate_post_id_2_column,
                "semantic_temporal_agreement",
                "thread_temporal_agreement",
                "reply_semantic_agreement",
                "interaction_strength",
                "repeated_target_agreement",
            ]
        )


if __name__ == "__main__":
    temporal_features = pd.DataFrame(
        [
            {"post_id_1": "p1", "post_id_2": "p2", "exponential_decay_score": 0.5},
            {"post_id_1": "p3", "post_id_2": "p4", "exponential_decay_score": 0.2},
        ]
    )
    interaction_features = pd.DataFrame(
        [
            {
                "post_id_1": "p1",
                "post_id_2": "p2",
                "same_thread": True,
                "same_reply_target": False,
                "same_quoted_post": True,
                "mention_jaccard_similarity": 0.5,
                "url_jaccard_similarity": 0.0,
                "hashtag_jaccard_similarity": 0.25,
            },
            {
                "post_id_1": "p3",
                "post_id_2": "p4",
                "same_thread": False,
                "same_reply_target": True,
                "same_quoted_post": False,
                "mention_jaccard_similarity": 0.0,
                "url_jaccard_similarity": 0.5,
                "hashtag_jaccard_similarity": 0.5,
            },
        ]
    )
    semantic_features = pd.DataFrame(
        [
            {"post_id_1": "p1", "post_id_2": "p2", "cosine_similarity": 0.8},
            {"post_id_1": "p3", "post_id_2": "p4", "cosine_similarity": 0.4},
        ]
    )
    repeated_behavior_features = pd.DataFrame(
        [
            {"post_id_1": "p1", "post_id_2": "p2", "repeated_coactivity_count": 2, "shared_target_count": 3},
            {"post_id_1": "p3", "post_id_2": "p4", "repeated_coactivity_count": 1, "shared_target_count": 1},
        ]
    )

    result = HigherOrderFeatureExtractor().transform(
        temporal_features,
        interaction_features,
        semantic_features,
        repeated_behavior_features,
    )
    assert list(result.columns) == [
        "post_id_1",
        "post_id_2",
        "semantic_temporal_agreement",
        "thread_temporal_agreement",
        "reply_semantic_agreement",
        "interaction_strength",
        "repeated_target_agreement",
    ]
    assert np.isclose(result.loc[0, "semantic_temporal_agreement"], 0.4)
