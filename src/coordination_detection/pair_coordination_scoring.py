"""Pair-level coordination scoring for candidate pairs."""

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
    from coordination_detection.config import PairCoordinationScoringConfig
else:
    from .config import PairCoordinationScoringConfig

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class _FrameSpec:
    """Configuration for one scored evidence frame."""

    name: str
    required_columns: tuple[str, ...]


class PairCoordinationScorer:
    """Fuse previously computed evidence into a pair-level coordination score."""

    def __init__(self, config: PairCoordinationScoringConfig | None = None) -> None:
        self.config = config or PairCoordinationScoringConfig()
        self._frame_specs = (
            _FrameSpec("temporal", (self.config.temporal_decay_column,)),
            _FrameSpec(
                "interaction",
                (
                    self.config.same_thread_column,
                    self.config.same_reply_target_column,
                    self.config.same_quoted_post_column,
                    self.config.mention_jaccard_similarity_column,
                    self.config.url_jaccard_similarity_column,
                    self.config.hashtag_jaccard_similarity_column,
                ),
            ),
            _FrameSpec("semantic", (self.config.semantic_similarity_column,)),
            _FrameSpec(
                "rarity",
                (
                    self.config.mention_rarity_column,
                    self.config.url_rarity_column,
                    self.config.hashtag_rarity_column,
                    self.config.thread_rarity_column,
                    self.config.reply_target_rarity_column,
                ),
            ),
            _FrameSpec(
                "repeated",
                (
                    self.config.repeated_coactivity_count_column,
                    self.config.shared_target_count_column,
                ),
            ),
            _FrameSpec(
                "higher_order",
                (
                    self.config.semantic_temporal_agreement_column,
                    self.config.thread_temporal_agreement_column,
                    self.config.reply_semantic_agreement_column,
                    self.config.interaction_strength_column,
                    self.config.repeated_target_agreement_column,
                ),
            ),
        )

    def transform(
        self,
        temporal_features: pd.DataFrame,
        interaction_features: pd.DataFrame,
        semantic_features: pd.DataFrame,
        rarity_features: pd.DataFrame,
        repeated_behavior_features: pd.DataFrame,
        higher_order_features: pd.DataFrame,
    ) -> pd.DataFrame:
        """Return a fused coordination score for each candidate pair."""

        start = perf_counter()
        merged = self._merge_frames(
            temporal_features,
            interaction_features,
            semantic_features,
            rarity_features,
            repeated_behavior_features,
            higher_order_features,
        )
        if merged.empty:
            LOGGER.info("Coordination scores processed: 0")
            LOGGER.info("Runtime: %.2fs", perf_counter() - start)
            return self._empty_output()

        components = self._build_components(merged)
        weight_map = self._weight_map()
        score = self._weighted_average(components)
        diagnostics = self._build_diagnostics_frame(merged, components, score, weight_map)
        output = pd.DataFrame(
            {
                self.config.candidate_post_id_1_column: merged[self.config.candidate_post_id_1_column],
                self.config.candidate_post_id_2_column: merged[self.config.candidate_post_id_2_column],
                "coordination_score": np.clip(score, 0.0, 1.0),
            }
        ).sort_values(
            [self.config.candidate_post_id_1_column, self.config.candidate_post_id_2_column],
            kind="stable",
        ).reset_index(drop=True)

        self._log_summary(output, start)
        self._log_diagnostics(diagnostics, weight_map, start)
        return output

    def _merge_frames(
        self,
        temporal_features: pd.DataFrame,
        interaction_features: pd.DataFrame,
        semantic_features: pd.DataFrame,
        rarity_features: pd.DataFrame,
        repeated_behavior_features: pd.DataFrame,
        higher_order_features: pd.DataFrame,
    ) -> pd.DataFrame:
        """Merge all evidence tables by candidate pair."""

        frames = [
            self._normalize_frame(temporal_features, self._frame_specs[0]),
            self._normalize_frame(interaction_features, self._frame_specs[1]),
            self._normalize_frame(semantic_features, self._frame_specs[2]),
            self._normalize_frame(rarity_features, self._frame_specs[3]),
            self._normalize_frame(repeated_behavior_features, self._frame_specs[4]),
            self._normalize_frame(higher_order_features, self._frame_specs[5]),
        ]
        frames = [frame for frame in frames if not frame.empty]
        if not frames:
            return pd.DataFrame()

        merged = frames[0]
        for frame in frames[1:]:
            merged = merged.merge(
                frame,
                on=[self.config.candidate_post_id_1_column, self.config.candidate_post_id_2_column],
                how="outer",
                sort=False,
            )
        return merged.drop_duplicates(
            subset=[self.config.candidate_post_id_1_column, self.config.candidate_post_id_2_column],
            keep="first",
        ).reset_index(drop=True)

    def _normalize_frame(self, frame: pd.DataFrame, spec: _FrameSpec) -> pd.DataFrame:
        """Validate and retain only the columns needed from one evidence frame."""

        if frame.empty:
            return pd.DataFrame()

        required = {
            self.config.candidate_post_id_1_column,
            self.config.candidate_post_id_2_column,
            *spec.required_columns,
        }
        missing = sorted(required.difference(frame.columns))
        if missing:
            raise ValueError(f"Missing required columns in {spec.name} features: {', '.join(missing)}")

        columns = [self.config.candidate_post_id_1_column, self.config.candidate_post_id_2_column, *spec.required_columns]
        normalized = frame[columns].copy()
        normalized = normalized.dropna(subset=[self.config.candidate_post_id_1_column, self.config.candidate_post_id_2_column])
        return normalized.drop_duplicates(
            subset=[self.config.candidate_post_id_1_column, self.config.candidate_post_id_2_column],
            keep="first",
        )

    def _build_components(self, merged: pd.DataFrame) -> dict[str, np.ndarray]:
        """Build normalized component scores ready for weighted fusion."""

        components: dict[str, np.ndarray] = {}
        temporal = self._series_or_zeros(merged, self.config.temporal_decay_column)
        semantic = self._clamp_non_negative(self._series_or_zeros(merged, self.config.semantic_similarity_column))
        higher_semantic_temporal = self._series_or_zeros(merged, self.config.semantic_temporal_agreement_column)
        higher_thread_temporal = self._series_or_zeros(merged, self.config.thread_temporal_agreement_column)
        higher_reply_semantic = self._series_or_zeros(merged, self.config.reply_semantic_agreement_column)
        higher_interaction = self._series_or_zeros(merged, self.config.interaction_strength_column)
        higher_repeated_target = self._normalize_positive(self._series_or_zeros(merged, self.config.repeated_target_agreement_column))

        if self.config.semantic_temporal_agreement_column in merged.columns:
            components["semantic_temporal"] = self._normalize_positive(higher_semantic_temporal)
        else:
            components["semantic_temporal"] = self._normalize_positive(semantic * temporal)

        if self.config.thread_temporal_agreement_column in merged.columns:
            components["thread_temporal"] = self._normalize_positive(higher_thread_temporal)
        else:
            components["thread_temporal"] = self._normalize_positive(
                self._boolean_to_float(merged, self.config.same_thread_column) * temporal
            )

        if self.config.reply_semantic_agreement_column in merged.columns:
            components["reply_semantic"] = self._normalize_positive(higher_reply_semantic)
        else:
            components["reply_semantic"] = self._normalize_positive(
                self._boolean_to_float(merged, self.config.same_reply_target_column) * semantic
            )

        if self.config.interaction_strength_column in merged.columns:
            components["interaction"] = self._normalize_positive(higher_interaction)
        else:
            components["interaction"] = self._interaction_strength_from_raw(merged)

        if self.config.repeated_target_agreement_column in merged.columns:
            components["repeated_target"] = higher_repeated_target
        else:
            repeated_strength = self._normalize_count_pair(
                merged,
                self.config.repeated_coactivity_count_column,
                self.config.shared_target_count_column,
            )
            components["repeated_target"] = self._normalize_positive(repeated_strength)

        components["temporal"] = self._normalize_positive(temporal)
        components["semantic"] = self._normalize_positive(semantic)
        components["rarity"] = self._rarity_strength(merged)
        components["repeated"] = self._normalize_count_pair(
            merged,
            self.config.repeated_coactivity_count_column,
            self.config.shared_target_count_column,
        )
        return components

    def _weighted_average(self, components: dict[str, np.ndarray]) -> np.ndarray:
        """Return the weighted average of available components."""

        weight_map = self._weight_map()

        score = np.zeros(len(next(iter(components.values()))), dtype=np.float64)
        total_weight = np.zeros(len(score), dtype=np.float64)
        for name, values in components.items():
            weight = float(weight_map[name])
            if weight <= 0:
                continue
            score += values * weight
            total_weight += weight

        return np.divide(score, total_weight, out=np.zeros_like(score), where=total_weight > 0)

    def _weight_map(self) -> dict[str, float]:
        """Return the configured feature-group weights."""

        return {
            "temporal": self.config.temporal_weight,
            "semantic": self.config.semantic_weight,
            "interaction": self.config.interaction_weight,
            "rarity": self.config.rarity_weight,
            "repeated": self.config.repeated_weight,
            "semantic_temporal": self.config.semantic_temporal_weight,
            "thread_temporal": self.config.thread_temporal_weight,
            "reply_semantic": self.config.reply_semantic_weight,
            "repeated_target": self.config.repeated_target_weight,
        }

    def _build_diagnostics_frame(
        self,
        merged: pd.DataFrame,
        components: dict[str, np.ndarray],
        score: np.ndarray,
        weight_map: dict[str, float],
    ) -> pd.DataFrame:
        """Build a diagnostic table with per-group contributions."""

        diagnostics = pd.DataFrame(
            {
                self.config.candidate_post_id_1_column: merged[self.config.candidate_post_id_1_column],
                self.config.candidate_post_id_2_column: merged[self.config.candidate_post_id_2_column],
                "coordination_score": np.clip(score, 0.0, 1.0),
            }
        )
        total_weight = float(sum(weight for weight in weight_map.values() if weight > 0))
        if total_weight <= 0.0:
            for column in (
                "temporal_contribution",
                "interaction_contribution",
                "semantic_contribution",
                "rarity_contribution",
                "repeated_behavior_contribution",
                "higher_order_contribution",
            ):
                diagnostics[column] = 0.0
            return diagnostics

        diagnostics["temporal_contribution"] = components["temporal"] * weight_map["temporal"] / total_weight
        diagnostics["interaction_contribution"] = components["interaction"] * weight_map["interaction"] / total_weight
        diagnostics["semantic_contribution"] = components["semantic"] * weight_map["semantic"] / total_weight
        diagnostics["rarity_contribution"] = components["rarity"] * weight_map["rarity"] / total_weight
        diagnostics["repeated_behavior_contribution"] = components["repeated"] * weight_map["repeated"] / total_weight
        diagnostics["higher_order_contribution"] = (
            components["semantic_temporal"] * weight_map["semantic_temporal"]
            + components["thread_temporal"] * weight_map["thread_temporal"]
            + components["reply_semantic"] * weight_map["reply_semantic"]
            + components["repeated_target"] * weight_map["repeated_target"]
        ) / total_weight

        component_columns = {
            "temporal": components["temporal"],
            "semantic": components["semantic"],
            "interaction": components["interaction"],
            "rarity": components["rarity"],
            "repeated": components["repeated"],
            "semantic_temporal": components["semantic_temporal"],
            "thread_temporal": components["thread_temporal"],
            "reply_semantic": components["reply_semantic"],
            "repeated_target": components["repeated_target"],
        }
        for name, values in component_columns.items():
            diagnostics[f"{name}_normalized"] = values

        return diagnostics

    def _interaction_strength_from_raw(self, merged: pd.DataFrame) -> np.ndarray:
        """Compute interaction strength from raw interaction similarity columns."""

        columns = [
            self.config.mention_jaccard_similarity_column,
            self.config.url_jaccard_similarity_column,
            self.config.hashtag_jaccard_similarity_column,
        ]
        frame = merged.reindex(columns=columns, fill_value=0.0).fillna(0.0).astype(np.float64)
        return frame.mean(axis=1).to_numpy(dtype=np.float64)

    def _rarity_strength(self, merged: pd.DataFrame) -> np.ndarray:
        """Aggregate normalized rarity evidence."""

        columns = [
            self.config.mention_rarity_column,
            self.config.url_rarity_column,
            self.config.hashtag_rarity_column,
            self.config.thread_rarity_column,
            self.config.reply_target_rarity_column,
        ]
        frame = merged.reindex(columns=columns, fill_value=0.0).fillna(0.0).astype(np.float64)
        normalized = self._normalize_positive(frame.to_numpy(dtype=np.float64))
        return normalized.mean(axis=1) if normalized.size else np.zeros(len(merged), dtype=np.float64)

    def _normalize_count_pair(self, merged: pd.DataFrame, left_column: str, right_column: str) -> np.ndarray:
        """Normalize a pair of count-based columns using log1p scaling."""

        left = self._series_or_zeros(merged, left_column)
        right = self._series_or_zeros(merged, right_column)
        left_norm = self._normalize_count(left)
        right_norm = self._normalize_count(right)
        return (left_norm + right_norm) / 2.0

    def _normalize_count(self, values: np.ndarray) -> np.ndarray:
        """Scale count-like values into [0, 1] using log1p normalization."""

        values = np.clip(values, 0.0, None)
        logged = np.log1p(values)
        maximum = float(logged.max(initial=0.0))
        if maximum <= 0.0:
            return np.zeros_like(logged, dtype=np.float64)
        return logged / maximum

    def _normalize_positive(self, values: np.ndarray) -> np.ndarray:
        """Map non-negative values to [0, 1] without changing order."""

        values = np.clip(values, 0.0, None)
        return values / (1.0 + values)

    def _clamp_non_negative(self, values: np.ndarray) -> np.ndarray:
        """Clamp values at zero."""

        return np.maximum(values, 0.0)

    def _boolean_to_float(self, merged: pd.DataFrame, column: str) -> np.ndarray:
        """Convert a boolean column to float values."""

        if column not in merged.columns:
            return np.zeros(len(merged), dtype=np.float64)
        return merged[column].fillna(False).astype(np.float64).to_numpy(dtype=np.float64)

    def _series_or_zeros(self, merged: pd.DataFrame, column: str) -> np.ndarray:
        """Return a numeric series or zeros when the column is absent."""

        if column not in merged.columns:
            return np.zeros(len(merged), dtype=np.float64)
        return merged[column].fillna(0.0).astype(np.float64).to_numpy(dtype=np.float64)

    def _log_summary(self, output: pd.DataFrame, start: float) -> None:
        """Log score distribution, runtime, and threshold recommendations."""

        score = output["coordination_score"].astype(np.float64)
        LOGGER.info("Coordination scores processed: %d", len(output))
        LOGGER.info("Runtime: %.2fs", perf_counter() - start)
        LOGGER.info("Score distribution: %s", score.describe(percentiles=(0.25, 0.5, 0.75, 0.9, 0.95)).to_dict())
        recommendations = {
            f"quantile_{int(quantile * 100)}": float(score.quantile(quantile))
            for quantile in self.config.recommendation_quantiles
        }
        LOGGER.info("Threshold recommendations: %s", recommendations)

    def _log_diagnostics(self, diagnostics: pd.DataFrame, weight_map: dict[str, float], start: float) -> None:
        """Log pair-level and group-level diagnostics for the scoring output."""

        if diagnostics.empty:
            return

        score_column = "coordination_score"
        total_weight = float(sum(weight for weight in weight_map.values() if weight > 0))
        if total_weight <= 0.0:
            LOGGER.info("Diagnostics skipped because all feature weights are non-positive.")
            return

        ordered = diagnostics.sort_values(
            [score_column, self.config.candidate_post_id_1_column, self.config.candidate_post_id_2_column],
            ascending=[False, True, True],
            kind="stable",
        ).reset_index(drop=True)

        top_n = ordered.head(20).copy()
        if not top_n.empty:
            LOGGER.info(
                "Top 20 highest-scoring pairs with group contributions:\n%s",
                top_n[
                    [
                        self.config.candidate_post_id_1_column,
                        self.config.candidate_post_id_2_column,
                        score_column,
                        "temporal_contribution",
                        "interaction_contribution",
                        "semantic_contribution",
                        "rarity_contribution",
                        "repeated_behavior_contribution",
                        "higher_order_contribution",
                    ]
                ].to_string(index=False),
            )

        component_names = (
            "temporal",
            "semantic",
            "interaction",
            "rarity",
            "repeated",
            "semantic_temporal",
            "thread_temporal",
            "reply_semantic",
            "repeated_target",
        )
        normalized_ranges = {
            name: {
                "min": float(ordered[f"{name}_normalized"].min()),
                "max": float(ordered[f"{name}_normalized"].max()),
            }
            for name in component_names
        }
        LOGGER.info("Feature ranges after normalization: %s", normalized_ranges)

        component_maxima = {name: float(ordered[f"{name}_normalized"].max()) for name in component_names}
        ceiling = float(
            sum(component_maxima[name] * weight_map[name] for name in component_names) / total_weight
        )
        LOGGER.info(
            "Maximum possible weighted score under observed normalized maxima: %.6f",
            ceiling,
        )
        LOGGER.info(
            "Per-component normalized maxima used for the ceiling calculation: %s",
            component_maxima,
        )

        group_contribution_columns = {
            "temporal": "temporal_contribution",
            "interaction": "interaction_contribution",
            "semantic": "semantic_contribution",
            "rarity": "rarity_contribution",
            "repeated_behaviour": "repeated_behavior_contribution",
            "higher_order": "higher_order_contribution",
        }
        group_totals = {
            name: float(ordered[column].sum()) for name, column in group_contribution_columns.items()
        }
        total_score_mass = float(ordered[score_column].sum())
        group_shares = {
            name: (value / total_score_mass if total_score_mass > 0.0 else 0.0)
            for name, value in group_totals.items()
        }
        LOGGER.info("Contribution of each feature group to the final score: %s", group_shares)

        dominance_threshold = 0.50
        negligible_threshold = 0.05
        dominant_groups = [name for name, share in group_shares.items() if share >= dominance_threshold]
        negligible_groups = [name for name, share in group_shares.items() if share <= negligible_threshold]
        LOGGER.info(
            "Dominance heuristic (>%.0f%% share) dominant groups: %s",
            dominance_threshold * 100.0,
            dominant_groups if dominant_groups else "none",
        )
        LOGGER.info(
            "Negligible heuristic (<%.0f%% share) negligible groups: %s",
            negligible_threshold * 100.0,
            negligible_groups if negligible_groups else "none",
        )

        LOGGER.info("Runtime: %.2fs", perf_counter() - start)

    def _empty_output(self) -> pd.DataFrame:
        """Return an empty output frame with the expected schema."""

        return pd.DataFrame(
            columns=[
                self.config.candidate_post_id_1_column,
                self.config.candidate_post_id_2_column,
                "coordination_score",
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
    rarity_features = pd.DataFrame(
        [
            {
                "post_id_1": "p1",
                "post_id_2": "p2",
                "mention_rarity": 0.5,
                "url_rarity": 0.2,
                "hashtag_rarity": 0.3,
                "thread_rarity": 0.7,
                "reply_target_rarity": 0.4,
            },
            {
                "post_id_1": "p3",
                "post_id_2": "p4",
                "mention_rarity": 0.1,
                "url_rarity": 0.3,
                "hashtag_rarity": 0.4,
                "thread_rarity": 0.2,
                "reply_target_rarity": 0.6,
            },
        ]
    )
    repeated_behavior_features = pd.DataFrame(
        [
            {"post_id_1": "p1", "post_id_2": "p2", "repeated_coactivity_count": 2, "shared_target_count": 3},
            {"post_id_1": "p3", "post_id_2": "p4", "repeated_coactivity_count": 1, "shared_target_count": 1},
        ]
    )
    higher_order_features = pd.DataFrame(
        [
            {
                "post_id_1": "p1",
                "post_id_2": "p2",
                "semantic_temporal_agreement": 0.4,
                "thread_temporal_agreement": 0.5,
                "reply_semantic_agreement": 0.0,
                "interaction_strength": 0.25,
                "repeated_target_agreement": 1.0,
            },
            {
                "post_id_1": "p3",
                "post_id_2": "p4",
                "semantic_temporal_agreement": 0.08,
                "thread_temporal_agreement": 0.0,
                "reply_semantic_agreement": 0.4,
                "interaction_strength": 0.3333333333,
                "repeated_target_agreement": 0.3,
            },
        ]
    )

    result = PairCoordinationScorer().transform(
        temporal_features,
        interaction_features,
        semantic_features,
        rarity_features,
        repeated_behavior_features,
        higher_order_features,
    )
    assert list(result.columns) == [
        "post_id_1",
        "post_id_2",
        "coordination_score",
    ]
    assert result["coordination_score"].between(0.0, 1.0).all()
