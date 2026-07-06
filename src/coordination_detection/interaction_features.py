"""Raw interaction feature extraction for candidate pairs."""

from __future__ import annotations

import logging
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd

LOGGER = logging.getLogger(__name__)


class InteractionFeatureExtractor:
    """Compute raw interaction evidence for candidate post pairs."""

    def __init__(
        self,
        *,
        post_id_column: str = "post_id",
        candidate_post_id_1_column: str = "post_id_1",
        candidate_post_id_2_column: str = "post_id_2",
        thread_id_column: str = "thread_id",
        reply_to_post_id_column: str = "reply_to_post_id",
        quoted_post_id_column: str = "quoted_post_id",
        mentions_column: str = "mentions",
        urls_column: str = "urls",
        hashtags_column: str = "hashtags",
    ) -> None:
        self.post_id_column = post_id_column
        self.candidate_post_id_1_column = candidate_post_id_1_column
        self.candidate_post_id_2_column = candidate_post_id_2_column
        self.thread_id_column = thread_id_column
        self.reply_to_post_id_column = reply_to_post_id_column
        self.quoted_post_id_column = quoted_post_id_column
        self.mentions_column = mentions_column
        self.urls_column = urls_column
        self.hashtags_column = hashtags_column

    def transform(self, candidate_pairs: pd.DataFrame, posts: pd.DataFrame) -> pd.DataFrame:
        """Return raw interaction features for each candidate pair."""

        start = perf_counter()
        pairs = self._prepare_candidate_pairs(candidate_pairs)
        if pairs.empty:
            LOGGER.info("Interaction features computed for 0 candidate pairs in %.2fs", perf_counter() - start)
            return self._empty_output()

        posts_frame = self._prepare_posts(posts)
        features = pairs[["_row_order", self.candidate_post_id_1_column, self.candidate_post_id_2_column]].copy()

        features = features.merge(
            self._same_value_feature(posts_frame, pairs, self.thread_id_column, "same_thread"),
            on="_row_order",
            how="left",
            sort=False,
        )
        features = features.merge(
            self._same_value_feature(posts_frame, pairs, self.reply_to_post_id_column, "same_reply_target"),
            on="_row_order",
            how="left",
            sort=False,
        )
        features = features.merge(
            self._same_value_feature(posts_frame, pairs, self.quoted_post_id_column, "same_quoted_post"),
            on="_row_order",
            how="left",
            sort=False,
        )
        features = features.merge(
            self._list_feature(posts_frame, pairs, self.mentions_column, "shared_mention_count", "mention_jaccard_similarity"),
            on="_row_order",
            how="left",
            sort=False,
        )
        features = features.merge(
            self._list_feature(posts_frame, pairs, self.urls_column, "shared_url_count", "url_jaccard_similarity"),
            on="_row_order",
            how="left",
            sort=False,
        )
        features = features.merge(
            self._list_feature(posts_frame, pairs, self.hashtags_column, "shared_hashtag_count", "hashtag_jaccard_similarity"),
            on="_row_order",
            how="left",
            sort=False,
        )

        output = features.sort_values("_row_order", kind="stable").reset_index(drop=True)
        output = output[
            [
                self.candidate_post_id_1_column,
                self.candidate_post_id_2_column,
                "same_thread",
                "same_reply_target",
                "same_quoted_post",
                "mention_jaccard_similarity",
                "url_jaccard_similarity",
                "hashtag_jaccard_similarity",
            ]
        ].copy()

        output["same_thread"] = output["same_thread"].fillna(False).astype(bool)
        output["same_reply_target"] = output["same_reply_target"].fillna(False).astype(bool)
        output["same_quoted_post"] = output["same_quoted_post"].fillna(False).astype(bool)
        for column in (
            "mention_jaccard_similarity",
            "url_jaccard_similarity",
            "hashtag_jaccard_similarity",
        ):
            output[column] = output[column].fillna(0.0).astype(np.float64)

        LOGGER.info("Interaction features computed for %d candidate pairs in %.2fs", len(output), perf_counter() - start)
        return output

    def _prepare_candidate_pairs(self, candidate_pairs: pd.DataFrame) -> pd.DataFrame:
        """Validate and normalize the candidate-pair frame."""

        required = {self.candidate_post_id_1_column, self.candidate_post_id_2_column}
        missing = sorted(required.difference(candidate_pairs.columns))
        if missing:
            raise ValueError(f"Missing required columns: {', '.join(missing)}")

        frame = candidate_pairs[[self.candidate_post_id_1_column, self.candidate_post_id_2_column]].copy()
        frame = frame.dropna(subset=[self.candidate_post_id_1_column, self.candidate_post_id_2_column])
        frame = frame.reset_index(drop=True)
        frame["_row_order"] = np.arange(len(frame), dtype=np.int64)
        return frame

    def _prepare_posts(self, posts: pd.DataFrame) -> pd.DataFrame:
        """Validate and normalize the posts frame."""

        if self.post_id_column not in posts.columns:
            raise ValueError(f"Missing required column: {self.post_id_column}")

        frame = posts.copy()
        frame = frame.dropna(subset=[self.post_id_column])
        frame = frame.drop_duplicates(subset=[self.post_id_column], keep="first")
        return frame

    def _same_value_feature(
        self,
        posts: pd.DataFrame,
        pairs: pd.DataFrame,
        source_column: str,
        output_column: str,
    ) -> pd.DataFrame:
        """Compare scalar interaction columns for exact equality."""

        if source_column not in posts.columns:
            return pd.DataFrame({"_row_order": pairs["_row_order"], output_column: False})

        lookup = posts[[self.post_id_column, source_column]].copy()
        lookup_left = lookup.rename(columns={self.post_id_column: self.candidate_post_id_1_column, source_column: "_left"})
        lookup_right = lookup.rename(columns={self.post_id_column: self.candidate_post_id_2_column, source_column: "_right"})

        merged = pairs[["_row_order", self.candidate_post_id_1_column, self.candidate_post_id_2_column]].merge(
            lookup_left,
            on=self.candidate_post_id_1_column,
            how="left",
            sort=False,
        )
        merged = merged.merge(
            lookup_right,
            on=self.candidate_post_id_2_column,
            how="left",
            sort=False,
        )
        same = merged["_left"].notna() & merged["_right"].notna() & merged["_left"].eq(merged["_right"])
        return pd.DataFrame({"_row_order": merged["_row_order"], output_column: same.to_numpy(dtype=bool)})

    def _list_feature(
        self,
        posts: pd.DataFrame,
        pairs: pd.DataFrame,
        source_column: str,
        shared_output_column: str,
        jaccard_output_column: str,
    ) -> pd.DataFrame:
        """Compute shared-token counts and Jaccard similarity for list columns."""

        empty = pd.DataFrame(
            {
                "_row_order": pairs["_row_order"],
                shared_output_column: 0,
                jaccard_output_column: 0.0,
            }
        )
        if source_column not in posts.columns:
            return empty

        exploded = posts[[self.post_id_column, source_column]].copy()
        exploded[source_column] = exploded[source_column].map(self._to_value_list)
        exploded = exploded.explode(source_column, ignore_index=True)
        exploded = exploded.dropna(subset=[source_column])
        if exploded.empty:
            return empty

        exploded = exploded.drop_duplicates(subset=[self.post_id_column, source_column], keep="first")
        token_counts = (
            exploded.groupby(self.post_id_column, sort=True)[source_column]
            .nunique()
            .rename("token_count")
            .reset_index()
        )

        pair_base = pairs[["_row_order", self.candidate_post_id_1_column, self.candidate_post_id_2_column]].copy()
        left = pair_base.merge(
            exploded.rename(columns={self.post_id_column: self.candidate_post_id_1_column, source_column: "_token"}),
            on=self.candidate_post_id_1_column,
            how="left",
            sort=False,
        )[["_row_order", "_token"]]
        right = pair_base.merge(
            exploded.rename(columns={self.post_id_column: self.candidate_post_id_2_column, source_column: "_token"}),
            on=self.candidate_post_id_2_column,
            how="left",
            sort=False,
        )[["_row_order", "_token"]]

        shared_tokens = left.merge(right, on=["_row_order", "_token"], how="inner")
        shared_counts = shared_tokens.groupby("_row_order", sort=False).size().rename(shared_output_column).reset_index()

        counts_left = token_counts.rename(
            columns={self.post_id_column: self.candidate_post_id_1_column, "token_count": "_count_1"}
        )
        counts_right = token_counts.rename(
            columns={self.post_id_column: self.candidate_post_id_2_column, "token_count": "_count_2"}
        )
        pair_counts = pair_base.merge(counts_left, on=self.candidate_post_id_1_column, how="left", sort=False)
        pair_counts = pair_counts.merge(counts_right, on=self.candidate_post_id_2_column, how="left", sort=False)
        pair_counts = pair_counts.merge(shared_counts, on="_row_order", how="left", sort=False)

        pair_counts["_count_1"] = pair_counts["_count_1"].fillna(0).astype(np.int64)
        pair_counts["_count_2"] = pair_counts["_count_2"].fillna(0).astype(np.int64)
        pair_counts[shared_output_column] = pair_counts[shared_output_column].fillna(0).astype(np.int64)

        union = pair_counts["_count_1"] + pair_counts["_count_2"] - pair_counts[shared_output_column]
        jaccard = np.divide(
            pair_counts[shared_output_column].to_numpy(dtype=np.float64),
            union.to_numpy(dtype=np.float64),
            out=np.zeros(len(pair_counts), dtype=np.float64),
            where=union.to_numpy(dtype=np.float64) > 0,
        )

        return pd.DataFrame(
            {
                "_row_order": pair_counts["_row_order"],
                shared_output_column: pair_counts[shared_output_column].to_numpy(dtype=np.int64),
                jaccard_output_column: jaccard,
            }
        )

    def _to_value_list(self, value: Any) -> list[Any]:
        """Normalize scalars and sequences into a clean list of values."""

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
        """Return True for values that should be treated as absent."""

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
                "same_thread",
                "same_reply_target",
                "same_quoted_post",
                "mention_jaccard_similarity",
                "url_jaccard_similarity",
                "hashtag_jaccard_similarity",
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
                "quoted_post_id": "q1",
                "mentions": ["@a", "@b"],
                "urls": ["u1"],
                "hashtags": ["#x", "#y"],
            },
            {
                "post_id": "p2",
                "thread_id": "t1",
                "reply_to_post_id": "r2",
                "quoted_post_id": "q1",
                "mentions": ["@b"],
                "urls": ["u1", "u2"],
                "hashtags": [],
            },
            {
                "post_id": "p3",
                "thread_id": "t2",
                "reply_to_post_id": "r1",
                "quoted_post_id": None,
                "mentions": None,
                "urls": None,
                "hashtags": ["#y"],
            },
        ]
    )

    result = InteractionFeatureExtractor().transform(sample_candidates, sample_posts)
    assert list(result.columns) == [
        "post_id_1",
        "post_id_2",
        "same_thread",
        "same_reply_target",
        "same_quoted_post",
        "mention_jaccard_similarity",
        "url_jaccard_similarity",
        "hashtag_jaccard_similarity",
    ]
    assert bool(result.loc[0, "same_thread"]) is True
    assert result.loc[0, "mention_jaccard_similarity"] == 0.5
    assert result.loc[1, "hashtag_jaccard_similarity"] == 0.5
