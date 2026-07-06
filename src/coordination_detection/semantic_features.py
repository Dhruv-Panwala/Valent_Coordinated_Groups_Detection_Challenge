"""Pairwise semantic feature extraction for candidate pairs."""

from __future__ import annotations

import logging
from time import perf_counter

import numpy as np
import pandas as pd

LOGGER = logging.getLogger(__name__)


class SemanticFeatureExtractor:
    """Compute pairwise semantic evidence for candidate post pairs."""

    def __init__(
        self,
        *,
        candidate_post_id_1_column: str = "post_id_1",
        candidate_post_id_2_column: str = "post_id_2",
        embedding_post_id_column: str = "post_id",
        embedding_vector_column: str = "vector",
    ) -> None:
        self.candidate_post_id_1_column = candidate_post_id_1_column
        self.candidate_post_id_2_column = candidate_post_id_2_column
        self.embedding_post_id_column = embedding_post_id_column
        self.embedding_vector_column = embedding_vector_column

    def transform(self, candidate_pairs: pd.DataFrame, embeddings: pd.DataFrame) -> pd.DataFrame:
        """Return cosine similarity for candidate pairs."""

        start = perf_counter()
        pairs = self._prepare_candidate_pairs(candidate_pairs)
        embedding_frame = self._prepare_embeddings(embeddings)

        if pairs.empty or embedding_frame.empty:
            LOGGER.info("Semantic features processed: 0")
            LOGGER.info("Runtime: %.2fs", perf_counter() - start)
            LOGGER.info("Cosine similarity summary: {}")
            LOGGER.info("Mean cosine similarity: 0.0")
            return self._empty_output()

        merged = self._attach_embeddings(pairs, embedding_frame)
        processed = len(merged)
        if processed == 0:
            LOGGER.info("Semantic features processed: 0")
            LOGGER.info("Runtime: %.2fs", perf_counter() - start)
            LOGGER.info("Cosine similarity summary: {}")
            LOGGER.info("Mean cosine similarity: 0.0")
            return self._empty_output()

        left_vectors = self._vector_matrix(merged, "vector_1")
        right_vectors = self._vector_matrix(merged, "vector_2")
        cosine_similarity = self._cosine_similarity(left_vectors, right_vectors)

        output = pd.DataFrame(
            {
                self.candidate_post_id_1_column: merged[self.candidate_post_id_1_column],
                self.candidate_post_id_2_column: merged[self.candidate_post_id_2_column],
                "cosine_similarity": cosine_similarity,
            }
        ).reset_index(drop=True)

        self._log_summary(output, processed, start)
        return output

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

    def _prepare_embeddings(self, embeddings: pd.DataFrame) -> pd.DataFrame:
        """Validate and normalize embedding input."""

        required = {self.embedding_post_id_column, self.embedding_vector_column}
        missing = sorted(required.difference(embeddings.columns))
        if missing:
            raise ValueError(f"Missing required embedding columns: {', '.join(missing)}")

        frame = embeddings[[self.embedding_post_id_column, self.embedding_vector_column]].copy()
        frame = frame.dropna(subset=[self.embedding_post_id_column, self.embedding_vector_column])
        frame = frame.drop_duplicates(subset=[self.embedding_post_id_column], keep="first")
        return frame

    def _attach_embeddings(self, pairs: pd.DataFrame, embeddings: pd.DataFrame) -> pd.DataFrame:
        """Join embeddings onto both sides of each candidate pair."""

        left = embeddings.rename(
            columns={
                self.embedding_post_id_column: self.candidate_post_id_1_column,
                self.embedding_vector_column: "vector_1",
            }
        )
        right = embeddings.rename(
            columns={
                self.embedding_post_id_column: self.candidate_post_id_2_column,
                self.embedding_vector_column: "vector_2",
            }
        )

        merged = pairs.merge(left, on=self.candidate_post_id_1_column, how="inner", sort=False)
        merged = merged.merge(right, on=self.candidate_post_id_2_column, how="inner", sort=False)
        return merged.sort_values("_row_order", kind="stable").reset_index(drop=True)

    def _vector_matrix(self, frame: pd.DataFrame, column: str) -> np.ndarray:
        """Convert a vector column into a contiguous float32 matrix."""

        vectors = [np.asarray(vector, dtype=np.float32) for vector in frame[column].to_list()]
        if not vectors:
            return np.empty((0, 0), dtype=np.float32)

        matrix = np.vstack(vectors).astype(np.float32, copy=False)
        if matrix.ndim != 2:
            raise ValueError("Embedding vectors must be two-dimensional.")
        return np.ascontiguousarray(matrix)

    def _cosine_similarity(self, left: np.ndarray, right: np.ndarray) -> np.ndarray:
        """Compute row-wise cosine similarity."""

        left_norm = np.linalg.norm(left, axis=1)
        right_norm = np.linalg.norm(right, axis=1)
        denominator = left_norm * right_norm
        numerator = np.einsum("ij,ij->i", left, right)
        return np.divide(
            numerator,
            denominator,
            out=np.zeros_like(numerator, dtype=np.float64),
            where=denominator > 0,
        )

    def _log_summary(self, output: pd.DataFrame, processed: int, start: float) -> None:
        """Log summary statistics for semantic pairwise features."""

        cosine = output["cosine_similarity"]
        LOGGER.info("Semantic features processed: %d", processed)
        LOGGER.info("Runtime: %.2fs", perf_counter() - start)
        LOGGER.info(
            "Cosine similarity summary: %s",
            cosine.describe(percentiles=(0.25, 0.5, 0.75)).to_dict(),
        )
        LOGGER.info("Mean cosine similarity: %.6f", float(cosine.mean()))

    def _empty_output(self) -> pd.DataFrame:
        """Return an empty output frame with the expected schema."""

        return pd.DataFrame(
            columns=[
                self.candidate_post_id_1_column,
                self.candidate_post_id_2_column,
                "cosine_similarity",
            ]
        )


if __name__ == "__main__":
    sample_candidates = pd.DataFrame(
        [
            {"post_id_1": "p1", "post_id_2": "p2"},
            {"post_id_1": "p1", "post_id_2": "p3"},
        ]
    )
    sample_embeddings = pd.DataFrame(
        [
            {"post_id": "p1", "vector": [1.0, 0.0]},
            {"post_id": "p2", "vector": [0.8, 0.6]},
            {"post_id": "p3", "vector": [0.0, 1.0]},
        ]
    )
    result = SemanticFeatureExtractor().transform(sample_candidates, sample_embeddings)
    assert list(result.columns) == [
        "post_id_1",
        "post_id_2",
        "cosine_similarity",
    ]
    assert np.isclose(result.loc[0, "cosine_similarity"], 0.8)
