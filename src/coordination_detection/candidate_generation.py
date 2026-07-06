"""Candidate pair generation for coordination analysis."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from itertools import combinations
from pathlib import Path
from time import perf_counter
from typing import Any

import faiss
import numpy as np
import pandas as pd

from . import config as pipeline_config
from .config import CandidateGenerationConfig

LOGGER = logging.getLogger(__name__)
LIST_INTERACTION_SOURCES = {"mentions", "rare_urls"}
SEMANTIC_SOURCE = "semantic"


@dataclass(slots=True)
class _PairRecord:
    """In-memory accumulator for one unique candidate pair."""

    post_id_1: Any
    account_id_1: Any
    post_id_2: Any
    account_id_2: Any
    shared_features: dict[str, set[Any]] = field(default_factory=dict)
    candidate_sources: set[str] = field(default_factory=set)
    semantic_cosine_similarity: float | None = None

    def add_interaction_value(self, source_name: str, value: Any) -> None:
        """Store one exact shared interaction value for the pair."""

        self.candidate_sources.add(source_name)
        self.shared_features.setdefault(source_name, set()).add(value)

    def add_semantic_similarity(self, similarity: float) -> None:
        """Store cosine similarity for a semantic neighbour pair."""

        self.candidate_sources.add(SEMANTIC_SOURCE)
        if self.semantic_cosine_similarity is None or similarity > self.semantic_cosine_similarity:
            self.semantic_cosine_similarity = similarity

    @property
    def candidate_source_count(self) -> int:
        """Return how many distinct sources supported the pair."""

        return len(self.candidate_sources)


@dataclass(slots=True)
class _KeyDiagnostic:
    """Candidate counts for one interaction key."""

    source: str
    interaction_key: Any
    generated_pairs: int


class CandidateGenerator:
    """Generate unique post pairs from shared interaction keys and semantic neighbours."""

    def __init__(self, config: CandidateGenerationConfig | None = None) -> None:
        self.config = config or CandidateGenerationConfig()
        self._last_statistics = self._empty_statistics()
        self._last_diagnostics = self._empty_diagnostics()

    def generate_candidates(
        self,
        posts: pd.DataFrame,
        *,
        embeddings: pd.DataFrame | None = None,
        embeddings_path: str | Path | None = None,
    ) -> pd.DataFrame:
        """Return a deduplicated candidate-pair table with provenance metadata."""

        start = perf_counter()
        frame = self._prepare_posts(posts)
        if frame.empty:
            self._last_statistics = self._empty_statistics()
            self._last_diagnostics = self._empty_diagnostics()
            LOGGER.info("Total unique candidate pairs: 0")
            LOGGER.info("Duplicate pairs removed: 0")
            LOGGER.info("Candidate generation finished in %.2fs", perf_counter() - start)
            return self._empty_output()

        pairs: dict[tuple[Any, Any], _PairRecord] = {}
        candidates_per_source: dict[str, int] = {}
        raw_candidate_pairs = 0
        key_diagnostics: list[_KeyDiagnostic] = []

        for source_name, column_name, is_list_field in self._interaction_sources():
            generated = self._generate_from_source(
                frame,
                column_name=column_name,
                source_name=source_name,
                is_list_field=is_list_field,
                pairs=pairs,
                key_diagnostics=key_diagnostics,
            )
            candidates_per_source[source_name] = generated
            raw_candidate_pairs += generated
            LOGGER.info("Generated %d candidate pairs from %s", generated, source_name)

        semantic_start = perf_counter()
        semantic_pair_similarities, semantic_raw_hits, embedding_account_map = self._generate_semantic_pairs(
            frame,
            embeddings=embeddings,
            embeddings_path=embeddings_path,
        )
        semantic_runtime_seconds = perf_counter() - semantic_start

        semantic_candidate_pairs_generated = len(semantic_pair_similarities)
        semantic_pairs_merged = 0
        semantic_only_pairs = 0
        cosine_similarities: list[float] = []

        for pair_key, cosine_similarity in semantic_pair_similarities.items():
            cosine_similarities.append(cosine_similarity)
            record = pairs.get(pair_key)
            if record is None:
                post_id_1, post_id_2 = pair_key
                record = _PairRecord(
                    post_id_1=post_id_1,
                    account_id_1=embedding_account_map[post_id_1],
                    post_id_2=post_id_2,
                    account_id_2=embedding_account_map[post_id_2],
                )
                pairs[pair_key] = record
                semantic_only_pairs += 1
            else:
                semantic_pairs_merged += 1
            record.add_semantic_similarity(cosine_similarity)

        output = self._pairs_to_frame(pairs)
        duplicate_pairs_removed = raw_candidate_pairs + semantic_raw_hits - len(output)
        average_sources = (
            float(output["candidate_source_count"].mean()) if not output.empty else 0.0
        )
        average_cosine_similarity = (
            float(np.mean(cosine_similarities)) if cosine_similarities else 0.0
        )

        self._last_statistics = {
            "total_candidate_pairs": len(output),
            "candidates_per_interaction_source": candidates_per_source,
            "duplicate_pairs_removed": duplicate_pairs_removed,
            "average_candidate_sources_per_pair": average_sources,
        }
        self._last_diagnostics = {
            "candidate_pairs_by_source": {
                **candidates_per_source,
                SEMANTIC_SOURCE: semantic_candidate_pairs_generated,
            },
            "semantic_candidate_pairs_generated": semantic_candidate_pairs_generated,
            "semantic_only_candidate_pairs": semantic_only_pairs,
            "semantic_pairs_merged_with_existing_candidates": semantic_pairs_merged,
            "total_candidate_pairs_after_merging": len(output),
            "average_cosine_similarity": average_cosine_similarity,
            "semantic_runtime_seconds": semantic_runtime_seconds,
            "runtime_seconds": perf_counter() - start,
            "top_interaction_keys": self._top_interaction_keys(key_diagnostics),
            "maximum_pairs_generated_by_single_thread": self._max_key(
                [item for item in key_diagnostics if item.source == "thread_id"]
            ),
            "maximum_pairs_generated_by_single_mention": self._max_key(
                [item for item in key_diagnostics if item.source == "mentions"]
            ),
            "maximum_pairs_generated_by_single_reply_target": self._max_key(
                [item for item in key_diagnostics if item.source == "reply_to_post_id"]
            ),
            "maximum_pairs_generated_by_single_url": self._max_key(
                [item for item in key_diagnostics if item.source == "rare_urls"]
            ),
        }

        LOGGER.info("Generated %d unique semantic candidate pairs", semantic_candidate_pairs_generated)
        LOGGER.info("Semantic-only candidate pairs: %d", semantic_only_pairs)
        LOGGER.info("Semantic pairs merged with existing candidates: %d", semantic_pairs_merged)
        LOGGER.info("Total unique candidate pairs: %d", len(output))
        LOGGER.info("Duplicate pairs removed: %d", duplicate_pairs_removed)
        LOGGER.info("Candidate generation finished in %.2fs", perf_counter() - start)
        return output

    def get_statistics(self) -> dict[str, Any]:
        """Return statistics from the most recent candidate-generation run."""

        return self._last_statistics.copy()

    def get_diagnostics(self) -> dict[str, Any]:
        """Return diagnostics from the most recent candidate-generation run."""

        return self._last_diagnostics.copy()

    def _prepare_posts(self, posts: pd.DataFrame) -> pd.DataFrame:
        """Validate required columns and remove rows without pairable identifiers."""

        required = {self.config.post_id_column, self.config.account_id_column}
        missing = sorted(required.difference(posts.columns))
        if missing:
            raise ValueError(f"Missing required columns: {', '.join(missing)}")

        frame = posts.copy()
        frame = frame.dropna(subset=[self.config.post_id_column, self.config.account_id_column])
        return frame

    def _interaction_sources(self) -> tuple[tuple[str, str, bool], ...]:
        """Return the configured interaction sources in processing order."""

        return (
            ("thread_id", self.config.thread_id_column, False),
            ("reply_to_post_id", self.config.reply_to_post_id_column, False),
            ("quoted_post_id", self.config.quoted_post_id_column, False),
            ("mentions", self.config.mentions_column, True),
            ("rare_urls", self.config.urls_column, True),
        )

    def _generate_from_source(
        self,
        frame: pd.DataFrame,
        *,
        column_name: str,
        source_name: str,
        is_list_field: bool,
        pairs: dict[tuple[Any, Any], _PairRecord],
        key_diagnostics: list[_KeyDiagnostic],
    ) -> int:
        """Generate candidate pairs for one interaction source."""

        if column_name not in frame.columns:
            LOGGER.warning("Skipping %s because column %s is missing", source_name, column_name)
            return 0

        working = frame[[self.config.post_id_column, self.config.account_id_column, column_name]].copy()
        if is_list_field and source_name == "rare_urls":
            working = self._explode_list_field(working, column_name)
            if working.empty:
                return 0
            counts = working.groupby(column_name)[self.config.post_id_column].nunique()
            rare_values = counts[counts <= self.config.rare_url_max_post_count].index
            working = working[working[column_name].isin(rare_values)]
        elif is_list_field and source_name == "mentions":
            working = self._explode_list_field(working, column_name)
            if working.empty:
                return 0
            counts = working.groupby(column_name)[self.config.post_id_column].nunique()
            allowed_values = counts[counts <= self.config.mention_max_post_count].index
            working = working[working[column_name].isin(allowed_values)]
        elif is_list_field:
            working = self._explode_list_field(working, column_name)

        working = working.dropna(subset=[column_name])
        if working.empty:
            return 0

        generated = 0
        working = working.sort_values(
            [column_name, self.config.post_id_column, self.config.account_id_column],
            kind="stable",
        )
        for value, group in working.groupby(column_name, sort=True, dropna=True):
            unique_posts = group.drop_duplicates(subset=[self.config.post_id_column]).sort_values(
                [self.config.post_id_column, self.config.account_id_column],
                kind="stable",
            )
            if len(unique_posts) < 2:
                continue
            pair_count = self._pair_count(len(unique_posts))
            generated += self._accumulate_group_pairs(
                unique_posts,
                source_name=source_name,
                source_value=value,
                pairs=pairs,
            )
            key_diagnostics.append(
                _KeyDiagnostic(
                    source=source_name,
                    interaction_key=value,
                    generated_pairs=pair_count,
                )
            )
        return generated

    def _generate_semantic_pairs(
        self,
        posts: pd.DataFrame,
        *,
        embeddings: pd.DataFrame | None,
        embeddings_path: str | Path | None,
    ) -> tuple[dict[tuple[Any, Any], float], int, dict[Any, Any]]:
        """Return unique semantic neighbour pairs and raw neighbour hits."""

        semantic_top_k = int(self.config.semantic_top_k)
        min_semantic_similarity = float(self.config.min_semantic_similarity)
        if semantic_top_k <= 0:
            return {}, 0, {}

        embedding_frame = self._prepare_embeddings(
            posts,
            embeddings=embeddings,
            embeddings_path=embeddings_path,
        )
        if embedding_frame.empty:
            return {}, 0, {}

        vectors = self._embedding_matrix(embedding_frame)
        if vectors.shape[0] < 2:
            return {}, 0, {}

        faiss.normalize_L2(vectors)
        index = faiss.IndexFlatIP(vectors.shape[1])
        index.add(vectors)

        k = min(semantic_top_k + 1, vectors.shape[0])
        scores, indices = index.search(vectors, k)

        semantic_pairs: dict[tuple[Any, Any], float] = {}
        raw_hits = 0
        embedding_account_map = dict(
            zip(embedding_frame[self.config.post_id_column], embedding_frame[self.config.account_id_column])
        )
        post_ids = embedding_frame[self.config.post_id_column].to_list()
        account_ids = embedding_frame[self.config.account_id_column].to_list()
        for row_index, source_row in enumerate(embedding_frame.itertuples(index=False)):
            for neighbor_rank in range(k):
                neighbor_index = int(indices[row_index, neighbor_rank])
                if neighbor_index < 0 or neighbor_index == row_index:
                    continue
                similarity = float(scores[row_index, neighbor_rank])
                if similarity < min_semantic_similarity:
                    continue
                raw_hits += 1
                pair_key = self._canonical_pair(
                    source_row.post_id,
                    source_row.account_id,
                    post_ids[neighbor_index],
                    account_ids[neighbor_index],
                )
                pair_key_by_post = (pair_key[0], pair_key[2])
                current = semantic_pairs.get(pair_key_by_post)
                if current is None or similarity > current:
                    semantic_pairs[pair_key_by_post] = similarity

        return semantic_pairs, raw_hits, embedding_account_map

    def _prepare_embeddings(
        self,
        posts: pd.DataFrame,
        *,
        embeddings: pd.DataFrame | None,
        embeddings_path: str | Path | None,
    ) -> pd.DataFrame:
        """Load, validate, and join embeddings to the current post set."""

        if embeddings is None:
            configured_path = embeddings_path if embeddings_path is not None else self.config.embeddings_path
            path = pipeline_config.resolve_project_path(configured_path)
            embeddings = pd.read_parquet(path)

        required = {self.config.post_id_column, "vector"}
        missing = sorted(required.difference(embeddings.columns))
        if missing:
            raise ValueError(f"Missing required embedding columns: {', '.join(missing)}")

        frame = embeddings[[self.config.post_id_column, "vector"]].copy()
        frame = frame.dropna(subset=[self.config.post_id_column, "vector"])
        frame = frame.drop_duplicates(subset=[self.config.post_id_column], keep="first")
        frame = frame.merge(
            posts[[self.config.post_id_column, self.config.account_id_column]],
            on=self.config.post_id_column,
            how="inner",
        )
        frame = frame.dropna(subset=[self.config.account_id_column])
        frame = frame.sort_values([self.config.post_id_column], kind="stable").reset_index(drop=True)
        return frame

    def _embedding_matrix(self, embedding_frame: pd.DataFrame) -> np.ndarray:
        """Convert embedding rows into a contiguous float32 matrix."""

        vectors = [np.asarray(vector, dtype=np.float32) for vector in embedding_frame["vector"].to_list()]
        if not vectors:
            return np.empty((0, 0), dtype=np.float32)
        matrix = np.vstack(vectors).astype(np.float32, copy=False)
        if matrix.ndim != 2:
            raise ValueError("Embedding vectors must be two-dimensional.")
        return np.ascontiguousarray(matrix)

    def _explode_list_field(self, frame: pd.DataFrame, column_name: str) -> pd.DataFrame:
        """Explode a list-valued column into one row per scalar value."""

        exploded = frame.copy()
        exploded[column_name] = exploded[column_name].map(self._to_value_list)
        exploded = exploded.explode(column_name, ignore_index=True)
        return exploded

    def _to_value_list(self, value: Any) -> list[Any]:
        """Normalize a scalar or sequence into a clean list of interaction values."""

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

    def _accumulate_group_pairs(
        self,
        group: pd.DataFrame,
        *,
        source_name: str,
        source_value: Any,
        pairs: dict[tuple[Any, Any], _PairRecord],
    ) -> int:
        """Add all pair combinations from one grouped interaction value."""

        rows = list(
            group[[self.config.post_id_column, self.config.account_id_column]].itertuples(
                index=False,
                name=None,
            )
        )
        generated = 0
        for left, right in combinations(rows, 2):
            post_id_1, account_id_1, post_id_2, account_id_2 = self._canonical_pair(*left, *right)
            pair_key = (post_id_1, post_id_2)
            record = pairs.get(pair_key)
            if record is None:
                record = _PairRecord(
                    post_id_1=post_id_1,
                    account_id_1=account_id_1,
                    post_id_2=post_id_2,
                    account_id_2=account_id_2,
                )
                pairs[pair_key] = record
            record.add_interaction_value(source_name, source_value)
            generated += 1
        return generated

    def _pair_count(self, unique_post_count: int) -> int:
        """Return the number of pairs produced by a key group."""

        return unique_post_count * (unique_post_count - 1) // 2

    def _canonical_pair(
        self,
        left_post: Any,
        left_account: Any,
        right_post: Any,
        right_account: Any,
    ) -> tuple[Any, Any, Any, Any]:
        """Return the pair in stable order so (A, B) and (B, A) match."""

        left_key = (str(left_post), str(left_account))
        right_key = (str(right_post), str(right_account))
        if left_key <= right_key:
            return left_post, left_account, right_post, right_account
        return right_post, right_account, left_post, left_account

    def _pairs_to_frame(self, pairs: dict[tuple[Any, Any], _PairRecord]) -> pd.DataFrame:
        """Convert the accumulator dictionary into the public output frame."""

        if not pairs:
            return self._empty_output()

        rows = []
        for record in sorted(
            pairs.values(),
            key=lambda item: (
                str(item.post_id_1),
                str(item.post_id_2),
                str(item.account_id_1),
                str(item.account_id_2),
            ),
        ):
            rows.append(
                {
                    "post_id_1": record.post_id_1,
                    "account_id_1": record.account_id_1,
                    "post_id_2": record.post_id_2,
                    "account_id_2": record.account_id_2,
                    "shared_features": self._format_shared_features(record.shared_features),
                    "candidate_sources": self._format_candidate_sources(record.candidate_sources),
                    "candidate_source_count": record.candidate_source_count,
                    "semantic_cosine_similarity": record.semantic_cosine_similarity,
                }
            )

        return pd.DataFrame(rows)

    def _format_shared_features(self, shared_features: dict[str, set[Any]]) -> dict[str, Any]:
        """Convert internal feature sets into deterministic output values."""

        formatted: dict[str, Any] = {}
        for source_name in self.config.interaction_order:
            if source_name not in shared_features:
                continue
            values = shared_features[source_name]
            if source_name in LIST_INTERACTION_SOURCES:
                formatted[source_name] = sorted(values, key=str)
                continue
            if len(values) == 1:
                formatted[source_name] = next(iter(values))
                continue
            formatted[source_name] = sorted(values, key=str)
        return formatted

    def _format_candidate_sources(self, candidate_sources: set[str]) -> list[str]:
        """Return candidate sources in deterministic order."""

        ordered = [source for source in self.config.interaction_order if source in candidate_sources]
        if SEMANTIC_SOURCE in candidate_sources:
            ordered.append(SEMANTIC_SOURCE)
        extras = sorted(candidate_sources.difference(ordered + [SEMANTIC_SOURCE]))
        ordered.extend(extras)
        return ordered

    def _empty_statistics(self) -> dict[str, Any]:
        """Return the zero-value statistics payload."""

        return {
            "total_candidate_pairs": 0,
            "candidates_per_interaction_source": {
                source_name: 0 for source_name, _, _ in self._interaction_sources()
            },
            "duplicate_pairs_removed": 0,
            "average_candidate_sources_per_pair": 0.0,
        }

    def _empty_diagnostics(self) -> dict[str, Any]:
        """Return the zero-value diagnostics payload."""

        return {
            "candidate_pairs_by_source": {
                **{source_name: 0 for source_name, _, _ in self._interaction_sources()},
                SEMANTIC_SOURCE: 0,
            },
            "semantic_candidate_pairs_generated": 0,
            "semantic_only_candidate_pairs": 0,
            "semantic_pairs_merged_with_existing_candidates": 0,
            "total_candidate_pairs_after_merging": 0,
            "average_cosine_similarity": 0.0,
            "semantic_runtime_seconds": 0.0,
            "runtime_seconds": 0.0,
            "top_interaction_keys": [],
            "maximum_pairs_generated_by_single_thread": None,
            "maximum_pairs_generated_by_single_mention": None,
            "maximum_pairs_generated_by_single_reply_target": None,
            "maximum_pairs_generated_by_single_url": None,
        }

    def _top_interaction_keys(self, key_diagnostics: list[_KeyDiagnostic]) -> list[dict[str, Any]]:
        """Return the highest-volume interaction keys in deterministic order."""

        top_n = int(getattr(self.config, "diagnostic_top_n_keys", 20))
        top_keys = sorted(
            key_diagnostics,
            key=lambda item: (-item.generated_pairs, item.source, str(item.interaction_key)),
        )[:top_n]
        return [
            {
                "source": item.source,
                "interaction_key": item.interaction_key,
                "generated_pairs": item.generated_pairs,
            }
            for item in top_keys
        ]

    def _max_key(self, items: list[_KeyDiagnostic]) -> dict[str, Any] | None:
        """Return the highest-volume key summary, if any."""

        if not items:
            return None
        best = max(items, key=lambda item: (item.generated_pairs, str(item.interaction_key)))
        return {
            "interaction_key": best.interaction_key,
            "generated_pairs": best.generated_pairs,
        }

    def _empty_output(self) -> pd.DataFrame:
        """Return an empty candidate-pair frame with the expected schema."""

        return pd.DataFrame(
            columns=[
                "post_id_1",
                "account_id_1",
                "post_id_2",
                "account_id_2",
                "shared_features",
                "candidate_sources",
                "candidate_source_count",
                "semantic_cosine_similarity",
            ]
        )


if __name__ == "__main__":
    sample_posts = pd.DataFrame(
        [
            {
                "post_id": "p1",
                "account_id": "a1",
                "thread_id": "t1",
                "reply_to_post_id": None,
                "quoted_post_id": None,
                "mentions": ["@x"],
                "urls": ["https://example.com"],
            },
            {
                "post_id": "p2",
                "account_id": "a2",
                "thread_id": "t1",
                "reply_to_post_id": None,
                "quoted_post_id": None,
                "mentions": ["@x"],
                "urls": [],
            },
        ]
    )
    sample_embeddings = pd.DataFrame(
        [
            {"post_id": "p1", "vector": [1.0, 0.0]},
            {"post_id": "p2", "vector": [0.99, 0.01]},
        ]
    )
    result = CandidateGenerator().generate_candidates(sample_posts, embeddings=sample_embeddings)
    assert not result.empty
    assert "semantic" in result.loc[0, "candidate_sources"]
