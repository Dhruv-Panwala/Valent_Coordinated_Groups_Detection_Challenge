"""Data loading for the coordination detection pipeline."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

LOGGER = logging.getLogger(__name__)


def load_accounts(file_path: str) -> pd.DataFrame:
    """Load and flatten account records from a JSONL file."""

    raw = _read_jsonl(file_path)
    flattened = _flatten_nested_records(raw)
    return _clean_frame(flattened, ("account_id",), timestamp_columns=())


def load_posts(file_path: str) -> pd.DataFrame:
    """Load post records from a JSONL file."""

    raw = _read_jsonl(file_path)
    return _clean_frame(raw, ("post_id", "account_id", "created_at", "text"), timestamp_columns=("created_at",))


def load_embeddings(file_path: str) -> pd.DataFrame:
    """Load embedding records from a Parquet file."""

    path = Path(file_path)
    LOGGER.info("Loading Parquet file: %s", path)
    try:
        frame = pd.read_parquet(path)
    except ImportError as exc:  # pragma: no cover - depends on local environment
        raise RuntimeError(
            "Parquet support requires `pyarrow` or `fastparquet` to be installed."
        ) from exc

    _validate_embeddings_frame(frame)
    return frame.copy()


def _read_jsonl(file_path: str) -> pd.DataFrame:
    """Read a JSONL file into a DataFrame."""

    path = Path(file_path)
    LOGGER.info("Loading JSONL file: %s", path)
    try:
        return pd.read_json(path, lines=True)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def _flatten_nested_records(frame: pd.DataFrame) -> pd.DataFrame:
    """Flatten nested dict fields without touching list-valued columns."""

    if frame.empty:
        return frame.copy()
    return pd.json_normalize(frame.to_dict(orient="records"), sep="_")


def _clean_frame(
    frame: pd.DataFrame,
    required_columns: tuple[str, ...],
    *,
    timestamp_columns: tuple[str, ...],
) -> pd.DataFrame:
    """Validate, normalize missing values, and coerce timestamps."""

    cleaned = frame.copy()
    if cleaned.empty and not len(cleaned.columns):
        cleaned = _empty_frame(required_columns)
    _validate_required_columns(cleaned, required_columns)
    cleaned = cleaned.convert_dtypes()
    cleaned = _convert_timestamps(cleaned, timestamp_columns)
    return cleaned


def _empty_frame(columns: tuple[str, ...]) -> pd.DataFrame:
    """Create an empty frame with the expected columns."""

    return pd.DataFrame({column: pd.Series(dtype="object") for column in columns})


def _validate_required_columns(frame: pd.DataFrame, required_columns: tuple[str, ...]) -> None:
    """Raise a clear error when required columns are missing."""

    missing = [column for column in required_columns if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")


def _validate_embeddings_frame(frame: pd.DataFrame) -> None:
    """Validate the expected embeddings schema and vector shape."""

    required_columns = ("post_id", "vector")
    _validate_required_columns(frame, required_columns)

    if frame["post_id"].isna().any():
        raise ValueError("Embeddings file contains null values in post_id.")

    duplicated_post_ids = frame.loc[frame["post_id"].duplicated(keep=False), "post_id"]
    if not duplicated_post_ids.empty:
        duplicate_values = duplicated_post_ids.astype(str).drop_duplicates().tolist()
        raise ValueError(
            "Embeddings file must contain unique post_id values. "
            f"Duplicate post_id values found: {', '.join(duplicate_values)}"
        )

    vector_dimensions: list[int] = []
    for row_index, value in enumerate(frame["vector"]):
        if _is_missing_embedding_vector(value):
            raise ValueError(f"Embeddings file contains null vector values at row {row_index}.")

        dimension = _embedding_vector_dimension(value, row_index=row_index)
        vector_dimensions.append(dimension)

    if vector_dimensions and len(set(vector_dimensions)) != 1:
        raise ValueError(
            "Embedding vectors must all have the same dimensionality. "
            f"Found dimensions: {sorted(set(vector_dimensions))}"
        )


def _is_missing_embedding_vector(value: object) -> bool:
    """Return True when an embedding vector cell is absent."""

    if value is None:
        return True
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def _embedding_vector_dimension(value: object, *, row_index: int) -> int:
    """Return the dimensionality of one embedding vector."""

    if isinstance(value, pd.Series):
        vector = value.to_numpy()
    elif isinstance(value, list):
        vector = value
    elif isinstance(value, tuple):
        vector = list(value)
    elif hasattr(value, "shape"):
        vector = value
    else:
        raise ValueError(f"Embedding vector at row {row_index} must be array-like.")

    try:
        dimension = len(vector)
    except TypeError as exc:
        raise ValueError(f"Embedding vector at row {row_index} must be one-dimensional.") from exc

    if dimension <= 0:
        raise ValueError(f"Embedding vector at row {row_index} must not be empty.")

    return int(dimension)


def _convert_timestamps(frame: pd.DataFrame, timestamp_columns: tuple[str, ...]) -> pd.DataFrame:
    """Convert known timestamp columns to datetime values."""

    cleaned = frame.copy()
    for column in timestamp_columns:
        if column in cleaned.columns:
            cleaned[column] = pd.to_datetime(cleaned[column], errors="coerce")
    return cleaned
