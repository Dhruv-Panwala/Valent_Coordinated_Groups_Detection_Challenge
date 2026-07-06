"""Mention-threshold analysis for candidate-pair explosion."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

if __package__ in (None, ""):
    import sys

    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from coordination_detection.config import MentionThresholdAnalysisConfig
    from coordination_detection.loader import load_accounts, load_posts
else:
    from .config import MentionThresholdAnalysisConfig
    from .loader import load_accounts, load_posts

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class MentionAnalysisResult:
    """Artifacts produced by the mention-threshold analysis workflow."""

    mention_frequency: pd.DataFrame
    pair_contribution: pd.DataFrame
    threshold_sweep: pd.DataFrame
    recommended_threshold: int | None
    report: str
    output_dir: Path


class MentionThresholdAnalyzer:
    """Analyze mention fan-out to choose a practical mention threshold."""

    def __init__(self, config: MentionThresholdAnalysisConfig | None = None) -> None:
        self.config = config or MentionThresholdAnalysisConfig()

    def run(
        self,
        posts: pd.DataFrame,
        *,
        output_dir: str | Path | None = None,
    ) -> MentionAnalysisResult:
        """Compute analysis tables, plots, and a concise report."""

        destination = Path(output_dir or self.config.output_dir)
        destination.mkdir(parents=True, exist_ok=True)

        mention_frequency = self._build_mention_frequency(posts)
        pair_contribution = self._build_pair_contribution(mention_frequency)
        threshold_sweep = self._build_threshold_sweep(mention_frequency)
        recommended_threshold = self._recommend_threshold(threshold_sweep)

        self._write_outputs(destination, mention_frequency, pair_contribution, threshold_sweep)
        self._write_plots(destination, mention_frequency, threshold_sweep, recommended_threshold)
        report = self._build_report(mention_frequency, pair_contribution, threshold_sweep, recommended_threshold)
        report_path = destination / "summary_report.txt"
        report_path.write_text(report, encoding="utf-8")

        return MentionAnalysisResult(
            mention_frequency=mention_frequency,
            pair_contribution=pair_contribution,
            threshold_sweep=threshold_sweep,
            recommended_threshold=recommended_threshold,
            report=report,
            output_dir=destination,
        )

    def _build_mention_frequency(self, posts: pd.DataFrame) -> pd.DataFrame:
        """Count how often each mentioned account appears across unique posts."""

        if self.config.post_id_column not in posts.columns:
            raise ValueError(f"Missing required column: {self.config.post_id_column}")
        if self.config.mention_column not in posts.columns:
            raise ValueError(f"Missing required column: {self.config.mention_column}")

        frame = posts[[self.config.post_id_column, self.config.mention_column]].copy()
        frame = frame.dropna(subset=[self.config.post_id_column])
        frame[self.config.mention_column] = frame[self.config.mention_column].map(self._to_value_list)
        exploded = frame.explode(self.config.mention_column, ignore_index=True)
        exploded = exploded.dropna(subset=[self.config.mention_column])
        exploded = exploded.drop_duplicates(subset=[self.config.post_id_column, self.config.mention_column])

        counts = (
            exploded.groupby(self.config.mention_column, sort=True)[self.config.post_id_column]
            .nunique()
            .rename("mention_post_count")
            .reset_index()
            .rename(columns={self.config.mention_column: "mentioned_account"})
        )
        counts["generated_pair_count"] = counts["mention_post_count"].map(self._pair_count)
        counts = counts.sort_values(
            ["mention_post_count", "generated_pair_count", "mentioned_account"],
            ascending=[False, False, True],
            kind="stable",
        ).reset_index(drop=True)
        return counts

    def _build_pair_contribution(self, mention_frequency: pd.DataFrame) -> pd.DataFrame:
        """Add percentage and cumulative contribution columns."""

        frame = mention_frequency.copy()
        total_pairs = int(frame["generated_pair_count"].sum())
        if total_pairs == 0:
            frame["pair_percentage"] = 0.0
            frame["cumulative_percentage"] = 0.0
            return frame

        frame["pair_percentage"] = frame["generated_pair_count"] / total_pairs * 100.0
        frame["cumulative_percentage"] = frame["pair_percentage"].cumsum()
        frame["pair_percentage"] = frame["pair_percentage"].round(4)
        frame["cumulative_percentage"] = frame["cumulative_percentage"].round(4)
        return frame

    def _build_threshold_sweep(self, mention_frequency: pd.DataFrame) -> pd.DataFrame:
        """Evaluate the configured mention thresholds without regenerating pairs."""

        total_mentions = len(mention_frequency)
        total_pairs = int(mention_frequency["generated_pair_count"].sum())
        rows: list[dict[str, Any]] = []

        for threshold in self.config.mention_thresholds:
            retained = mention_frequency[mention_frequency["mention_post_count"] <= threshold]
            candidate_pairs = int(retained["generated_pair_count"].sum())
            mentions_kept = int(len(retained))
            mentions_removed = int(total_mentions - mentions_kept)
            pairs_removed = int(total_pairs - candidate_pairs)
            percent_pairs_retained = (candidate_pairs / total_pairs * 100.0) if total_pairs else 0.0
            percent_pairs_removed = 100.0 - percent_pairs_retained if total_pairs else 0.0
            rows.append(
                {
                    "threshold": threshold,
                    "mentions_kept": mentions_kept,
                    "mentions_removed": mentions_removed,
                    "candidate_pairs": candidate_pairs,
                    "pairs_removed": pairs_removed,
                    "percent_pairs_retained": round(percent_pairs_retained, 4),
                    "percent_pairs_removed": round(percent_pairs_removed, 4),
                }
            )

        return pd.DataFrame(rows)

    def _recommend_threshold(self, threshold_sweep: pd.DataFrame) -> int | None:
        """Pick the elbow threshold using distance from the end-to-end line."""

        if threshold_sweep.empty:
            return None
        if len(threshold_sweep) == 1:
            return int(threshold_sweep.iloc[0]["threshold"])

        x = threshold_sweep["threshold"].to_numpy(dtype=float)
        y = threshold_sweep["candidate_pairs"].to_numpy(dtype=float)
        x_scaled = self._scale(x)
        y_scaled = self._scale(y)

        start = np.array([x_scaled[0], y_scaled[0]], dtype=float)
        end = np.array([x_scaled[-1], y_scaled[-1]], dtype=float)
        line = end - start
        line_norm = np.linalg.norm(line)
        if line_norm == 0:
            return int(threshold_sweep.iloc[-1]["threshold"])

        points = np.column_stack([x_scaled, y_scaled])
        distances = np.abs(
            line[0] * (start[1] - points[:, 1]) - (start[0] - points[:, 0]) * line[1]
        ) / line_norm
        return int(threshold_sweep.iloc[int(np.argmax(distances))]["threshold"])

    def _write_outputs(
        self,
        output_dir: Path,
        mention_frequency: pd.DataFrame,
        pair_contribution: pd.DataFrame,
        threshold_sweep: pd.DataFrame,
    ) -> None:
        """Persist CSV outputs in a reproducible order."""

        mention_frequency[["mentioned_account", "mention_post_count", "generated_pair_count"]].to_csv(
            output_dir / "mention_frequency_analysis.csv",
            index=False,
        )
        pair_contribution[
            [
                "mentioned_account",
                "mention_post_count",
                "generated_pair_count",
                "pair_percentage",
                "cumulative_percentage",
            ]
        ].to_csv(output_dir / "pair_contribution_analysis.csv", index=False)
        threshold_sweep.to_csv(output_dir / "threshold_sweep.csv", index=False)

    def _write_plots(
        self,
        output_dir: Path,
        mention_frequency: pd.DataFrame,
        threshold_sweep: pd.DataFrame,
        recommended_threshold: int | None,
    ) -> None:
        """Save the requested plots to disk."""

        self._plot_threshold_curve(output_dir, threshold_sweep, recommended_threshold)
        self._plot_histogram(output_dir, mention_frequency, log_scale=False)
        self._plot_histogram(output_dir, mention_frequency, log_scale=True)
        self._plot_ccdf(output_dir, mention_frequency)

    def _plot_threshold_curve(
        self,
        output_dir: Path,
        threshold_sweep: pd.DataFrame,
        recommended_threshold: int | None,
    ) -> None:
        """Plot candidate pairs against the sweep thresholds."""

        fig, ax = plt.subplots(figsize=(9, 5))
        ax.plot(threshold_sweep["threshold"], threshold_sweep["candidate_pairs"], marker="o", linewidth=2)
        ax.set_xlabel("MAX_MENTION_POST_COUNT")
        ax.set_ylabel("Total candidate pairs")
        ax.set_title("Mention Threshold Sweep")
        ax.grid(True, alpha=0.25)

        if recommended_threshold is not None:
            row = threshold_sweep[threshold_sweep["threshold"] == recommended_threshold].iloc[0]
            ax.axvline(recommended_threshold, color="crimson", linestyle="--", alpha=0.7)
            ax.scatter([recommended_threshold], [row["candidate_pairs"]], color="crimson", zorder=3)
            ax.annotate(
                f"elbow: {recommended_threshold}",
                xy=(recommended_threshold, row["candidate_pairs"]),
                xytext=(10, 10),
                textcoords="offset points",
                fontsize=9,
                color="crimson",
            )

        fig.tight_layout()
        fig.savefig(output_dir / "threshold_vs_pairs.png", dpi=160)
        plt.close(fig)

    def _plot_histogram(self, output_dir: Path, mention_frequency: pd.DataFrame, *, log_scale: bool) -> None:
        """Plot mention-frequency histograms."""

        values = mention_frequency["mention_post_count"].to_numpy(dtype=int)
        fig, ax = plt.subplots(figsize=(9, 5))
        bins = self._histogram_bins(values)
        ax.hist(values, bins=bins, color="#3b82f6", edgecolor="white")
        ax.set_xlabel("Mention post count")
        ax.set_ylabel("Mentioned accounts")
        ax.set_title("Mention Frequency Distribution")
        ax.grid(True, alpha=0.2)
        if log_scale:
            ax.set_yscale("log")
            file_name = "mention_frequency_log_histogram.png"
        else:
            file_name = "mention_frequency_histogram.png"
        fig.tight_layout()
        fig.savefig(output_dir / file_name, dpi=160)
        plt.close(fig)

    def _plot_ccdf(self, output_dir: Path, mention_frequency: pd.DataFrame) -> None:
        """Plot the complementary cumulative distribution of mention counts."""

        values = np.sort(mention_frequency["mention_post_count"].to_numpy(dtype=float))
        if len(values) == 0:
            values = np.array([0.0])
        y = 1.0 - np.arange(1, len(values) + 1) / len(values)

        fig, ax = plt.subplots(figsize=(9, 5))
        ax.step(values, y, where="post", color="#10b981")
        ax.set_xlabel("Mention post count")
        ax.set_ylabel("CCDF")
        ax.set_title("Mention Frequency CCDF")
        ax.grid(True, alpha=0.2)
        fig.tight_layout()
        fig.savefig(output_dir / "mention_frequency_ccdf.png", dpi=160)
        plt.close(fig)

    def _build_report(
        self,
        mention_frequency: pd.DataFrame,
        pair_contribution: pd.DataFrame,
        threshold_sweep: pd.DataFrame,
        recommended_threshold: int | None,
    ) -> str:
        """Assemble a concise plain-text summary."""

        total_mentions = len(mention_frequency)
        total_pairs = int(mention_frequency["generated_pair_count"].sum())
        top_10 = pair_contribution.head(10)
        lines = [
            f"Total mentioned accounts: {total_mentions}",
            f"Total candidate pairs: {total_pairs}",
            "",
            "Top 10 pair-generating mentions:",
        ]
        for _, row in top_10.iterrows():
            lines.append(
                f"- {row['mentioned_account']}: {int(row['generated_pair_count'])} pairs "
                f"from {int(row['mention_post_count'])} posts"
            )

        lines.extend(["", "Threshold Summary", "Threshold  Pairs  Reduction"])
        for _, row in threshold_sweep.iterrows():
            lines.append(
                f"{int(row['threshold']):<10}{int(row['candidate_pairs']):<8}"
                f"{row['percent_pairs_removed']:.1f}%"
            )

        if recommended_threshold is not None and not threshold_sweep.empty:
            row = threshold_sweep[threshold_sweep["threshold"] == recommended_threshold].iloc[0]
            kept_mentions = int(mention_frequency[mention_frequency["mention_post_count"] <= recommended_threshold].shape[0])
            excluded_mentions = total_mentions - kept_mentions
            excluded_percent = (excluded_mentions / total_mentions * 100.0) if total_mentions else 0.0
            lines.extend(
                [
                    "",
                    f"Recommended threshold: {recommended_threshold}",
                    (
                        "Reason: elbow point from the threshold-vs-pairs curve, where the pair "
                        "growth starts flattening."
                    ),
                    f"Candidate pairs removed: {int(total_pairs - row['candidate_pairs'])}",
                    f"Mentions excluded: {excluded_mentions} ({excluded_percent:.1f}%)",
                ]
            )

        return "\n".join(lines)

    def _to_value_list(self, value: Any) -> list[Any]:
        """Normalize list-like mention values."""

        if value is None or (isinstance(value, float) and np.isnan(value)):
            return []
        if isinstance(value, list):
            values = value
        elif isinstance(value, (tuple, set)):
            values = list(value)
        else:
            return [value]
        return [item for item in values if item is not None and not (isinstance(item, float) and np.isnan(item))]

    def _pair_count(self, mention_post_count: int) -> int:
        """Return n choose 2 for a mention group."""

        return mention_post_count * (mention_post_count - 1) // 2

    def _scale(self, values: np.ndarray) -> np.ndarray:
        """Scale values to [0, 1] for elbow detection."""

        minimum = float(values.min())
        maximum = float(values.max())
        if maximum == minimum:
            return np.zeros_like(values, dtype=float)
        return (values - minimum) / (maximum - minimum)

    def _histogram_bins(self, values: np.ndarray) -> np.ndarray:
        """Build simple histogram bins that keep the plots readable."""

        if len(values) == 0:
            return np.array([0, 1])
        upper = int(values.max())
        return np.arange(0, upper + 2)


def main() -> None:
    """Run the mention-threshold analysis on the repository data files."""

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    root = Path(__file__).resolve().parents[2]
    accounts = load_accounts(str(root / "accounts.jsonl"))
    LOGGER.info("Loaded %d accounts from %s", len(accounts), root / "accounts.jsonl")
    posts = load_posts(str(root / "posts.jsonl"))
    result = MentionThresholdAnalyzer().run(posts, output_dir=root / "analysis")
    LOGGER.info("%s", result.report)
    LOGGER.info("Outputs written to: %s", result.output_dir)


if __name__ == "__main__":
    main()
