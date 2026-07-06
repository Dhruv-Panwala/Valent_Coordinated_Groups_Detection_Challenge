"""Detect weighted account communities from a coordination graph."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from time import perf_counter
from typing import Any

import igraph as ig
import leidenalg
import networkx as nx
import pandas as pd

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class CommunityDetectionConfig:
    """Configuration for Leiden community detection."""

    weight_attribute: str = "weight"
    seed: int = 42
    n_iterations: int = 2


class CommunityDetector:
    """Detect communities from a weighted undirected account graph."""

    def __init__(self, config: CommunityDetectionConfig | None = None) -> None:
        self.config = config or CommunityDetectionConfig()

    def transform(self, graph: nx.Graph) -> tuple[pd.DataFrame, dict[int, list[Any]]]:
        """Run Leiden community detection and return memberships and groups."""

        start = perf_counter()
        if graph.number_of_nodes() == 0:
            LOGGER.info("Algorithm used: Leiden")
            LOGGER.info("Number of communities: 0")
            LOGGER.info("Largest community size: 0")
            LOGGER.info("Smallest community size: 0")
            LOGGER.info("Average community size: 0.00")
            LOGGER.info("Modularity score: unavailable")
            LOGGER.info("Runtime: %.2fs", perf_counter() - start)
            return self._empty_output()

        igraph_graph, node_order = self._to_igraph(graph)
        partition = leidenalg.find_partition(
            igraph_graph,
            leidenalg.ModularityVertexPartition,
            weights=self.config.weight_attribute,
            n_iterations=self.config.n_iterations,
            seed=self.config.seed,
        )

        communities = self._relabel_communities(partition, node_order)
        membership_frame = self._build_membership_frame(communities)
        self._log_diagnostics(partition, communities, start)
        return membership_frame, communities

    def _to_igraph(self, graph: nx.Graph) -> tuple[ig.Graph, list[Any]]:
        """Convert a NetworkX graph to igraph with stable node ordering."""

        node_order = sorted(graph.nodes(), key=str)
        index_lookup = {node: index for index, node in enumerate(node_order)}
        igraph_graph = ig.Graph()
        igraph_graph.add_vertices(len(node_order))
        igraph_graph.vs["name"] = [str(node) for node in node_order]

        edges: list[tuple[int, int]] = []
        weights: list[float] = []
        for left, right, data in graph.edges(data=True):
            edges.append((index_lookup[left], index_lookup[right]))
            weights.append(float(data.get(self.config.weight_attribute, 1.0)))

        igraph_graph.add_edges(edges)
        igraph_graph.es[self.config.weight_attribute] = weights
        return igraph_graph, node_order

    def _relabel_communities(
        self,
        partition: leidenalg.VertexPartition,
        node_order: list[Any],
    ) -> dict[int, list[Any]]:
        """Convert Leiden membership assignments into deterministic community ids."""

        raw_groups: list[list[Any]] = []
        for community in partition:
            raw_groups.append(sorted((node_order[index] for index in community), key=str))

        raw_groups.sort(key=lambda group: (-len(group), [str(node) for node in group]))
        return {community_id: group for community_id, group in enumerate(raw_groups)}

    def _build_membership_frame(self, communities: dict[int, list[Any]]) -> pd.DataFrame:
        """Build a long-form community membership table."""

        rows = [
            {"community_id": community_id, "account_id": account_id}
            for community_id, members in communities.items()
            for account_id in members
        ]
        return pd.DataFrame(rows, columns=["community_id", "account_id"])

    def _log_diagnostics(
        self,
        partition: leidenalg.VertexPartition,
        communities: dict[int, list[Any]],
        start: float,
    ) -> None:
        """Log community-detection diagnostics."""

        sizes = [len(members) for members in communities.values()]
        LOGGER.info("Algorithm used: Leiden")
        LOGGER.info("Number of communities: %d", len(communities))
        LOGGER.info("Largest community size: %d", max(sizes, default=0))
        LOGGER.info("Smallest community size: %d", min(sizes, default=0))
        LOGGER.info("Average community size: %.2f", float(sum(sizes) / len(sizes)) if sizes else 0.0)
        modularity = getattr(partition, "modularity", None)
        if modularity is None:
            LOGGER.info("Modularity score: unavailable")
        else:
            LOGGER.info("Modularity score: %.6f", float(modularity))
        LOGGER.info("Runtime: %.2fs", perf_counter() - start)

    def _empty_output(self) -> tuple[pd.DataFrame, dict[int, list[Any]]]:
        """Return empty community outputs."""

        return pd.DataFrame(columns=["community_id", "account_id"]), {}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    graph = nx.Graph()
    graph.add_edge("a1", "a2", weight=0.9)
    graph.add_edge("a2", "a3", weight=0.8)
    graph.add_edge("b1", "b2", weight=0.7)
    membership, communities = CommunityDetector().transform(graph)
    assert list(membership.columns) == ["community_id", "account_id"]
    assert isinstance(communities, dict)
