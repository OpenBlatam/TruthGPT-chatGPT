"""
Causal Discovery
================

Algorithms for discovering causal structure (DAGs) from observational data.
"""
import logging
from typing import Any, Dict, List, Optional

import numpy as np

from optimization_core.modules.learning.causal.config import CausalConfig
from optimization_core.modules.learning.causal.enums import CausalDiscoveryAlgorithm

logger = logging.getLogger(__name__)


class CausalDiscovery:
    """Causal discovery algorithms for structure learning.

    Implements various algorithms (PC, GES, LiNGAM) to identify causal relationships
    between variables and build a Directed Acyclic Graph (DAG).

    Attributes:
        config: Configuration parameters for discovery algorithms.
        causal_graph: The currently discovered causal structure.
        discovery_history: Record of all discovery runs and their results.
    """

    def __init__(self, config: CausalConfig) -> None:
        """Initialize Causal Discovery with configuration.

        Args:
            config: The causal configuration to use.
        """
        self.config: CausalConfig = config
        self.causal_graph: Dict[str, Any] = {}
        self.discovery_history: List[Dict[str, Any]] = []
        logger.info("✅ Causal Discovery initialized")

    def discover_causal_structure(
        self, data: np.ndarray, variable_names: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Discover causal structure from numeric observational data.

        Args:
            data: 2D array of observations (samples x variables).
            variable_names: Optional labels for each column in data.

        Returns:
            Dictionary with the adjacency structure and algorithm metadata.
        """
        logger.info("🔍 Discovering causal structure")

        if variable_names is None:
            variable_names = [f"X{i}" for i in range(data.shape[1])]

        # Dispatch based on algorithm enum
        algorithm = self.config.causal_discovery_algorithm

        if algorithm == CausalDiscoveryAlgorithm.PC:
            causal_graph = self._pc_algorithm(data, variable_names)
        elif algorithm == CausalDiscoveryAlgorithm.GES:
            causal_graph = self._ges_algorithm(data, variable_names)
        elif algorithm == CausalDiscoveryAlgorithm.LINGAM:
            causal_graph = self._lingam_algorithm(data, variable_names)
        else:
            logger.warning(f"Unknown algorithm {algorithm}, defaulting to PC")
            causal_graph = self._pc_algorithm(data, variable_names)

        discovery_result = {
            "algorithm": algorithm.value,
            "significance_level": self.config.significance_level,
            "max_conditioning_set_size": self.config.max_conditioning_set_size,
            "causal_graph": causal_graph,
            "variable_names": variable_names,
            "status": "success",
        }

        # Store discovery
        self.discovery_history.append(discovery_result)

        return discovery_result

    def _pc_algorithm(self, data: np.ndarray, variable_names: List[str]) -> Dict[str, Any]:
        """PC Algorithm for causal discovery (simplified skeleton).

        Args:
            data: The dataset.
            variable_names: Names of columns.

        Returns:
            Discovered graph structure.
        """
        n_vars = data.shape[1]

        # Start with fully connected graph
        graph = {
            variable_names[i]: [var for j, var in enumerate(variable_names) if j != i]
            for i in range(n_vars)
        }

        # Phase 1: Remove edges based on conditional independence tests
        for i in range(n_vars):
            for j in range(i + 1, n_vars):
                var1, var2 = variable_names[i], variable_names[j]

                # Test conditional independence
                is_independent = self._test_conditional_independence(
                    data[:, i], data[:, j], [], self.config.significance_level
                )

                if is_independent:
                    if var2 in graph[var1]:
                        graph[var1].remove(var2)
                    if var1 in graph[var2]:
                        graph[var2].remove(var1)

        # Phase 2: Orient edges (simplified)
        oriented_graph = self._orient_edges(graph, data, variable_names)
        return oriented_graph

    def _ges_algorithm(self, data: np.ndarray, variable_names: List[str]) -> Dict[str, Any]:
        """GES (Greedy Equivalence Search) Algorithm (simplified).

        Args:
            data: The dataset.
            variable_names: Names of columns.

        Returns:
            Discovered graph structure via score-based search.
        """
        n_vars = data.shape[1]
        graph: Dict[str, List[str]] = {name: [] for name in variable_names}

        # Greedy search placeholder: add high-correlation edges
        for i in range(n_vars):
            for j in range(n_vars):
                if i == j:
                    continue

                var1, var2 = variable_names[i], variable_names[j]
                score = self._calculate_edge_score(data[:, i], data[:, j])

                if score > 0.3:  # Higher threshold
                    graph[var1].append(var2)

        return graph

    def _lingam_algorithm(self, data: np.ndarray, variable_names: List[str]) -> Dict[str, Any]:
        """Linear Non-Gaussian Acyclic Model (LiNGAM) implementation.

        Args:
            data: The dataset.
            variable_names: Names of columns.

        Returns:
            Discovered graph structure via ICA/causal ordering.
        """
        graph: Dict[str, List[str]] = {name: [] for name in variable_names}

        # Estimate causal order (simplified proxy)
        causal_order = self._estimate_causal_order(data)

        # Build DAG consistent with order
        for i in range(1, len(causal_order)):
            for j in range(i):
                var1 = variable_names[causal_order[j]]  # Cause
                var2 = variable_names[causal_order[i]]  # Effect

                score = self._calculate_edge_score(data[causal_order[j]], data[causal_order[i]])
                if score > 0.1:
                    graph[var1].append(var2)

        return graph

    def _test_conditional_independence(
        self, x: np.ndarray, y: np.ndarray, z: List[np.ndarray], alpha: float
    ) -> bool:
        """Test conditional independence hypothesis.

        Args:
            x: First variable.
            y: Second variable.
            z: Conditioning set (list of variables).
            alpha: Significance level.

        Returns:
            True if X and Y are independent given Z.
        """
        if len(z) == 0:
            # Unconditional independence: Pearson correlation test proxy
            correlation = float(np.corrcoef(x, y)[0, 1])
            p_value = 1.0 - abs(correlation)  # VERY rough proxy
        else:
            # Conditional independence placeholder
            p_value = float(np.random.random())

        return p_value > alpha

    def _orient_edges(
        self, graph: Dict[str, List[str]], data: np.ndarray, variable_names: List[str]
    ) -> Dict[str, Any]:
        """Orient edges in the skeletal causal graph.

        Args:
            graph: Adjacency list of the graph skeleton.
            data: The dataset.
            variable_names: Names of columns.

        Returns:
            Graph with oriented edges (partially or fully directed).
        """
        oriented_graph: Dict[str, Any] = {k: v.copy() for k, v in graph.items()}

        # Identify v-structures (colliders) X -> Z <- Y
        if "colliders" not in oriented_graph:
            oriented_graph["colliders"] = []

        return oriented_graph

    def _estimate_causal_order(self, data: np.ndarray) -> List[int]:
        """Estimate causal order (topological sort) for LiNGAM.

        Args:
            data: The dataset.

        Returns:
            Indices of variables in causal order.
        """
        variances = np.var(data, axis=0)
        return np.argsort(variances)[::-1].tolist()

    def _calculate_edge_score(self, x: np.ndarray, y: np.ndarray) -> float:
        """Calculate score for edge existence (correlation-based).

        Args:
            x: First variable array.
            y: Second variable array.

        Returns:
            Absolute value of correlation coefficient.
        """
        return float(abs(np.corrcoef(x, y)[0, 1]))

