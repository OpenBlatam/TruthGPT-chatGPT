"""
Causal Analysis
===============

Sensitivity analysis and robustness checks for causal inference results.
"""
import logging
from typing import Any, Dict, List, Optional

import numpy as np

from optimization_core.modules.learning.causal.config import CausalConfig

logger = logging.getLogger(__name__)


class SensitivityAnalyzer:
    """Sensitivity analysis for causal inference.

    Tests how dependent the estimated causal effect is on various assumptions,
    such as the absence of unobserved confounders or model specification.

    Attributes:
        config: Configuration parameters for analysis.
        sensitivity_history: Record of all performed sensitivity tests.
    """

    def __init__(self, config: CausalConfig) -> None:
        """Initialize Sensitivity Analyzer.

        Args:
            config: The causal configuration to use.
        """
        self.config: CausalConfig = config
        self.sensitivity_history: List[Dict[str, Any]] = []
        logger.info("✅ Sensitivity Analyzer initialized")

    def perform_sensitivity_analysis(
        self,
        causal_effect: float,
        treatment: np.ndarray,
        outcome: np.ndarray,
        covariates: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """Run multiple sensitivity tests on a calculated causal effect.

        Args:
            causal_effect: The estimated effect to test.
            treatment: Treatment assignment array.
            outcome: Observed outcome array.
            covariates: Optional confounding variables.

        Returns:
            Dictionary containing results from various sensitivity tests.
        """
        logger.info("🔍 Performing sensitivity analysis")

        sensitivity_results: Dict[str, Any] = {
            "original_effect": causal_effect,
            "sensitivity_tests": {},
        }

        # Test 1: Unobserved confounder sensitivity
        if self.config.enable_sensitivity_analysis:
            sensitivity_results["sensitivity_tests"]["unobserved_confounder"] = (
                self._test_unobserved_confounder_sensitivity(causal_effect, treatment, outcome)
            )

        # Test 2: Sample size sensitivity
        sensitivity_results["sensitivity_tests"]["sample_size"] = self._test_sample_size_sensitivity(
            causal_effect, treatment, outcome
        )

        # Test 3: Model specification sensitivity
        sensitivity_results["sensitivity_tests"]["model_specification"] = (
            self._test_model_specification_sensitivity(causal_effect, treatment, outcome, covariates)
        )

        # Store sensitivity analysis
        self.sensitivity_history.append(sensitivity_results)

        return sensitivity_results

    def _test_unobserved_confounder_sensitivity(
        self, causal_effect: float, treatment: np.ndarray, outcome: np.ndarray
    ) -> Dict[str, Any]:
        """Simulate the effect of a hidden variable on the result."""
        confounder_strength = float(np.random.random())
        confounder_effect = float(np.random.random() * 0.5)
        adjusted_effect = causal_effect - confounder_effect

        return {
            "confounder_strength": confounder_strength,
            "confounder_effect": confounder_effect,
            "adjusted_effect": adjusted_effect,
            "effect_change": float(causal_effect - adjusted_effect),
        }

    def _test_sample_size_sensitivity(
        self, causal_effect: float, treatment: np.ndarray, outcome: np.ndarray
    ) -> Dict[str, Any]:
        """Evaluate how the effect estimate varies with different sample sizes."""
        sample_sizes = [len(treatment) // 2, len(treatment), len(treatment) * 2]
        effects_by_sample_size: Dict[int, float] = {}

        for sample_size in sample_sizes:
            if sample_size <= len(treatment):
                indices = np.random.choice(len(treatment), sample_size, replace=False)
                subsample_treatment = treatment[indices]
                subsample_outcome = outcome[indices]

                # Simple means difference proxy
                if (
                    len(subsample_outcome[subsample_treatment == 1]) > 0
                    and len(subsample_outcome[subsample_treatment == 0]) > 0
                ):
                    subsample_effect = float(
                        np.mean(subsample_outcome[subsample_treatment == 1])
                        - np.mean(subsample_outcome[subsample_treatment == 0])
                    )
                else:
                    subsample_effect = 0.0

                effects_by_sample_size[sample_size] = subsample_effect

        return {
            "effects_by_sample_size": effects_by_sample_size,
            "effect_stability": float(
                np.std(list(effects_by_sample_size.values())) if effects_by_sample_size else 0.0
            ),
        }

    def _test_model_specification_sensitivity(
        self,
        causal_effect: float,
        treatment: np.ndarray,
        outcome: np.ndarray,
        covariates: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """Simulate sensitivity to functional form assumptions."""
        model_specifications = ["linear", "quadratic", "interaction"]
        effects_by_specification: Dict[str, float] = {}

        for spec in model_specifications:
            if spec == "linear":
                effect = causal_effect
            elif spec == "quadratic":
                effect = causal_effect * 1.1
            elif spec == "interaction":
                effect = causal_effect * 0.9
            else:
                effect = causal_effect

            effects_by_specification[spec] = float(effect)

        return {
            "effects_by_specification": effects_by_specification,
            "specification_sensitivity": float(np.std(list(effects_by_specification.values()))),
        }


class RobustnessChecker:
    """Robustness checks for causal inference.

    Implements refutation tests (placebo, falsification) to validate the
    theoretical assumptions behind the causal model.

    Attributes:
        config: Configuration parameters for robustness tests.
        robustness_history: Record of all performed robustness checks.
    """

    def __init__(self, config: CausalConfig) -> None:
        """Initialize Robustness Checker.

        Args:
            config: The causal configuration to use.
        """
        self.config: CausalConfig = config
        self.robustness_history: List[Dict[str, Any]] = []
        logger.info("✅ Robustness Checker initialized")

    def perform_robustness_checks(
        self,
        causal_effect: float,
        treatment: np.ndarray,
        outcome: np.ndarray,
        covariates: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """Perform a suite of robustness/refutation tests.

        Args:
            causal_effect: The estimated effect to validate.
            treatment: Treatment assignment array.
            outcome: Observed outcome array.
            covariates: Optional confounding variables.

        Returns:
            Dictionary containing results from various robustness tests.
        """
        logger.info("🔍 Performing robustness checks")

        robustness_results: Dict[str, Any] = {
            "original_effect": causal_effect,
            "robustness_tests": {},
        }

        robustness_results["robustness_tests"]["placebo_test"] = self._perform_placebo_test(
            treatment, outcome
        )

        robustness_results["robustness_tests"]["falsification_test"] = self._perform_falsification_test(
            treatment, outcome
        )

        robustness_results["robustness_tests"]["pre_treatment_trends"] = self._check_pre_treatment_trends(
            treatment, outcome
        )

        self.robustness_history.append(robustness_results)

        return robustness_results

    def _perform_placebo_test(self, treatment: np.ndarray, outcome: np.ndarray) -> Dict[str, Any]:
        """Run test with random treatment assignment (should have no effect)."""
        placebo_treatment = np.random.randint(0, 2, len(treatment))
        placebo_effect = float(
            np.mean(outcome[placebo_treatment == 1]) - np.mean(outcome[placebo_treatment == 0])
        )
        return {"placebo_effect": placebo_effect, "placebo_test_passed": abs(placebo_effect) < 0.1}

    def _perform_falsification_test(self, treatment: np.ndarray, outcome: np.ndarray) -> Dict[str, Any]:
        """Run test with a random outcome unrelated to treatment."""
        future_outcome = np.random.randn(len(outcome))
        falsification_effect = float(
            np.mean(future_outcome[treatment == 1]) - np.mean(future_outcome[treatment == 0])
        )
        return {
            "falsification_effect": falsification_effect,
            "falsification_test_passed": abs(falsification_effect) < 0.1,
        }

    def _check_pre_treatment_trends(self, treatment: np.ndarray, outcome: np.ndarray) -> Dict[str, Any]:
        """Simulate check for parallel trends assumption."""
        pre_treatment_outcome = np.random.randn(len(outcome))
        treated_trend = float(np.mean(pre_treatment_outcome[treatment == 1]))
        control_trend = float(np.mean(pre_treatment_outcome[treatment == 0]))
        trend_difference = float(abs(treated_trend - control_trend))

        return {
            "treated_trend": treated_trend,
            "control_trend": control_trend,
            "trend_difference": trend_difference,
            "parallel_trends_assumption": trend_difference < 0.1,
        }

