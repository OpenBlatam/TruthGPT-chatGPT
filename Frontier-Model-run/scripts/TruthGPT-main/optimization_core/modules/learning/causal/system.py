"""
Causal System
=============

Main causal inference system integrator.
"""
import logging
import time
from typing import Any, Dict, List, Optional

import numpy as np

from optimization_core.modules.learning.causal.analysis import (
    RobustnessChecker,
    SensitivityAnalyzer,
)
from optimization_core.modules.learning.causal.config import CausalConfig
from optimization_core.modules.learning.causal.discovery import CausalDiscovery
from optimization_core.modules.learning.causal.estimation import CausalEffectEstimator

logger = logging.getLogger(__name__)


class CausalInferenceSystem:
    """Main causal inference system orchestrator.

    Integrates causal discovery, effect estimation, sensitivity analysis,
    and robustness checking into a unified workflow.

    Attributes:
        config: Configuration for all causal inference stages.
        causal_discovery: Component for uncovering causal DAGs.
        causal_effect_estimator: Component for calculating treatment effects.
        sensitivity_analyzer: Component for testing assumptions via sensitivity.
        robustness_checker: Component for verifying model via refutations.
        causal_inference_history: record of all completed analysis runs.
    """

    def __init__(self, config: CausalConfig) -> None:
        """Initialize the Causal Inference System.

        Args:
            config: The causal configuration to use.
        """
        self.config: CausalConfig = config

        # Components
        self.causal_discovery: CausalDiscovery = CausalDiscovery(config)
        self.causal_effect_estimator: CausalEffectEstimator = CausalEffectEstimator(config)
        self.sensitivity_analyzer: SensitivityAnalyzer = SensitivityAnalyzer(config)
        self.robustness_checker: RobustnessChecker = RobustnessChecker(config)

        # Causal inference state
        self.causal_inference_history: List[Dict[str, Any]] = []

        logger.info("✅ Causal Inference System initialized")

    def run_causal_inference(
        self,
        data: np.ndarray,
        treatment: np.ndarray,
        outcome: np.ndarray,
        covariates: Optional[np.ndarray] = None,
        variable_names: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Run complete causal inference analysis through all configured stages.

        Args:
            data: Observational dataset for discovery.
            treatment: Treatment assignment vector.
            outcome: Outcome variable vector.
            covariates: Optional matrix of confounding variables.
            variable_names: Optional labels for variables in the dataset.

        Returns:
            Dictionary containing results from each stage of calculation.
        """
        logger.info(
            f"🚀 Running causal inference analysis using method: {self.config.causal_method.value}"
        )

        start_time = time.time()
        causal_results: Dict[str, Any] = {
            "start_time": start_time,
            "config": self.config,
            "stages": {},
        }

        # Stage 1: Causal Discovery
        if self.config.enable_causal_discovery:
            logger.info("🔍 Stage 1: Causal Discovery")
            discovery_result = self.causal_discovery.discover_causal_structure(data, variable_names)
            causal_results["stages"]["causal_discovery"] = discovery_result

        # Stage 2: Causal Effect Estimation
        logger.info("📊 Stage 2: Causal Effect Estimation")
        effect_estimation_result = self.causal_effect_estimator.estimate_causal_effect(
            treatment, outcome, covariates
        )
        causal_results["stages"]["causal_effect_estimation"] = effect_estimation_result

        # Determine ATE logic safely
        causal_effect_data = effect_estimation_result.get("causal_effect", {})
        ate = self._extract_ate(causal_effect_data)

        # Stage 3: Sensitivity Analysis
        if self.config.enable_sensitivity_analysis:
            logger.info("🔍 Stage 3: Sensitivity Analysis")
            sensitivity_result = self.sensitivity_analyzer.perform_sensitivity_analysis(
                ate, treatment, outcome, covariates
            )
            causal_results["stages"]["sensitivity_analysis"] = sensitivity_result

        # Stage 4: Robustness Checks
        if self.config.enable_robustness_checks:
            logger.info("🔍 Stage 4: Robustness Checks")
            robustness_result = self.robustness_checker.perform_robustness_checks(
                ate, treatment, outcome, covariates
            )
            causal_results["stages"]["robustness_checks"] = robustness_result

        # Final evaluation
        end_time = time.time()
        causal_results["end_time"] = end_time
        causal_results["total_duration"] = end_time - start_time

        self.causal_inference_history.append(causal_results)

        logger.info("✅ Causal inference analysis completed")
        return causal_results

    def _extract_ate(self, causal_effect_data: Dict[str, Any]) -> float:
        """Helper to extract Average Treatment Effect from results.

        Args:
            causal_effect_data: Output from estimation component.

        Returns:
            Extracted ATE value as a float.
        """
        keys = [
            "average_treatment_effect",
            "propensity_score_matching_effect",
            "difference_in_differences_effect",
            "instrumental_variable_effect",
        ]
        for key in keys:
            if key in causal_effect_data:
                return float(causal_effect_data[key])
        return 0.0

