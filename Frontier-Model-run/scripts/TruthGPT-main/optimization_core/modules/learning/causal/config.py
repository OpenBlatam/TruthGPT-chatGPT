"""
Causal Configuration
====================

Configuration class for causal inference systems, discovery settings, and estimation parameters.
"""
from dataclasses import dataclass
from typing import Optional

from optimization_core.modules.learning.causal.enums import (
    CausalDiscoveryAlgorithm,
    CausalEffectType,
    CausalMethod,
)


@dataclass
class CausalConfig:
    """Configuration for causal inference system.

    Attributes:
        causal_method: Primary method for effect estimation (e.g., PSM, IV).
        causal_effect_type: Type of effect to calculate (e.g., ATE).
        enable_causal_discovery: Whether to perform causal discovery.
        causal_discovery_algorithm: Algorithm for structure learning.
        significance_level: Alpha level for statistical significance tests.
        max_conditioning_set_size: Maximum size of conditioning set in PC.
        enable_instrumental_variables: Whether to allow IV estimation.
        iv_estimation_method: Algorithm for IV (e.g., 2SLS).
        iv_robust_standard_errors: Use heteroscedasticity-robust SEs in IV.
        enable_propensity_score_matching: Whether to allow PSM.
        propensity_score_method: Model for propensity scores (e.g., Logistic).
        matching_algorithm: Strategy for matching (e.g., nearest neighbor).
        caliper: Distance threshold for valid matches.
        enable_difference_in_differences: Whether to allow DiD.
        did_estimation_method: Algorithm for DiD (e.g., TWFE).
        did_cluster_standard_errors: Enable clustered SEs for DiD.
        enable_sensitivity_analysis: Whether to perform sensitivity analysis.
        enable_robustness_checks: Whether to run placebo and falsification tests.
        enable_heterogeneity_analysis: Whether to check for varied effects across groups.
    """

    # Basic settings
    causal_method: CausalMethod = CausalMethod.RANDOMIZED_CONTROLLED_TRIAL
    causal_effect_type: CausalEffectType = CausalEffectType.AVERAGE_TREATMENT_EFFECT

    # Causal discovery settings
    enable_causal_discovery: bool = True
    causal_discovery_algorithm: CausalDiscoveryAlgorithm = CausalDiscoveryAlgorithm.PC
    significance_level: float = 0.05
    max_conditioning_set_size: int = 3

    # Instrumental variables settings
    enable_instrumental_variables: bool = True
    iv_estimation_method: str = "two_stage_least_squares"
    iv_robust_standard_errors: bool = True

    # Propensity score settings
    enable_propensity_score_matching: bool = True
    propensity_score_method: str = "logistic_regression"
    matching_algorithm: str = "nearest_neighbor"
    caliper: float = 0.1

    # Difference-in-differences settings
    enable_difference_in_differences: bool = True
    did_estimation_method: str = "two_way_fixed_effects"
    did_cluster_standard_errors: bool = True

    # Advanced features
    enable_sensitivity_analysis: bool = True
    enable_robustness_checks: bool = True
    enable_heterogeneity_analysis: bool = True

    def __post_init__(self) -> None:
        """Validate causal inference configuration parameters.

        Raises:
            ValueError: If configuration parameters are out of valid ranges.
        """
        if not (0 < self.significance_level < 1):
            raise ValueError("Significance level must be between 0 and 1")
        if self.max_conditioning_set_size <= 0:
            raise ValueError("Max conditioning set size must be positive")
        if not (0 < self.caliper < 1):
            raise ValueError("Caliper must be between 0 and 1")

