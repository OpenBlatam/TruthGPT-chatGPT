"""
Causal Enums
============

Enumeration definitions for causal inference methods and discovery algorithms.
"""
from enum import Enum


class CausalMethod(Enum):
    """Estimation methods for causal effects."""

    RCT = "rct"
    IV = "iv"
    PSM = "psm"
    DID = "did"
    REGRESSION_DISCONTINUITY = "regression_discontinuity"
    SYNTHETIC_CONTROL = "synthetic_control"
    RANDOMIZED_CONTROLLED_TRIAL = "randomized_controlled_trial"
    INSTRUMENTAL_VARIABLES = "instrumental_variables"
    DIFFERENCE_IN_DIFFERENCES = "difference_in_differences"
    PROPENSITY_SCORE_MATCHING = "propensity_score_matching"
    CAUSAL_DISCOVERY = "causal_discovery"
    STRUCTURAL_EQUATION_MODELING = "structural_equation_modeling"


class CausalEffectType(Enum):
    """Types of causal effects to estimate."""

    ATE = "ate"  # Average Causal Effect
    ATT = "att"  # Average Causal Effect on Treated
    ATC = "atc"  # Average Causal Effect on Control
    ITE = "ite"  # Individual Causal Effect
    CATE = "cate"  # Conditional Average Causal Effect
    AVERAGE_TREATMENT_EFFECT = "average_treatment_effect"
    LOCAL_AVERAGE_TREATMENT_EFFECT = "local_average_treatment_effect"
    COMPLIER_AVERAGE_TREATMENT_EFFECT = "complier_average_treatment_effect"
    CONDITIONAL_AVERAGE_TREATMENT_EFFECT = "conditional_average_treatment_effect"
    QUANTILE_TREATMENT_EFFECT = "quantile_treatment_effect"
    MARGINAL_TREATMENT_EFFECT = "marginal_treatment_effect"


class CausalDiscoveryAlgorithm(Enum):
    """Algorithms for discovering causal structures (DAGs)."""

    PC = "pc"
    GES = "ges"
    LINGAM = "lingam"
    FCI = "fci"
    NOTEARS = "notears"
    DYNOTEARS = "dynotears"

