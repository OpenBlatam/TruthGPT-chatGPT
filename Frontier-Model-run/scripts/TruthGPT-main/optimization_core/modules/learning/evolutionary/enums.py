"""
Evolutionary Enums
==================

Enumeration definitions for evolutionary computing methods and strategies.
"""
from enum import Enum


class SelectionMethod(Enum):
    """Selection methods for choosing parents in evolutionary algorithms."""
    ROULETTE_WHEEL = "roulette_wheel"
    TOURNAMENT = "tournament"
    RANK = "rank"
    ELITIST = "elitist"
    STOCHASTIC_UNIVERSAL = "stochastic_universal"
    TRUNCATION = "truncation"


class CrossoverMethod(Enum):
    """Crossover methods for generating offspring."""
    SINGLE_POINT = "single_point"
    TWO_POINT = "two_point"
    UNIFORM = "uniform"
    ARITHMETIC = "arithmetic"
    BLEND = "blend"
    SIMULATED_BINARY = "simulated_binary"


class MutationMethod(Enum):
    """Mutation methods for introducing diversity."""
    GAUSSIAN = "gaussian"
    UNIFORM = "uniform"
    POLYNOMIAL = "polynomial"
    NON_UNIFORM = "non_uniform"
    BOUNDARY = "boundary"
    CREEP = "creep"


class EvolutionaryAlgorithm(Enum):
    """Types of evolutionary algorithms and meta-heuristics."""
    GENETIC_ALGORITHM = "genetic_algorithm"
    EVOLUTIONARY_STRATEGY = "evolutionary_strategy"
    DIFFERENTIAL_EVOLUTION = "differential_evolution"
    GENETIC_PROGRAMMING = "genetic_programming"
    PARTICLE_SWARM = "particle_swarm"
    ANT_COLONY = "ant_colony"

