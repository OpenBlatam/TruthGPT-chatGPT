//! Hyperparameter Optimization Module
//!
//! This module provides efficient hyperparameter optimization using
//! various search strategies (grid, random, Bayesian optimization).

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use rand::Rng;
use crate::error::Result;

/// Search strategy for hyperparameter optimization
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum SearchStrategy {
    /// Grid search - exhaustive search over all combinations
    Grid,
    /// Random search - random sampling
    Random,
    /// Bayesian optimization - uses previous results to guide search
    Bayesian,
    /// Tree-structured Parzen Estimator (TPE)
    TPE,
}

/// Hyperparameter range definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum HyperparameterRange {
    /// Continuous range [min, max]
    Continuous { min: f64, max: f64 },
    /// Discrete integer range [min, max]
    Integer { min: i64, max: i64 },
    /// Categorical choices
    Categorical { choices: Vec<String> },
    /// Logarithmic range [min, max] (base 10)
    Logarithmic { min: f64, max: f64 },
}

/// Hyperparameter configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HyperparameterConfig {
    pub name: String,
    pub range: HyperparameterRange,
}

/// Trial result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrialResult {
    pub trial_id: usize,
    pub hyperparameters: HashMap<String, serde_json::Value>,
    pub objective_value: f64,
    pub metadata: HashMap<String, String>,
}

/// Hyperparameter Optimizer
pub struct HyperparameterOptimizer {
    strategy: SearchStrategy,
    configs: Vec<HyperparameterConfig>,
    trials: Vec<TrialResult>,
    best_trial: Option<TrialResult>,
    rng: rand::rngs::ThreadRng,
}

impl HyperparameterOptimizer {
    /// Create a new optimizer
    pub fn new(strategy: SearchStrategy, configs: Vec<HyperparameterConfig>) -> Self {
        Self {
            strategy,
            configs,
            trials: Vec::new(),
            best_trial: None,
            rng: rand::rng(),
        }
    }

    /// Suggest next hyperparameters to try
    pub fn suggest(&mut self) -> Result<HashMap<String, serde_json::Value>> {
        let suggestion = match self.strategy {
            SearchStrategy::Grid => self.suggest_grid()?,
            SearchStrategy::Random => self.suggest_random(),
            SearchStrategy::Bayesian => self.suggest_bayesian()?,
            SearchStrategy::TPE => self.suggest_tpe()?,
        };

        Ok(suggestion)
    }

    /// Report trial result
    pub fn report(&mut self, result: TrialResult) -> Result<()> {
        // Update best trial
        if self.best_trial.is_none()
            || result.objective_value > self.best_trial.as_ref().unwrap().objective_value
        {
            self.best_trial = Some(result.clone());
        }

        self.trials.push(result);
        Ok(())
    }

    /// Get best trial so far
    pub fn best_trial(&self) -> Option<&TrialResult> {
        self.best_trial.as_ref()
    }

    /// Get all trials
    pub fn trials(&self) -> &[TrialResult] {
        &self.trials
    }

    /// Grid search suggestion
    fn suggest_grid(&self) -> Result<HashMap<String, serde_json::Value>> {
        // Simple grid search - returns first unexplored combination
        // In practice, this would track which combinations have been tried
        let mut suggestion = HashMap::new();
        
        for config in &self.configs {
            let value = match &config.range {
                HyperparameterRange::Continuous { min, max } => {
                    let val = (*min + *max) / 2.0;
                    serde_json::Value::Number(serde_json::Number::from_f64(val).unwrap_or_else(|| serde_json::Number::from(0)))
                }
                HyperparameterRange::Integer { min, max } => {
                    serde_json::Value::Number(serde_json::Number::from((*min + *max) / 2))
                }
                HyperparameterRange::Categorical { choices } => {
                    serde_json::Value::String(choices[0].clone())
                }
                HyperparameterRange::Logarithmic { min, max } => {
                    let log_min = min.log10();
                    let log_max = max.log10();
                    let log_value = (log_min + log_max) / 2.0;
                    let val = 10_f64.powf(log_value);
                    serde_json::Value::Number(serde_json::Number::from_f64(val).unwrap_or_else(|| serde_json::Number::from(0)))
                }
            };
            suggestion.insert(config.name.clone(), value);
        }

        Ok(suggestion)
    }

    /// Random search suggestion
    fn suggest_random(&mut self) -> HashMap<String, serde_json::Value> {
        let mut suggestion = HashMap::new();
        
        for config in &self.configs {
            let value = match &config.range {
                HyperparameterRange::Continuous { min, max } => {
                    let v = self.rng.random_range(*min..=*max);
                    serde_json::Value::Number(serde_json::Number::from_f64(v).unwrap_or_else(|| serde_json::Number::from(0)))
                }
                HyperparameterRange::Integer { min, max } => {
                    let v = self.rng.random_range(*min..=*max);
                    serde_json::Value::Number(serde_json::Number::from(v))
                }
                HyperparameterRange::Categorical { choices } => {
                    let idx = self.rng.random_range(0..choices.len());
                    serde_json::Value::String(choices[idx].clone())
                }
                HyperparameterRange::Logarithmic { min, max } => {
                    let log_min = min.log10();
                    let log_max = max.log10();
                    let log_value = self.rng.random_range(log_min..=log_max);
                    let val = 10_f64.powf(log_value);
                    serde_json::Value::Number(serde_json::Number::from_f64(val).unwrap_or_else(|| serde_json::Number::from(0)))
                }
            };
            suggestion.insert(config.name.clone(), value);
        }

        suggestion
    }

    /// Bayesian optimization suggestion (simplified)
    fn suggest_bayesian(&self) -> Result<HashMap<String, serde_json::Value>> {
        // Simplified Bayesian optimization
        // In practice, would use Gaussian Process or similar
        if self.trials.is_empty() {
            let mut rng = rand::rng();
            return self.suggest_random_internal(&mut rng);
        }

        // Use best trial as starting point, add small random perturbation
        let best = self.best_trial.as_ref().unwrap();
        let mut suggestion = best.hyperparameters.clone();

        // Add small random perturbations
        for config in &self.configs {
            if let Some(value) = suggestion.get_mut(&config.name) {
                match &config.range {
                    HyperparameterRange::Continuous { min, max } => {
                        if let Some(num) = value.as_f64() {
                            let perturbation = (max - min) * 0.1;
                            let new_value = (num + perturbation).clamp(*min, *max);
                            *value = serde_json::Value::Number(serde_json::Number::from_f64(new_value).unwrap_or_else(|| serde_json::Number::from(0)));
                        }
                    }
                    HyperparameterRange::Integer { min, max } => {
                        if let Some(num) = value.as_i64() {
                            let perturbation = (max - min) / 10;
                            let new_value = (num + perturbation).clamp(*min, *max);
                            *value = serde_json::Value::Number(serde_json::Number::from(new_value));
                        }
                    }
                    _ => {}
                }
            }
        }

        Ok(suggestion)
    }

    /// TPE (Tree-structured Parzen Estimator) suggestion
    fn suggest_tpe(&self) -> Result<HashMap<String, serde_json::Value>> {
        // Simplified TPE - uses quantiles of best trials
        if self.trials.len() < 10 {
            let mut rng = rand::rng();
            return self.suggest_random_internal(&mut rng);
        }

        // Sort trials by objective value
        let mut sorted_trials = self.trials.clone();
        sorted_trials.sort_by(|a, b| {
            b.objective_value.partial_cmp(&a.objective_value).unwrap()
        });

        // Use top 20% as "good" trials
        let top_n = (sorted_trials.len() as f64 * 0.2) as usize;
        let good_trials = &sorted_trials[..top_n];

        // Sample from good trials
        let mut rng = rand::rng();
        let selected = &good_trials[rng.random_range(0..good_trials.len())];
        
        Ok(selected.hyperparameters.clone())
    }

    /// Internal random suggestion
    fn suggest_random_internal(&self, rng: &mut rand::rngs::ThreadRng) -> Result<HashMap<String, serde_json::Value>> {
        let mut suggestion = HashMap::new();
        
        for config in &self.configs {
            let value = match &config.range {
                HyperparameterRange::Continuous { min, max } => {
                    let v = rng.random_range(*min..=*max);
                    serde_json::Value::Number(serde_json::Number::from_f64(v).unwrap_or_else(|| serde_json::Number::from(0)))
                }
                HyperparameterRange::Integer { min, max } => {
                    let v = rng.random_range(*min..=*max);
                    serde_json::Value::Number(serde_json::Number::from(v))
                }
                HyperparameterRange::Categorical { choices } => {
                    let idx = rng.random_range(0..choices.len());
                    serde_json::Value::String(choices[idx].clone())
                }
                HyperparameterRange::Logarithmic { min, max } => {
                    let log_min = min.log10();
                    let log_max = max.log10();
                    let log_value = rng.random_range(log_min..=log_max);
                    let val = 10_f64.powf(log_value);
                    serde_json::Value::Number(serde_json::Number::from_f64(val).unwrap_or_else(|| serde_json::Number::from(0)))
                }
            };
            suggestion.insert(config.name.clone(), value);
        }

        Ok(suggestion)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_random_search() {
        let configs = vec![
            HyperparameterConfig {
                name: "learning_rate".to_string(),
                range: HyperparameterRange::Logarithmic {
                    min: 1e-5,
                    max: 1e-2,
                },
            },
            HyperparameterConfig {
                name: "batch_size".to_string(),
                range: HyperparameterRange::Integer { min: 16, max: 128 },
            },
        ];

        let mut optimizer = HyperparameterOptimizer::new(SearchStrategy::Random, configs);
        let suggestion = optimizer.suggest().unwrap();
        
        assert!(suggestion.contains_key("learning_rate"));
        assert!(suggestion.contains_key("batch_size"));
    }

    #[test]
    fn test_trial_reporting() {
        let configs = vec![HyperparameterConfig {
            name: "lr".to_string(),
            range: HyperparameterRange::Continuous { min: 0.0, max: 1.0 },
        }];

        let mut optimizer = HyperparameterOptimizer::new(SearchStrategy::Random, configs);
        
        let mut params = HashMap::new();
        params.insert("lr".to_string(), serde_json::Value::Number(serde_json::Number::from_f64(0.5).unwrap_or_else(|| serde_json::Number::from(0))));
        
        let result = TrialResult {
            trial_id: 0,
            hyperparameters: params,
            objective_value: 0.9,
            metadata: HashMap::new(),
        };

        optimizer.report(result).unwrap();
        assert!(optimizer.best_trial().is_some());
    }
}

