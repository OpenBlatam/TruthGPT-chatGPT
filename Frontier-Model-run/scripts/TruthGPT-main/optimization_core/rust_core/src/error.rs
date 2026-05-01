//! Error Module - Unified Error Handling for TruthGPT Rust Core
//!
//! Provides consistent error types with Python integration via PyO3.

use thiserror::Error;

#[cfg(feature = "python")]
use pyo3::exceptions::{PyIOError, PyRuntimeError, PyValueError};
#[cfg(feature = "python")]
use pyo3::prelude::*;

/// Main error type for the TruthGPT Rust Core module.
#[derive(Error, Debug)]
pub enum TruthGPTError {
    /// KV Cache errors
    #[error("Cache error: {0}")]
    Cache(String),

    /// Compression errors
    #[error("Compression error: {0}")]
    Compression(String),

    /// Tokenization errors
    #[error("Tokenization error: {0}")]
    Tokenization(String),

    /// Attention computation errors
    #[error("Attention error: {0}")]
    Attention(String),

    /// Data loading errors
    #[error("Data loading error: {0}")]
    DataLoading(String),

    /// Model errors
    #[error("Model error: {0}")]
    Model(String),

    /// Tensor/array errors
    #[error("Tensor error: {0}")]
    Tensor(String),

    /// I/O errors
    #[error("I/O error: {0}")]
    Io(String),

    /// Configuration errors
    #[error("Configuration error: {0}")]
    Config(String),

    /// Internal errors
    #[error("Internal error: {0}")]
    Internal(String),

    /// Serialization errors
    #[error("Serialization error: {0}")]
    Serialization(String),
}

impl TruthGPTError {
    /// Create a cache error
    pub fn cache(msg: impl Into<String>) -> Self {
        Self::Cache(msg.into())
    }

    /// Create a compression error
    pub fn compression(msg: impl Into<String>) -> Self {
        Self::Compression(msg.into())
    }

    /// Create a tokenization error
    pub fn tokenization(msg: impl Into<String>) -> Self {
        Self::Tokenization(msg.into())
    }

    /// Create an attention error
    pub fn attention(msg: impl Into<String>) -> Self {
        Self::Attention(msg.into())
    }

    /// Create a data loading error
    pub fn data_loading(msg: impl Into<String>) -> Self {
        Self::DataLoading(msg.into())
    }

    /// Create a model error
    pub fn model(msg: impl Into<String>) -> Self {
        Self::Model(msg.into())
    }

    /// Create a tensor error
    pub fn tensor(msg: impl Into<String>) -> Self {
        Self::Tensor(msg.into())
    }

    /// Create an I/O error
    pub fn io(msg: impl Into<String>) -> Self {
        Self::Io(msg.into())
    }

    /// Create a configuration error
    pub fn config(msg: impl Into<String>) -> Self {
        Self::Config(msg.into())
    }

    /// Create an internal error
    pub fn internal(msg: impl Into<String>) -> Self {
        Self::Internal(msg.into())
    }

    /// Create a serialization error
    pub fn serialization(msg: impl Into<String>) -> Self {
        Self::Serialization(msg.into())
    }
}

#[cfg(feature = "python")]
impl From<TruthGPTError> for PyErr {
    fn from(err: TruthGPTError) -> PyErr {
        match err {
            TruthGPTError::Cache(_) | TruthGPTError::Attention(_) | TruthGPTError::Model(_) => {
                PyRuntimeError::new_err(err.to_string())
            }
            TruthGPTError::Compression(_) | TruthGPTError::Tensor(_) => {
                PyRuntimeError::new_err(err.to_string())
            }
            TruthGPTError::Tokenization(_) | TruthGPTError::Config(_) => {
                PyValueError::new_err(err.to_string())
            }
            TruthGPTError::Io(_) | TruthGPTError::DataLoading(_) => {
                PyIOError::new_err(err.to_string())
            }
            TruthGPTError::Internal(_) | TruthGPTError::Serialization(_) => PyRuntimeError::new_err(err.to_string()),
        }
    }
}

impl From<std::io::Error> for TruthGPTError {
    fn from(err: std::io::Error) -> Self {
        TruthGPTError::Io(err.to_string())
    }
}

impl From<serde_json::Error> for TruthGPTError {
    fn from(err: serde_json::Error) -> Self {
        TruthGPTError::DataLoading(err.to_string())
    }
}

impl From<anyhow::Error> for TruthGPTError {
    fn from(err: anyhow::Error) -> Self {
        TruthGPTError::Internal(err.to_string())
    }
}

/// Result type alias for TruthGPTError
pub type Result<T> = std::result::Result<T, TruthGPTError>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_display() {
        let err = TruthGPTError::cache("test error");
        assert_eq!(err.to_string(), "Cache error: test error");
    }

    #[test]
    fn test_error_constructors() {
        assert!(matches!(TruthGPTError::cache("test"), TruthGPTError::Cache(_)));
        assert!(matches!(
            TruthGPTError::compression("test"),
            TruthGPTError::Compression(_)
        ));
        assert!(matches!(
            TruthGPTError::tokenization("test"),
            TruthGPTError::Tokenization(_)
        ));
    }
}

