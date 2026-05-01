//! JSON Processing Module - Ultra-fast JSON parsing and serialization
//!
//! This module provides high-performance JSON operations using simd-json
//! and serde_json for maximum throughput.

use serde::{Deserialize, Serialize};
use serde_json::Value;
use crate::error::{TruthGPTError, Result};

/// JSON Processor with SIMD optimizations
pub struct JsonProcessor {
    /// Use SIMD optimizations when available
    use_simd: bool,
}

impl JsonProcessor {
    /// Create a new JSON processor
    #[must_use]
    pub fn new(use_simd: bool) -> Self {
        Self { use_simd }
    }

    /// Parse JSON string to Value
    pub fn parse(&self, json_str: &str) -> Result<Value> {
        if self.use_simd {
            // Use simd-json for faster parsing
            #[cfg(feature = "simd-json")]
            {
                simd_json::to_borrowed_value(json_str.as_bytes())
                    .map_err(|e| TruthGPTError::Io(format!("JSON parse error: {}", e)))
            }
            #[cfg(not(feature = "simd-json"))]
            {
                serde_json::from_str(json_str)
                    .map_err(|e| TruthGPTError::Io(format!("JSON parse error: {}", e)))
            }
        } else {
            serde_json::from_str(json_str)
                .map_err(|e| TruthGPTError::serialization(format!("JSON parse error: {}", e)))
        }
    }

    /// Serialize Value to JSON string
    pub fn stringify(&self, value: &Value) -> Result<String> {
        serde_json::to_string(value)
            .map_err(|e| TruthGPTError::Io(format!("JSON stringify error: {}", e)))
    }

    /// Parse JSON to typed struct
    pub fn parse_typed<T: for<'de> Deserialize<'de>>(&self, json_str: &str) -> Result<T> {
        serde_json::from_str(json_str)
            .map_err(|e| TruthGPTError::Io(format!("JSON parse error: {}", e)))
    }

    /// Serialize struct to JSON
    pub fn stringify_typed<T: Serialize>(&self, value: &T) -> Result<String> {
        serde_json::to_string(value)
            .map_err(|e| TruthGPTError::Io(format!("JSON stringify error: {}", e)))
    }

    /// Batch parse multiple JSON strings
    pub fn parse_batch(&self, json_strings: &[String]) -> Result<Vec<Value>> {
        json_strings
            .iter()
            .map(|s| self.parse(s))
            .collect()
    }

    /// Extract nested value by path (e.g., "user.name")
    pub fn extract_path<'a>(&self, value: &'a Value, path: &str) -> Option<&'a Value> {
        let parts: Vec<&str> = path.split('.').collect();
        let mut current = value;

        for part in parts {
            match current {
                Value::Object(map) => {
                    current = map.get(part)?;
                }
                Value::Array(arr) => {
                    let idx: usize = part.parse().ok()?;
                    current = arr.get(idx)?;
                }
                _ => return None,
            }
        }

        Some(current)
    }

    /// Merge two JSON objects
    pub fn merge(&self, base: &mut Value, overlay: &Value) -> Result<()> {
        match (base, overlay) {
            (Value::Object(base_map), Value::Object(overlay_map)) => {
                for (key, value) in overlay_map {
                    if let Some(existing) = base_map.get_mut(key) {
                        if existing.is_object() && value.is_object() {
                            self.merge(existing, value)?;
                        } else {
                            *existing = value.clone();
                        }
                    } else {
                        base_map.insert(key.clone(), value.clone());
                    }
                }
                Ok(())
            }
            _ => Err(TruthGPTError::Io("Cannot merge non-object values".to_string())),
        }
    }
}

impl Default for JsonProcessor {
    fn default() -> Self {
        Self::new(true)
    }
}

/// Fast JSON parsing (standalone function)
pub fn fast_parse(json_str: &str) -> Result<Value> {
    serde_json::from_str(json_str)
        .map_err(|e| TruthGPTError::Io(format!("JSON parse error: {}", e)))
}

/// Fast JSON stringification (standalone function)
pub fn fast_stringify(value: &Value) -> Result<String> {
    serde_json::to_string(value)
        .map_err(|e| TruthGPTError::Io(format!("JSON stringify error: {}", e)))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_json_parse() {
        let processor = JsonProcessor::default();
        let json = r#"{"name": "test", "value": 42}"#;
        let value = processor.parse(json).unwrap();
        assert_eq!(value["name"], "test");
        assert_eq!(value["value"], 42);
    }

    #[test]
    fn test_json_stringify() {
        let processor = JsonProcessor::default();
        let mut map = serde_json::Map::new();
        map.insert("name".to_string(), Value::String("test".to_string()));
        map.insert("value".to_string(), Value::Number(serde_json::Number::from(42)));
        let value = Value::Object(map);
        let json = processor.stringify(&value).unwrap();
        assert!(json.contains("test"));
        assert!(json.contains("42"));
    }

    #[test]
    fn test_extract_path() {
        let processor = JsonProcessor::default();
        let json = r#"{"user": {"name": "test", "age": 30}}"#;
        let value = processor.parse(json).unwrap();
        let name = processor.extract_path(&value, "user.name").unwrap();
        assert_eq!(name, "test");
    }
}

