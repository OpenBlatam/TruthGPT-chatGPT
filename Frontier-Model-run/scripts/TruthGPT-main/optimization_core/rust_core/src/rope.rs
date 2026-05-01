//! Rotary Position Embeddings (RoPE) for TruthGPT
//!
//! High-performance implementation of RoPE used in LLaMA, Mistral, etc.
//!
//! ## Features
//!
//! - Standard RoPE
//! - NTK-aware scaled RoPE for extended context
//! - YaRN (Yet another RoPE extensioN)
//! - Dynamic NTK scaling

use ndarray::{Array1, Array2, Array3};
use rayon::prelude::*;

// ═══════════════════════════════════════════════════════════════════════════════
// CONFIGURATION
// ═══════════════════════════════════════════════════════════════════════════════

/// RoPE configuration
#[derive(Debug, Clone)]
pub struct RoPEConfig {
    /// Hidden dimension
    pub dim: usize,
    /// Maximum sequence length
    pub max_seq_len: usize,
    /// Base for frequency computation
    pub base: f32,
    /// Scaling type
    pub scaling: RoPEScaling,
}

impl Default for RoPEConfig {
    fn default() -> Self {
        Self {
            dim: 128,
            max_seq_len: 4096,
            base: 10000.0,
            scaling: RoPEScaling::None,
        }
    }
}

/// RoPE scaling strategies for extended context
#[derive(Debug, Clone)]
pub enum RoPEScaling {
    /// No scaling
    None,
    /// Linear interpolation
    Linear { factor: f32 },
    /// NTK-aware scaling
    NTK { factor: f32 },
    /// Dynamic NTK (adjusts based on sequence length)
    DynamicNTK { factor: f32, original_max_len: usize },
    /// YaRN scaling
    YaRN {
        factor: f32,
        beta_fast: f32,
        beta_slow: f32,
        original_max_len: usize,
    },
}

// ═══════════════════════════════════════════════════════════════════════════════
// ROPE EMBEDDINGS
// ═══════════════════════════════════════════════════════════════════════════════

/// Rotary Position Embeddings
pub struct RoPE {
    config: RoPEConfig,
    /// Precomputed cos values [max_seq_len, dim/2]
    cos_cache: Array2<f32>,
    /// Precomputed sin values [max_seq_len, dim/2]
    sin_cache: Array2<f32>,
    /// Inverse frequencies
    inv_freq: Array1<f32>,
}

impl RoPE {
    /// Create new RoPE with given configuration
    pub fn new(config: RoPEConfig) -> Self {
        let _half_dim = config.dim / 2;
        
        // Compute inverse frequencies
        let inv_freq = Self::compute_inv_freq(&config);
        
        // Precompute cos/sin cache
        let (cos_cache, sin_cache) = Self::compute_cache(&config, &inv_freq);
        
        Self {
            config,
            cos_cache,
            sin_cache,
            inv_freq,
        }
    }

    /// Compute inverse frequencies based on scaling
    fn compute_inv_freq(config: &RoPEConfig) -> Array1<f32> {
        let half_dim = config.dim / 2;
        let base = match &config.scaling {
            RoPEScaling::NTK { factor } => {
                // NTK-aware scaling: increase base
                config.base * factor.powf((config.dim as f32) / (config.dim as f32 - 2.0))
            }
            _ => config.base,
        };
        
        Array1::from_iter((0..half_dim).map(|i| {
            1.0 / base.powf(2.0 * i as f32 / config.dim as f32)
        }))
    }

    /// Precompute cos/sin cache
    fn compute_cache(config: &RoPEConfig, inv_freq: &Array1<f32>) -> (Array2<f32>, Array2<f32>) {
        let half_dim = config.dim / 2;
        let max_len = config.max_seq_len;
        
        let mut cos_cache = Array2::zeros((max_len, half_dim));
        let mut sin_cache = Array2::zeros((max_len, half_dim));
        
        for pos in 0..max_len {
            let pos_scaled = match &config.scaling {
                RoPEScaling::Linear { factor } => pos as f32 / factor,
                _ => pos as f32,
            };
            
            for (i, &freq) in inv_freq.iter().enumerate() {
                let angle = pos_scaled * freq;
                cos_cache[[pos, i]] = angle.cos();
                sin_cache[[pos, i]] = angle.sin();
            }
        }
        
        (cos_cache, sin_cache)
    }

    /// Apply RoPE to query and key tensors
    ///
    /// # Arguments
    /// * `q` - Query tensor [batch, seq_len, num_heads, head_dim]
    /// * `k` - Key tensor [batch, seq_len, num_kv_heads, head_dim]
    /// * `position_ids` - Position indices [batch, seq_len]
    ///
    /// # Returns
    /// Rotated (q, k) tensors
    pub fn apply(
        &self,
        q: &Array3<f32>,  // [batch * heads, seq_len, head_dim]
        k: &Array3<f32>,  // [batch * heads, seq_len, head_dim]
        start_pos: usize,
    ) -> (Array3<f32>, Array3<f32>) {
        let (batch_heads, seq_len, head_dim) = q.dim();
        let half_dim = head_dim / 2;
        
        let mut q_rotated = q.clone();
        let mut k_rotated = k.clone();
        
        // Apply rotation
        for b in 0..batch_heads {
            for s in 0..seq_len {
                let pos = start_pos + s;
                
                for i in 0..half_dim {
                    let cos = self.cos_cache[[pos.min(self.config.max_seq_len - 1), i]];
                    let sin = self.sin_cache[[pos.min(self.config.max_seq_len - 1), i]];
                    
                    // Rotate q
                    let q0 = q[[b, s, i]];
                    let q1 = q[[b, s, i + half_dim]];
                    q_rotated[[b, s, i]] = q0 * cos - q1 * sin;
                    q_rotated[[b, s, i + half_dim]] = q0 * sin + q1 * cos;
                    
                    // Rotate k
                    let k0 = k[[b, s, i]];
                    let k1 = k[[b, s, i + half_dim]];
                    k_rotated[[b, s, i]] = k0 * cos - k1 * sin;
                    k_rotated[[b, s, i + half_dim]] = k0 * sin + k1 * cos;
                }
            }
        }
        
        (q_rotated, k_rotated)
    }

    /// Apply RoPE in-place (more efficient)
    pub fn apply_inplace(
        &self,
        q: &mut [f32],  // Flat buffer
        k: &mut [f32],  // Flat buffer
        _batch_size: usize,
        _num_heads: usize,
        seq_len: usize,
        head_dim: usize,
        start_pos: usize,
    ) {
        let half_dim = head_dim / 2;
        
        q.par_chunks_mut(seq_len * head_dim)
            .zip(k.par_chunks_mut(seq_len * head_dim))
            .for_each(|(q_head, k_head)| {
                for s in 0..seq_len {
                    let pos = (start_pos + s).min(self.config.max_seq_len - 1);
                    
                    for i in 0..half_dim {
                        let cos = self.cos_cache[[pos, i]];
                        let sin = self.sin_cache[[pos, i]];
                        
                        let q_idx = s * head_dim;
                        let q0 = q_head[q_idx + i];
                        let q1 = q_head[q_idx + i + half_dim];
                        q_head[q_idx + i] = q0 * cos - q1 * sin;
                        q_head[q_idx + i + half_dim] = q0 * sin + q1 * cos;
                        
                        let k0 = k_head[q_idx + i];
                        let k1 = k_head[q_idx + i + half_dim];
                        k_head[q_idx + i] = k0 * cos - k1 * sin;
                        k_head[q_idx + i + half_dim] = k0 * sin + k1 * cos;
                    }
                }
            });
    }

    /// Extend context length dynamically (Dynamic NTK)
    pub fn extend_context(&mut self, new_max_len: usize) {
        if new_max_len <= self.config.max_seq_len {
            return;
        }
        
        // Recompute with dynamic scaling
        if let RoPEScaling::DynamicNTK { factor: _, original_max_len } = &self.config.scaling {
            let scale = (new_max_len as f32 / *original_max_len as f32).max(1.0);
            let new_base = self.config.base * scale.powf(
                self.config.dim as f32 / (self.config.dim as f32 - 2.0)
            );
            
            // Update config
            self.config.max_seq_len = new_max_len;
            
            // Recompute cache with new base
            let half_dim = self.config.dim / 2;
            self.inv_freq = Array1::from_iter((0..half_dim).map(|i| {
                1.0 / new_base.powf(2.0 * i as f32 / self.config.dim as f32)
            }));
            
            let (cos, sin) = Self::compute_cache(&self.config, &self.inv_freq);
            self.cos_cache = cos;
            self.sin_cache = sin;
        }
    }

    /// Get configuration
    pub fn config(&self) -> &RoPEConfig {
        &self.config
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// YARN IMPLEMENTATION
// ═══════════════════════════════════════════════════════════════════════════════

/// YaRN (Yet another RoPE extensioN) implementation
pub struct YaRN {
    config: RoPEConfig,
    /// Interpolation factors per dimension
    mscale: f32,
    /// Attention scaling factor
    attention_factor: f32,
}

impl YaRN {
    /// Create new YaRN RoPE
    pub fn new(
        dim: usize,
        max_seq_len: usize,
        base: f32,
        factor: f32,
        beta_fast: f32,
        beta_slow: f32,
        original_max_len: usize,
    ) -> Self {
        let mscale = Self::compute_mscale(factor);
        let attention_factor = Self::compute_attention_factor(factor);
        
        Self {
            config: RoPEConfig {
                dim,
                max_seq_len,
                base,
                scaling: RoPEScaling::YaRN {
                    factor,
                    beta_fast,
                    beta_slow,
                    original_max_len,
                },
            },
            mscale,
            attention_factor,
        }
    }

    /// Compute magnitude scaling factor
    fn compute_mscale(factor: f32) -> f32 {
        if factor <= 1.0 {
            1.0
        } else {
            0.1 * factor.ln() + 1.0
        }
    }

    /// Compute attention scaling factor
    fn compute_attention_factor(factor: f32) -> f32 {
        if factor <= 1.0 {
            1.0
        } else {
            0.1 * factor.ln() + 1.0
        }
    }

    /// Get the magnitude scaling factor
    pub fn get_mscale(&self) -> f32 {
        self.mscale
    }

    /// Get attention scaling factor
    pub fn get_attention_factor(&self) -> f32 {
        self.attention_factor
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// ALIBI (Alternative to RoPE)
// ═══════════════════════════════════════════════════════════════════════════════

/// ALiBi (Attention with Linear Biases) implementation
pub struct ALiBi {
    num_heads: usize,
    slopes: Vec<f32>,
}

impl ALiBi {
    /// Create new ALiBi
    pub fn new(num_heads: usize) -> Self {
        let slopes = Self::compute_slopes(num_heads);
        Self { num_heads, slopes }
    }

    /// Compute head-specific slopes
    fn compute_slopes(num_heads: usize) -> Vec<f32> {
        // Geometric sequence of slopes
        let start = 2.0_f32.powf(-(2.0_f32.powf(-(((num_heads as f32).log2()).floor() - 3.0))));
        let ratio = start;
        
        (0..num_heads)
            .map(|i| start * ratio.powi(i as i32))
            .collect()
    }

    /// Compute ALiBi bias matrix
    ///
    /// # Arguments
    /// * `seq_len` - Sequence length
    ///
    /// # Returns
    /// Bias matrix [num_heads, seq_len, seq_len]
    pub fn compute_bias(&self, seq_len: usize) -> Array3<f32> {
        let mut bias = Array3::zeros((self.num_heads, seq_len, seq_len));
        
        for h in 0..self.num_heads {
            let slope = self.slopes[h];
            
            for i in 0..seq_len {
                for j in 0..seq_len {
                    // Causal bias: only positions <= current
                    if j <= i {
                        bias[[h, i, j]] = slope * (j as f32 - i as f32);
                    } else {
                        bias[[h, i, j]] = f32::NEG_INFINITY;
                    }
                }
            }
        }
        
        bias
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// TESTS
// ═══════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rope_creation() {
        let config = RoPEConfig {
            dim: 64,
            max_seq_len: 512,
            base: 10000.0,
            scaling: RoPEScaling::None,
        };
        
        let rope = RoPE::new(config);
        
        assert_eq!(rope.cos_cache.dim(), (512, 32));
        assert_eq!(rope.sin_cache.dim(), (512, 32));
    }

    #[test]
    fn test_rope_apply() {
        let config = RoPEConfig {
            dim: 8,
            max_seq_len: 16,
            base: 10000.0,
            scaling: RoPEScaling::None,
        };
        
        let rope = RoPE::new(config);
        
        // Use non-uniform values to ensure rotation is detectable
        let mut q = Array3::zeros((2, 4, 8));
        let mut k = Array3::zeros((2, 4, 8));
        for i in 0..8 {
            q[[0, 0, i]] = i as f32;
            k[[0, 0, i]] = (i + 1) as f32;
        }
        
        // Apply rotation starting at position 1 to ensure rotation is applied
        let (q_rot, k_rot) = rope.apply(&q, &k, 1);
        
        assert_eq!(q_rot.dim(), (2, 4, 8));
        assert_eq!(k_rot.dim(), (2, 4, 8));
        
        // Verify rotation changed values (at least some values should differ)
        // Check multiple positions to ensure rotation is applied
        let mut changed = false;
        for i in 0..q_rot.dim().2 {
            if (q_rot[[0, 0, i]] - q[[0, 0, i]]).abs() > 1e-6 {
                changed = true;
                break;
            }
        }
        assert!(changed, "RoPE should modify at least some values");
    }

    #[test]
    fn test_alibi() {
        let alibi = ALiBi::new(8);
        let bias = alibi.compute_bias(16);
        
        assert_eq!(bias.dim(), (8, 16, 16));
        
        // Check causal mask
        for h in 0..8 {
            for i in 0..16 {
                for j in (i+1)..16 {
                    assert!(bias[[h, i, j]].is_infinite());
                }
            }
        }
    }

    #[test]
    fn test_ntk_scaling() {
        let config = RoPEConfig {
            dim: 64,
            max_seq_len: 8192,
            base: 10000.0,
            scaling: RoPEScaling::NTK { factor: 2.0 },
        };
        
        let rope = RoPE::new(config);
        
        // NTK should change inv_freq when scaling is applied
        let default_config = RoPEConfig {
            dim: 64,
            max_seq_len: 8192,
            base: 10000.0,
            scaling: RoPEScaling::None,  // No scaling
        };
        let default_rope = RoPE::new(default_config);
        
        // The frequencies should be different when NTK scaling is applied
        // NTK scaling modifies the base frequency, so inv_freq should differ
        // Check that at least some frequencies differ
        let mut frequencies_differ = false;
        for i in 0..rope.inv_freq.len().min(default_rope.inv_freq.len()) {
            if (rope.inv_freq[[i]] - default_rope.inv_freq[[i]]).abs() > 1e-6 {
                frequencies_differ = true;
                break;
            }
        }
        assert!(frequencies_differ, "NTK scaling should change frequencies");
    }

    #[test]
    fn test_yarn_mscale() {
        // factor = 1 -> mscale = 1
        assert!((YaRN::compute_mscale(1.0) - 1.0).abs() < 1e-6);
        
        // factor > 1 -> mscale > 1
        assert!(YaRN::compute_mscale(2.0) > 1.0);
    }
}

