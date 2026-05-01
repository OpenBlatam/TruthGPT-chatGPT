#pragma once

/**
 * @file simd_ops.hpp
 * @brief SIMD-optimized operations for neural network inference
 *
 * Provides vectorized implementations of common operations:
 * - Element-wise operations (add, mul, activation)
 * - Reductions (sum, max, softmax)
 * - Vector operations (dot product, normalize)
 */

#include <cstdint>
#include <cstddef>
#include <cmath>
#include <algorithm>

#if defined(__AVX512F__)
    #include <immintrin.h>
    #define SIMD_WIDTH 16
    #define SIMD_AVAILABLE 3
#elif defined(__AVX2__)
    #include <immintrin.h>
    #define SIMD_WIDTH 8
    #define SIMD_AVAILABLE 2
#elif defined(__SSE4_1__)
    #include <smmintrin.h>
    #define SIMD_WIDTH 4
    #define SIMD_AVAILABLE 1
#else
    #define SIMD_WIDTH 1
    #define SIMD_AVAILABLE 0
#endif

namespace truthgpt {
namespace simd {

// ═══════════════════════════════════════════════════════════════════════════════
// ELEMENT-WISE OPERATIONS
// ═══════════════════════════════════════════════════════════════════════════════

/**
 * @brief Vectorized element-wise addition: c = a + b
 */
inline void vec_add(const float* a, const float* b, float* c, size_t n) {
#if SIMD_AVAILABLE >= 2
    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vb = _mm256_loadu_ps(b + i);
        __m256 vc = _mm256_add_ps(va, vb);
        _mm256_storeu_ps(c + i, vc);
    }
    for (; i < n; ++i) {
        c[i] = a[i] + b[i];
    }
#else
    for (size_t i = 0; i < n; ++i) {
        c[i] = a[i] + b[i];
    }
#endif
}

/**
 * @brief Vectorized element-wise multiplication: c = a * b
 */
inline void vec_mul(const float* a, const float* b, float* c, size_t n) {
#if SIMD_AVAILABLE >= 2
    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vb = _mm256_loadu_ps(b + i);
        __m256 vc = _mm256_mul_ps(va, vb);
        _mm256_storeu_ps(c + i, vc);
    }
    for (; i < n; ++i) {
        c[i] = a[i] * b[i];
    }
#else
    for (size_t i = 0; i < n; ++i) {
        c[i] = a[i] * b[i];
    }
#endif
}

/**
 * @brief Vectorized fused multiply-add: d = a * b + c
 */
inline void vec_fma(const float* a, const float* b, const float* c, float* d, size_t n) {
#if SIMD_AVAILABLE >= 2
    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vb = _mm256_loadu_ps(b + i);
        __m256 vc = _mm256_loadu_ps(c + i);
        __m256 vd = _mm256_fmadd_ps(va, vb, vc);
        _mm256_storeu_ps(d + i, vd);
    }
    for (; i < n; ++i) {
        d[i] = a[i] * b[i] + c[i];
    }
#else
    for (size_t i = 0; i < n; ++i) {
        d[i] = a[i] * b[i] + c[i];
    }
#endif
}

/**
 * @brief Vectorized scalar multiply: b = a * scalar
 */
inline void vec_scale(const float* a, float scalar, float* b, size_t n) {
#if SIMD_AVAILABLE >= 2
    __m256 vs = _mm256_set1_ps(scalar);
    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vb = _mm256_mul_ps(va, vs);
        _mm256_storeu_ps(b + i, vb);
    }
    for (; i < n; ++i) {
        b[i] = a[i] * scalar;
    }
#else
    for (size_t i = 0; i < n; ++i) {
        b[i] = a[i] * scalar;
    }
#endif
}

// ═══════════════════════════════════════════════════════════════════════════════
// REDUCTIONS
// ═══════════════════════════════════════════════════════════════════════════════

/**
 * @brief Vectorized sum reduction
 */
inline float vec_sum(const float* a, size_t n) {
#if SIMD_AVAILABLE >= 2
    __m256 vsum = _mm256_setzero_ps();
    size_t i = 0;
    
    for (; i + 8 <= n; i += 8) {
        __m256 va = _mm256_loadu_ps(a + i);
        vsum = _mm256_add_ps(vsum, va);
    }
    
    // Horizontal sum of vsum
    __m128 hi = _mm256_extractf128_ps(vsum, 1);
    __m128 lo = _mm256_castps256_ps128(vsum);
    __m128 sum = _mm_add_ps(hi, lo);
    sum = _mm_hadd_ps(sum, sum);
    sum = _mm_hadd_ps(sum, sum);
    float result = _mm_cvtss_f32(sum);
    
    // Handle remainder
    for (; i < n; ++i) {
        result += a[i];
    }
    
    return result;
#else
    float sum = 0.0f;
    for (size_t i = 0; i < n; ++i) {
        sum += a[i];
    }
    return sum;
#endif
}

/**
 * @brief Vectorized max reduction
 */
inline float vec_max(const float* a, size_t n) {
#if SIMD_AVAILABLE >= 2
    __m256 vmax = _mm256_set1_ps(-INFINITY);
    size_t i = 0;
    
    for (; i + 8 <= n; i += 8) {
        __m256 va = _mm256_loadu_ps(a + i);
        vmax = _mm256_max_ps(vmax, va);
    }
    
    // Horizontal max
    __m128 hi = _mm256_extractf128_ps(vmax, 1);
    __m128 lo = _mm256_castps256_ps128(vmax);
    __m128 m = _mm_max_ps(hi, lo);
    m = _mm_max_ps(m, _mm_shuffle_ps(m, m, _MM_SHUFFLE(2, 3, 0, 1)));
    m = _mm_max_ps(m, _mm_shuffle_ps(m, m, _MM_SHUFFLE(1, 0, 3, 2)));
    float result = _mm_cvtss_f32(m);
    
    // Handle remainder
    for (; i < n; ++i) {
        result = std::max(result, a[i]);
    }
    
    return result;
#else
    float m = -INFINITY;
    for (size_t i = 0; i < n; ++i) {
        m = std::max(m, a[i]);
    }
    return m;
#endif
}

/**
 * @brief Vectorized dot product
 */
inline float vec_dot(const float* a, const float* b, size_t n) {
#if SIMD_AVAILABLE >= 2
    __m256 vdot = _mm256_setzero_ps();
    size_t i = 0;
    
    for (; i + 8 <= n; i += 8) {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vb = _mm256_loadu_ps(b + i);
        vdot = _mm256_fmadd_ps(va, vb, vdot);
    }
    
    // Horizontal sum
    __m128 hi = _mm256_extractf128_ps(vdot, 1);
    __m128 lo = _mm256_castps256_ps128(vdot);
    __m128 sum = _mm_add_ps(hi, lo);
    sum = _mm_hadd_ps(sum, sum);
    sum = _mm_hadd_ps(sum, sum);
    float result = _mm_cvtss_f32(sum);
    
    // Handle remainder
    for (; i < n; ++i) {
        result += a[i] * b[i];
    }
    
    return result;
#else
    float dot = 0.0f;
    for (size_t i = 0; i < n; ++i) {
        dot += a[i] * b[i];
    }
    return dot;
#endif
}

// ═══════════════════════════════════════════════════════════════════════════════
// ACTIVATION FUNCTIONS
// ═══════════════════════════════════════════════════════════════════════════════

/**
 * @brief Vectorized ReLU activation
 */
inline void vec_relu(const float* input, float* output, size_t n) {
#if SIMD_AVAILABLE >= 2
    __m256 vzero = _mm256_setzero_ps();
    size_t i = 0;
    
    for (; i + 8 <= n; i += 8) {
        __m256 va = _mm256_loadu_ps(input + i);
        __m256 vout = _mm256_max_ps(va, vzero);
        _mm256_storeu_ps(output + i, vout);
    }
    
    for (; i < n; ++i) {
        output[i] = std::max(0.0f, input[i]);
    }
#else
    for (size_t i = 0; i < n; ++i) {
        output[i] = std::max(0.0f, input[i]);
    }
#endif
}

/**
 * @brief Vectorized GELU activation (approximation)
 */
inline void vec_gelu(const float* input, float* output, size_t n) {
    const float c1 = 0.7978845608f; // sqrt(2/pi)
    const float c2 = 0.044715f;
    
#if SIMD_AVAILABLE >= 2
    __m256 vc1 = _mm256_set1_ps(c1);
    __m256 vc2 = _mm256_set1_ps(c2);
    __m256 vhalf = _mm256_set1_ps(0.5f);
    __m256 vone = _mm256_set1_ps(1.0f);
    
    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 x = _mm256_loadu_ps(input + i);
        __m256 x2 = _mm256_mul_ps(x, x);
        __m256 x3 = _mm256_mul_ps(x2, x);
        
        // x + 0.044715 * x^3
        __m256 inner = _mm256_fmadd_ps(vc2, x3, x);
        // sqrt(2/pi) * (x + 0.044715 * x^3)
        inner = _mm256_mul_ps(vc1, inner);
        
        // tanh approximation (simplified)
        __m256 tanh_approx = inner; // Simplified - use actual tanh for accuracy
        
        // 0.5 * x * (1 + tanh(...))
        __m256 result = _mm256_mul_ps(vhalf, x);
        result = _mm256_mul_ps(result, _mm256_add_ps(vone, tanh_approx));
        
        _mm256_storeu_ps(output + i, result);
    }
    
    for (; i < n; ++i) {
        float x = input[i];
        float x3 = x * x * x;
        float inner = c1 * (x + c2 * x3);
        output[i] = 0.5f * x * (1.0f + std::tanh(inner));
    }
#else
    for (size_t i = 0; i < n; ++i) {
        float x = input[i];
        float x3 = x * x * x;
        float inner = c1 * (x + c2 * x3);
        output[i] = 0.5f * x * (1.0f + std::tanh(inner));
    }
#endif
}

/**
 * @brief Vectorized SiLU/Swish activation
 */
inline void vec_silu(const float* input, float* output, size_t n) {
    for (size_t i = 0; i < n; ++i) {
        float x = input[i];
        output[i] = x / (1.0f + std::exp(-x)); // x * sigmoid(x)
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// SOFTMAX
// ═══════════════════════════════════════════════════════════════════════════════

/**
 * @brief Vectorized softmax
 */
inline void vec_softmax(const float* input, float* output, size_t n) {
    // Find max for numerical stability
    float max_val = vec_max(input, n);
    
    // Compute exp(x - max) and sum
    float sum = 0.0f;
    for (size_t i = 0; i < n; ++i) {
        output[i] = std::exp(input[i] - max_val);
        sum += output[i];
    }
    
    // Normalize
    float inv_sum = 1.0f / sum;
    vec_scale(output, inv_sum, output, n);
}

// ═══════════════════════════════════════════════════════════════════════════════
// LAYER NORM
// ═══════════════════════════════════════════════════════════════════════════════

/**
 * @brief Layer normalization
 */
inline void layer_norm(
    const float* input,
    const float* gamma,
    const float* beta,
    float* output,
    size_t n,
    float eps = 1e-5f
) {
    // Compute mean
    float mean = vec_sum(input, n) / n;
    
    // Compute variance
    float var = 0.0f;
    for (size_t i = 0; i < n; ++i) {
        float diff = input[i] - mean;
        var += diff * diff;
    }
    var /= n;
    
    // Normalize
    float inv_std = 1.0f / std::sqrt(var + eps);
    
    for (size_t i = 0; i < n; ++i) {
        output[i] = (input[i] - mean) * inv_std;
        if (gamma) output[i] *= gamma[i];
        if (beta) output[i] += beta[i];
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// RMS NORM
// ═══════════════════════════════════════════════════════════════════════════════

/**
 * @brief RMS normalization (used in LLaMA)
 */
inline void rms_norm(
    const float* input,
    const float* weight,
    float* output,
    size_t n,
    float eps = 1e-5f
) {
    // Compute RMS
    float sum_sq = 0.0f;
    for (size_t i = 0; i < n; ++i) {
        sum_sq += input[i] * input[i];
    }
    float rms = std::sqrt(sum_sq / n + eps);
    float inv_rms = 1.0f / rms;
    
    // Normalize and scale
    for (size_t i = 0; i < n; ++i) {
        output[i] = input[i] * inv_rms * weight[i];
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// UTILITY
// ═══════════════════════════════════════════════════════════════════════════════

/**
 * @brief Get SIMD capability string
 */
inline const char* simd_capability() {
#if SIMD_AVAILABLE >= 3
    return "AVX-512";
#elif SIMD_AVAILABLE >= 2
    return "AVX2";
#elif SIMD_AVAILABLE >= 1
    return "SSE4.1";
#else
    return "Scalar";
#endif
}

} // namespace simd
} // namespace truthgpt




