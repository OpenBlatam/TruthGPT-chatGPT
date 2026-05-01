#pragma once

/**
 * @file quantize.hpp
 * @brief High-performance quantization for LLM inference
 * 
 * Supports:
 * - INT8 symmetric/asymmetric
 * - INT4 packed
 * - FP16/BF16 conversion
 * - Per-channel and per-tensor quantization
 */

#include <cstdint>
#include <vector>
#include <algorithm>
#include <cmath>
#include <immintrin.h>

namespace truthgpt {
namespace quantization {

// ═══════════════════════════════════════════════════════════════════════════════
// QUANTIZATION TYPES
// ═══════════════════════════════════════════════════════════════════════════════

enum class QuantType {
    FP32,
    FP16,
    BF16,
    INT8,
    INT4
};

/**
 * @brief Quantization parameters for INT8/INT4
 */
struct QuantParams {
    float scale;
    int32_t zero_point;
    int32_t min_val;
    int32_t max_val;
    
    static QuantParams int8_symmetric(float abs_max) {
        return {abs_max / 127.0f, 0, -127, 127};
    }
    
    static QuantParams int8_asymmetric(float min_val, float max_val) {
        float scale = (max_val - min_val) / 255.0f;
        int32_t zp = static_cast<int32_t>(std::round(-min_val / scale));
        return {scale, std::clamp(zp, 0, 255), 0, 255};
    }
    
    static QuantParams int4_symmetric(float abs_max) {
        return {abs_max / 7.0f, 0, -8, 7};
    }
};

// ═══════════════════════════════════════════════════════════════════════════════
// SCALAR QUANTIZATION
// ═══════════════════════════════════════════════════════════════════════════════

/**
 * @brief Quantize single FP32 value to INT8
 */
inline int8_t quantize_int8(float value, const QuantParams& params) {
    float scaled = value / params.scale + params.zero_point;
    return static_cast<int8_t>(std::clamp(
        static_cast<int32_t>(std::round(scaled)),
        params.min_val,
        params.max_val
    ));
}

/**
 * @brief Dequantize single INT8 value to FP32
 */
inline float dequantize_int8(int8_t value, const QuantParams& params) {
    return (static_cast<float>(value) - params.zero_point) * params.scale;
}

// ═══════════════════════════════════════════════════════════════════════════════
// VECTORIZED QUANTIZATION (AVX2/AVX-512)
// ═══════════════════════════════════════════════════════════════════════════════

#ifdef __AVX2__

/**
 * @brief AVX2-accelerated INT8 quantization
 * @param input Input FP32 array
 * @param output Output INT8 array  
 * @param n Number of elements
 * @param scale Quantization scale
 */
inline void quantize_int8_avx2(
    const float* input,
    int8_t* output,
    size_t n,
    float scale
) {
    const __m256 vscale = _mm256_set1_ps(1.0f / scale);
    const __m256 vmin = _mm256_set1_ps(-127.0f);
    const __m256 vmax = _mm256_set1_ps(127.0f);
    
    size_t i = 0;
    
    // Process 8 floats at a time
    for (; i + 8 <= n; i += 8) {
        __m256 vx = _mm256_loadu_ps(input + i);
        vx = _mm256_mul_ps(vx, vscale);
        vx = _mm256_round_ps(vx, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
        vx = _mm256_max_ps(vx, vmin);
        vx = _mm256_min_ps(vx, vmax);
        
        // Convert to int32
        __m256i vi = _mm256_cvtps_epi32(vx);
        
        // Pack to int8
        __m128i lo = _mm256_castsi256_si128(vi);
        __m128i hi = _mm256_extracti128_si256(vi, 1);
        __m128i packed16 = _mm_packs_epi32(lo, hi);
        __m128i packed8 = _mm_packs_epi16(packed16, packed16);
        
        _mm_storel_epi64(reinterpret_cast<__m128i*>(output + i), packed8);
    }
    
    // Handle remainder
    for (; i < n; ++i) {
        float scaled = input[i] / scale;
        output[i] = static_cast<int8_t>(std::clamp(
            static_cast<int32_t>(std::round(scaled)), -127, 127
        ));
    }
}

/**
 * @brief AVX2-accelerated INT8 dequantization
 */
inline void dequantize_int8_avx2(
    const int8_t* input,
    float* output,
    size_t n,
    float scale
) {
    const __m256 vscale = _mm256_set1_ps(scale);
    
    size_t i = 0;
    
    // Process 8 values at a time
    for (; i + 8 <= n; i += 8) {
        // Load 8 int8 values
        __m128i v8 = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(input + i));
        
        // Sign-extend to int16
        __m128i v16 = _mm_cvtepi8_epi16(v8);
        
        // Sign-extend to int32 (lower 4)
        __m128i v32_lo = _mm_cvtepi16_epi32(v16);
        // Sign-extend to int32 (upper 4)
        __m128i v32_hi = _mm_cvtepi16_epi32(_mm_srli_si128(v16, 8));
        
        // Combine to __m256i
        __m256i v32 = _mm256_insertf128_si256(
            _mm256_castsi128_si256(v32_lo), v32_hi, 1
        );
        
        // Convert to float and scale
        __m256 vf = _mm256_cvtepi32_ps(v32);
        vf = _mm256_mul_ps(vf, vscale);
        
        _mm256_storeu_ps(output + i, vf);
    }
    
    // Handle remainder
    for (; i < n; ++i) {
        output[i] = static_cast<float>(input[i]) * scale;
    }
}

#endif // __AVX2__

// ═══════════════════════════════════════════════════════════════════════════════
// FP16/BF16 CONVERSION
// ═══════════════════════════════════════════════════════════════════════════════

/**
 * @brief Convert FP32 to FP16
 */
inline uint16_t float_to_fp16(float value) {
#ifdef __F16C__
    return _cvtss_sh(value, 0);
#else
    // Software fallback
    uint32_t bits;
    std::memcpy(&bits, &value, sizeof(bits));
    
    uint32_t sign = (bits >> 16) & 0x8000;
    int32_t exp = ((bits >> 23) & 0xFF) - 127 + 15;
    uint32_t mantissa = bits & 0x7FFFFF;
    
    if (exp <= 0) {
        return sign; // Flush to zero
    } else if (exp >= 31) {
        return sign | 0x7C00; // Infinity
    }
    
    return sign | (exp << 10) | (mantissa >> 13);
#endif
}

/**
 * @brief Convert FP16 to FP32
 */
inline float fp16_to_float(uint16_t value) {
#ifdef __F16C__
    return _cvtsh_ss(value);
#else
    // Software fallback
    uint32_t sign = (value & 0x8000) << 16;
    uint32_t exp = (value >> 10) & 0x1F;
    uint32_t mantissa = value & 0x3FF;
    
    if (exp == 0) {
        if (mantissa == 0) {
            uint32_t result = sign;
            float f;
            std::memcpy(&f, &result, sizeof(f));
            return f;
        }
        // Denormalized
        exp = 1;
        while ((mantissa & 0x400) == 0) {
            mantissa <<= 1;
            exp--;
        }
        mantissa &= 0x3FF;
    } else if (exp == 31) {
        uint32_t result = sign | 0x7F800000 | (mantissa << 13);
        float f;
        std::memcpy(&f, &result, sizeof(f));
        return f;
    }
    
    uint32_t result = sign | ((exp + 127 - 15) << 23) | (mantissa << 13);
    float f;
    std::memcpy(&f, &result, sizeof(f));
    return f;
#endif
}

/**
 * @brief Convert FP32 to BF16
 */
inline uint16_t float_to_bf16(float value) {
    uint32_t bits;
    std::memcpy(&bits, &value, sizeof(bits));
    
    // Round to nearest even
    uint32_t rounding = 0x7FFF + ((bits >> 16) & 1);
    bits += rounding;
    
    return static_cast<uint16_t>(bits >> 16);
}

/**
 * @brief Convert BF16 to FP32
 */
inline float bf16_to_float(uint16_t value) {
    uint32_t bits = static_cast<uint32_t>(value) << 16;
    float f;
    std::memcpy(&f, &bits, sizeof(f));
    return f;
}

// ═══════════════════════════════════════════════════════════════════════════════
// INT4 PACKED QUANTIZATION
// ═══════════════════════════════════════════════════════════════════════════════

/**
 * @brief Quantize FP32 to packed INT4 (2 values per byte)
 */
inline void quantize_int4_packed(
    const float* input,
    uint8_t* output,
    size_t n,
    float scale
) {
    for (size_t i = 0; i < n; i += 2) {
        float v0 = input[i] / scale;
        float v1 = (i + 1 < n) ? input[i + 1] / scale : 0.0f;
        
        int8_t q0 = static_cast<int8_t>(std::clamp(
            static_cast<int32_t>(std::round(v0)), -8, 7
        ));
        int8_t q1 = static_cast<int8_t>(std::clamp(
            static_cast<int32_t>(std::round(v1)), -8, 7
        ));
        
        // Pack: lower nibble = q0, upper nibble = q1
        output[i / 2] = ((q0 + 8) & 0x0F) | (((q1 + 8) & 0x0F) << 4);
    }
}

/**
 * @brief Dequantize packed INT4 to FP32
 */
inline void dequantize_int4_packed(
    const uint8_t* input,
    float* output,
    size_t n,
    float scale
) {
    for (size_t i = 0; i < n; i += 2) {
        uint8_t packed = input[i / 2];
        
        int8_t v0 = static_cast<int8_t>(packed & 0x0F) - 8;
        int8_t v1 = static_cast<int8_t>((packed >> 4) & 0x0F) - 8;
        
        output[i] = static_cast<float>(v0) * scale;
        if (i + 1 < n) {
            output[i + 1] = static_cast<float>(v1) * scale;
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// QUANTIZED MATMUL
// ═══════════════════════════════════════════════════════════════════════════════

/**
 * @brief INT8 matrix-vector multiplication
 * @param input Input vector [K]
 * @param weights Weight matrix [N, K] row-major
 * @param output Output vector [N]
 * @param N Number of output features
 * @param K Number of input features
 * @param scale Weight quantization scale
 */
inline void matmul_int8(
    const float* input,
    const int8_t* weights,
    float* output,
    size_t N,
    size_t K,
    float scale
) {
#ifdef __AVX2__
    const __m256 vscale_sq = _mm256_set1_ps(scale * scale);
    
    #pragma omp parallel for
    for (size_t i = 0; i < N; ++i) {
        __m256i vacc = _mm256_setzero_si256();
        
        size_t j = 0;
        for (; j + 32 <= K; j += 32) {
            // Load weights
            __m256i vw = _mm256_loadu_si256(
                reinterpret_cast<const __m256i*>(weights + i * K + j)
            );
            
            // Load and quantize input
            __m256 vx0 = _mm256_loadu_ps(input + j);
            __m256 vx1 = _mm256_loadu_ps(input + j + 8);
            __m256 vx2 = _mm256_loadu_ps(input + j + 16);
            __m256 vx3 = _mm256_loadu_ps(input + j + 24);
            
            // Scale and round
            const __m256 vinv_scale = _mm256_set1_ps(1.0f / scale);
            vx0 = _mm256_mul_ps(vx0, vinv_scale);
            vx1 = _mm256_mul_ps(vx1, vinv_scale);
            vx2 = _mm256_mul_ps(vx2, vinv_scale);
            vx3 = _mm256_mul_ps(vx3, vinv_scale);
            
            // Simplified accumulation (actual impl would use VNNI)
        }
        
        // Horizontal sum and scale
        int32_t sum = 0;
        for (size_t k = 0; k < K; ++k) {
            int32_t w = weights[i * K + k];
            int32_t x = static_cast<int32_t>(std::round(input[k] / scale));
            sum += w * x;
        }
        
        output[i] = static_cast<float>(sum) * scale * scale;
    }
#else
    #pragma omp parallel for
    for (size_t i = 0; i < N; ++i) {
        int32_t sum = 0;
        for (size_t k = 0; k < K; ++k) {
            int32_t w = weights[i * K + k];
            int32_t x = static_cast<int32_t>(std::round(input[k] / scale));
            sum += w * x;
        }
        output[i] = static_cast<float>(sum) * scale * scale;
    }
#endif
}

} // namespace quantization
} // namespace truthgpt




