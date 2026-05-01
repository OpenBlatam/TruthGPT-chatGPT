# Attention Kernels C++ Refactoring Analysis

## Overview

This document analyzes `attention_kernels.cpp` to identify repetitive patterns that can be abstracted into reusable helper functions and utilities.

---

## 1. Code Review

### File Analyzed

- **File:** `src/attention/attention_kernels.cpp`
- **Lines:** 373
- **Namespace:** `optimization_core::attention`

---

## 2. Repetitive Patterns Identified

### Pattern 1: TBB vs OpenMP Parallelization ⚠️ HIGH PRIORITY

**Location:** Multiple functions (3 occurrences)

**Problem:** The same TBB vs OpenMP pattern is repeated in multiple functions with nearly identical structure.

**Examples:**

**Location 1:** `batch_softmax()` (lines 67-83)
```cpp
#ifdef HAVE_TBB
tbb::parallel_for(tbb::blocked_range<int>(0, scores.rows()),
    [&](const tbb::blocked_range<int>& r) {
        for (int i = r.begin(); i != r.end(); ++i) {
            Eigen::VectorXf row = scores.row(i);
            eigen_softmax(row);
            scores.row(i) = row;
        }
    });
#else
#pragma omp parallel for
for (int i = 0; i < scores.rows(); ++i) {
    Eigen::VectorXf row = scores.row(i);
    eigen_softmax(row);
    scores.row(i) = row;
}
#endif
```

**Location 2:** `multi_head_attention()` (lines 148-179)
```cpp
#ifdef HAVE_TBB
tbb::parallel_for(0, n_heads, [&](int h) {
    int start = h * head_dim;
    // ... head processing ...
});
#else
#pragma omp parallel for
for (int h = 0; h < n_heads; ++h) {
    int start = h * head_dim;
    // ... same head processing ...
}
#endif
```

**Location 3:** `grouped_query_attention()` (lines 283-317)
```cpp
#ifdef HAVE_TBB
tbb::parallel_for(0, n_kv_heads, [&](int kv_h) {
    // ... KV head processing ...
});
#else
#pragma omp parallel for
for (int kv_h = 0; kv_h < n_kv_heads; ++kv_h) {
    // ... same KV head processing ...
}
#endif
```

**Pattern Analysis:**
- **Same conditional compilation pattern**: `#ifdef HAVE_TBB` / `#else` / `#endif`
- **Same parallelization logic**: TBB uses lambda, OpenMP uses pragma
- **Same loop body**: Identical code in both branches
- **Only difference**: Parallelization mechanism

**Opportunity:** Create a helper macro or template function that abstracts the parallelization mechanism.

---

### Pattern 2: Head Slice Extraction ⚠️ HIGH PRIORITY

**Location:** `multi_head_attention()` and `grouped_query_attention()` (4+ occurrences)

**Problem:** Repeated pattern of extracting head slices from projected matrices.

**Examples:**

**Location 1:** `multi_head_attention()` - TBB version (lines 154-156)
```cpp
Eigen::MatrixXf Q_h = Q_proj.middleCols(start, head_dim);
Eigen::MatrixXf K_h = K_proj.middleCols(start, head_dim);
Eigen::MatrixXf V_h = V_proj.middleCols(start, head_dim);
```

**Location 2:** `multi_head_attention()` - OpenMP version (lines 170-172)
```cpp
Eigen::MatrixXf Q_h = Q_proj.middleCols(start, head_dim);
Eigen::MatrixXf K_h = K_proj.middleCols(start, head_dim);
Eigen::MatrixXf V_h = V_proj.middleCols(start, head_dim);
```

**Location 3:** `grouped_query_attention()` - TBB version (lines 286-287, 292)
```cpp
Eigen::MatrixXf K_h = K.middleCols(kv_h * head_dim, head_dim);
Eigen::MatrixXf V_h = V.middleCols(kv_h * head_dim, head_dim);
// ...
Eigen::MatrixXf Q_h = Q.middleCols(q_h * head_dim, head_dim);
```

**Location 4:** `grouped_query_attention()` - OpenMP version (lines 304-305, 309)
```cpp
Eigen::MatrixXf K_h = K.middleCols(kv_h * head_dim, head_dim);
Eigen::MatrixXf V_h = V.middleCols(kv_h * head_dim, head_dim);
// ...
Eigen::MatrixXf Q_h = Q.middleCols(q_h * head_dim, head_dim);
```

**Pattern Analysis:**
- **Same extraction pattern**: `matrix.middleCols(start, head_dim)`
- **Same variable naming**: `Q_h`, `K_h`, `V_h`
- **Same usage**: Extract → compute → write back

**Opportunity:** Create helper functions or struct to encapsulate head slice extraction.

---

### Pattern 3: Scale Calculation ⚠️ MEDIUM PRIORITY

**Location:** Multiple functions

**Problem:** Repeated calculation of attention scale factor.

**Examples:**

**Location 1:** `multi_head_attention()` (line 137)
```cpp
float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
```

**Location 2:** `scaled_dot_product_attention()` (parameter, but computed elsewhere)
```cpp
// scale is passed as parameter, but computed as:
// float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
```

**Pattern Analysis:**
- **Same formula**: `1.0f / std::sqrt(static_cast<float>(head_dim))`
- **Same type casting**: `static_cast<float>(head_dim)`
- **Used in multiple places**: Different functions compute it

**Opportunity:** Create inline helper function for scale calculation.

---

### Pattern 4: Causal Mask Application ⚠️ MEDIUM PRIORITY

**Location:** `flash_attention_block()` (lines 220-228)

**Problem:** Nested loop pattern for applying causal mask could be abstracted.

**Example:**
```cpp
if (causal) {
    for (int i = 0; i < seq_q; ++i) {
        for (int k = 0; k < block_len; ++k) {
            if (j + k > i) {
                S_j(i, k) = -std::numeric_limits<float>::infinity();
            }
        }
    }
}
```

**Pattern Analysis:**
- **Same mask value**: `-std::numeric_limits<float>::infinity()`
- **Same condition**: `j + k > i` (causal check)
- **Could be reused**: Other attention functions might need similar masking

**Opportunity:** Create helper function for applying causal mask to a block.

---

### Pattern 5: Matrix Initialization ⚠️ LOW PRIORITY

**Location:** Multiple functions

**Problem:** Repeated pattern of initializing output matrices.

**Examples:**

**Location 1:** `multi_head_attention()` (line 145)
```cpp
Eigen::MatrixXf output = Eigen::MatrixXf::Zero(seq_len, d_model);
```

**Location 2:** `flash_attention_block()` (lines 204-206)
```cpp
Eigen::MatrixXf O = Eigen::MatrixXf::Zero(seq_q, head_dim);
Eigen::VectorXf m = Eigen::VectorXf::Constant(seq_q, -std::numeric_limits<float>::infinity());
Eigen::VectorXf l = Eigen::VectorXf::Zero(seq_q);
```

**Location 3:** `grouped_query_attention()` (line 280)
```cpp
Eigen::MatrixXf output = Eigen::MatrixXf::Zero(seq_len, n_heads * head_dim);
```

**Pattern Analysis:**
- **Same initialization pattern**: `Eigen::MatrixXf::Zero(...)`
- **Similar for vectors**: `Eigen::VectorXf::Zero(...)` or `Constant(...)`
- **Could be abstracted**: But might be too simple to warrant abstraction

**Opportunity:** Consider helper functions if initialization becomes more complex.

---

## 3. Proposed Helper Functions

### Helper 1: Parallel Execution Macro/Function

**File:** `include/attention/parallel_utils.hpp` (new file)

**Purpose:** Abstract TBB vs OpenMP parallelization

**Option A: Macro-based (simpler, less type-safe)**
```cpp
/**
 * @brief Parallel for loop that works with TBB or OpenMP
 * 
 * Usage:
 *   PARALLEL_FOR(i, 0, n, {
 *       // loop body
 *   });
 */
#ifdef HAVE_TBB
#define PARALLEL_FOR(index, start, end, body) \
    tbb::parallel_for(start, end, [&](int index) { body });
#elif defined(HAVE_OPENMP)
#define PARALLEL_FOR(index, start, end, body) \
    _Pragma("omp parallel for") \
    for (int index = start; index < end; ++index) { body }
#else
#define PARALLEL_FOR(index, start, end, body) \
    for (int index = start; index < end; ++index) { body }
#endif
```

**Option B: Template function (type-safe, more flexible)**
```cpp
/**
 * @brief Execute function in parallel over range
 * 
 * @tparam Func Function type (callable)
 * @param start Start index (inclusive)
 * @param end End index (exclusive)
 * @param func Function to execute for each index
 */
template<typename Func>
void parallel_for(int start, int end, Func&& func) {
#ifdef HAVE_TBB
    tbb::parallel_for(start, end, std::forward<Func>(func));
#elif defined(HAVE_OPENMP)
    #pragma omp parallel for
    for (int i = start; i < end; ++i) {
        func(i);
    }
#else
    for (int i = start; i < end; ++i) {
        func(i);
    }
#endif
}
```

**Recommendation:** Use Option B (template function) for type safety and flexibility.

---

### Helper 2: Head Slice Extraction

**File:** `include/attention/head_utils.hpp` (new file)

**Purpose:** Encapsulate head slice extraction logic

```cpp
/**
 * @brief Structure holding head slices for Q, K, V
 */
struct HeadSlices {
    Eigen::Ref<Eigen::MatrixXf> Q_h;
    Eigen::Ref<Eigen::MatrixXf> K_h;
    Eigen::Ref<Eigen::MatrixXf> V_h;
    
    HeadSlices(
        Eigen::MatrixXf& Q_proj,
        Eigen::MatrixXf& K_proj,
        Eigen::MatrixXf& V_proj,
        int head_idx,
        int head_dim
    ) : Q_h(Q_proj.middleCols(head_idx * head_dim, head_dim)),
        K_h(K_proj.middleCols(head_idx * head_dim, head_dim)),
        V_h(V_proj.middleCols(head_idx * head_dim, head_dim)) {}
};

/**
 * @brief Extract head slices from projected matrices
 * 
 * @param Q_proj Projected query matrix
 * @param K_proj Projected key matrix
 * @param V_proj Projected value matrix
 * @param head_idx Head index
 * @param head_dim Head dimension
 * @return HeadSlices structure with references to head slices
 */
inline HeadSlices extract_head_slices(
    Eigen::MatrixXf& Q_proj,
    Eigen::MatrixXf& K_proj,
    Eigen::MatrixXf& V_proj,
    int head_idx,
    int head_dim
) {
    return HeadSlices(Q_proj, K_proj, V_proj, head_idx, head_dim);
}
```

---

### Helper 3: Scale Calculation

**File:** `include/attention/math_utils.hpp` (new file)

**Purpose:** Centralize attention scale calculation

```cpp
/**
 * @brief Compute attention scale factor
 * 
 * @param head_dim Head dimension
 * @return Scale factor: 1.0 / sqrt(head_dim)
 */
inline float compute_attention_scale(int head_dim) {
    return 1.0f / std::sqrt(static_cast<float>(head_dim));
}

/**
 * @brief Compute attention scale factor (template version)
 * 
 * @tparam T Numeric type
 * @param head_dim Head dimension
 * @return Scale factor: 1.0 / sqrt(head_dim)
 */
template<typename T>
inline T compute_attention_scale(T head_dim) {
    return T(1) / std::sqrt(head_dim);
}
```

---

### Helper 4: Causal Mask Application

**File:** `include/attention/mask_utils.hpp` (new file)

**Purpose:** Apply causal mask to attention scores

```cpp
/**
 * @brief Apply causal mask to attention scores block
 * 
 * @param scores Attention scores matrix [seq_q, block_len]
 * @param block_start Starting index of the block in sequence
 * @param seq_q Query sequence length
 * @param block_len Block length
 */
inline void apply_causal_mask_block(
    Eigen::Ref<Eigen::MatrixXf> scores,
    int block_start,
    int seq_q,
    int block_len
) {
    const float mask_value = -std::numeric_limits<float>::infinity();
    
    for (int i = 0; i < seq_q; ++i) {
        for (int k = 0; k < block_len; ++k) {
            if (block_start + k > i) {
                scores(i, k) = mask_value;
            }
        }
    }
}
```

---

## 4. Integration Examples

### Example 1: Refactored `batch_softmax()` Using Parallel Helper

**Before (17 lines):**
```cpp
void batch_softmax(Eigen::Ref<Eigen::MatrixXf> scores) {
    // Apply softmax to each row
    #ifdef HAVE_TBB
    tbb::parallel_for(tbb::blocked_range<int>(0, scores.rows()),
        [&](const tbb::blocked_range<int>& r) {
            for (int i = r.begin(); i != r.end(); ++i) {
                Eigen::VectorXf row = scores.row(i);
                eigen_softmax(row);
                scores.row(i) = row;
            }
        });
    #else
    #pragma omp parallel for
    for (int i = 0; i < scores.rows(); ++i) {
        Eigen::VectorXf row = scores.row(i);
        eigen_softmax(row);
        scores.row(i) = row;
    }
    #endif
}
```

**After (8 lines):**
```cpp
#include "attention/parallel_utils.hpp"

void batch_softmax(Eigen::Ref<Eigen::MatrixXf> scores) {
    // Apply softmax to each row
    parallel_for(0, scores.rows(), [&](int i) {
        Eigen::VectorXf row = scores.row(i);
        eigen_softmax(row);
        scores.row(i) = row;
    });
}
```

**Improvements:**
- ✅ 17 lines → 8 lines (53% reduction)
- ✅ Single source of truth for parallelization
- ✅ Easier to add new parallelization backends
- ✅ Cleaner, more readable code

---

### Example 2: Refactored `multi_head_attention()` Using Helpers

**Before (35 lines for parallel section):**
```cpp
// Process each head in parallel
#ifdef HAVE_TBB
tbb::parallel_for(0, n_heads, [&](int h) {
    int start = h * head_dim;
    int end = start + head_dim;
    
    // Extract head slices
    Eigen::MatrixXf Q_h = Q_proj.middleCols(start, head_dim);
    Eigen::MatrixXf K_h = K_proj.middleCols(start, head_dim);
    Eigen::MatrixXf V_h = V_proj.middleCols(start, head_dim);
    
    // Compute attention for this head
    Eigen::MatrixXf head_output = scaled_dot_product_attention(
        Q_h, K_h, V_h, scale, mask);
    
    // Write to output (thread-safe: different columns)
    output.middleCols(start, head_dim) = head_output;
});
#else
#pragma omp parallel for
for (int h = 0; h < n_heads; ++h) {
    int start = h * head_dim;
    
    Eigen::MatrixXf Q_h = Q_proj.middleCols(start, head_dim);
    Eigen::MatrixXf K_h = K_proj.middleCols(start, head_dim);
    Eigen::MatrixXf V_h = V_proj.middleCols(start, head_dim);
    
    Eigen::MatrixXf head_output = scaled_dot_product_attention(
        Q_h, K_h, V_h, scale, mask);
    
    output.middleCols(start, head_dim) = head_output;
}
#endif
```

**After (15 lines):**
```cpp
#include "attention/parallel_utils.hpp"
#include "attention/head_utils.hpp"
#include "attention/math_utils.hpp"

// ... in multi_head_attention() ...

int head_dim = d_model / n_heads;
float scale = compute_attention_scale(head_dim);

// ... linear projections ...

Eigen::MatrixXf output = Eigen::MatrixXf::Zero(seq_len, d_model);

// Process each head in parallel
parallel_for(0, n_heads, [&](int h) {
    // Extract head slices
    auto slices = extract_head_slices(Q_proj, K_proj, V_proj, h, head_dim);
    
    // Compute attention for this head
    Eigen::MatrixXf head_output = scaled_dot_product_attention(
        slices.Q_h, slices.K_h, slices.V_h, scale, mask);
    
    // Write to output (thread-safe: different columns)
    output.middleCols(h * head_dim, head_dim) = head_output;
});
```

**Improvements:**
- ✅ 35 lines → 15 lines (57% reduction)
- ✅ No code duplication between TBB/OpenMP
- ✅ Clearer intent with helper function names
- ✅ Easier to maintain and test

---

### Example 3: Refactored `flash_attention_block()` Using Mask Helper

**Before (9 lines for causal mask):**
```cpp
// Apply causal mask if needed
if (causal) {
    for (int i = 0; i < seq_q; ++i) {
        for (int k = 0; k < block_len; ++k) {
            if (j + k > i) {
                S_j(i, k) = -std::numeric_limits<float>::infinity();
            }
        }
    }
}
```

**After (2 lines):**
```cpp
#include "attention/mask_utils.hpp"

// Apply causal mask if needed
if (causal) {
    apply_causal_mask_block(S_j, j, seq_q, block_len);
}
```

**Improvements:**
- ✅ 9 lines → 2 lines (78% reduction)
- ✅ Reusable mask application logic
- ✅ Consistent mask value handling
- ✅ Easier to test mask logic independently

---

### Example 4: Refactored Scale Calculation

**Before (1 line, but repeated):**
```cpp
float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
```

**After (1 line, but centralized):**
```cpp
#include "attention/math_utils.hpp"

float scale = compute_attention_scale(head_dim);
```

**Improvements:**
- ✅ Consistent calculation across all functions
- ✅ Easier to change formula if needed
- ✅ Self-documenting function name
- ✅ Type-safe template version available

---

## 5. Benefits Summary

### Code Reduction

| Pattern | Before | After | Reduction |
|---------|--------|-------|-----------|
| Parallelization (3 functions) | ~51 lines | ~24 lines | **53%** |
| Head extraction (2 functions) | ~20 lines | ~10 lines | **50%** |
| Causal mask | 9 lines | 2 lines | **78%** |
| Scale calculation | 1 line × N | 1 line × N | **Centralized** |
| **Total** | **~80 lines** | **~36 lines** | **~55%** |

### Maintainability

- ✅ **Single source of truth** for parallelization logic
- ✅ **Consistent patterns** across all attention functions
- ✅ **Easy to update** - change logic in one place
- ✅ **Clear, self-documenting code**

### Reusability

- ✅ **Parallel utilities** can be used in other kernels
- ✅ **Head extraction** can be used in other attention variants
- ✅ **Mask utilities** can be used for different mask types
- ✅ **Math utilities** can be used throughout the codebase

### Error Prevention

- ✅ **Consistent parallelization** prevents missing parallelization
- ✅ **Type-safe helpers** prevent type errors
- ✅ **Centralized calculations** prevent formula errors
- ✅ **Testable components** enable unit testing

---

## 6. Implementation Priority

### High Priority (Immediate Impact)

1. ✅ **Parallel Execution Helper** - Eliminates ~33 lines of duplicated code
2. ✅ **Head Slice Extraction** - Eliminates ~20 lines of repetitive code

### Medium Priority (Future Enhancement)

3. 🔄 **Scale Calculation Helper** - Improves consistency
4. 🔄 **Causal Mask Helper** - Improves reusability

---

## 7. Estimated Impact

### Code Reduction
- **~44 lines** of repetitive code eliminated
- **~55% reduction** in parallelization/head extraction code
- **4 helper modules** created

### Quality Improvements
- ✅ Consistent parallelization patterns
- ✅ Better code organization
- ✅ Clearer code intent
- ✅ Easier to test

### Future Benefits
- ✅ Easy to add new parallelization backends
- ✅ Easy to support new attention variants
- ✅ Easy to update parallelization logic
- ✅ Reusable across other kernels

---

## 8. Conclusion

The identified patterns represent **significant opportunities** for code optimization:

1. **Parallelization pattern** appears in 3 functions with ~51 lines of duplicated code
2. **Head extraction pattern** appears in 2 functions with ~20 lines of repetitive code
3. **Causal mask pattern** appears in 1 function but could be reused
4. **Scale calculation** appears in multiple places and should be centralized

**Creating these helper functions will:**
- Eliminate ~44 lines of repetitive code
- Improve code consistency
- Make future updates easier
- Reduce potential for errors

**Recommended Action:** Implement helper functions and refactor attention kernels to use them.








