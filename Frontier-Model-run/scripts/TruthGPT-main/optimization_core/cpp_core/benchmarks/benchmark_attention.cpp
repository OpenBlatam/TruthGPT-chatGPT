/**
 * @file benchmark_attention.cpp
 * @brief Benchmarks for attention implementations
 * 
 * Compares performance of different attention backends:
 * - Standard attention (Eigen)
 * - Flash attention (block-wise)
 * - Grouped Query Attention (GQA)
 */

#include <benchmark/benchmark.h>
#include <random>
#include <vector>

#ifdef HAVE_EIGEN
#include <Eigen/Dense>
#endif

namespace optimization_core {
namespace benchmarks {

#ifdef HAVE_EIGEN

using Matrix = Eigen::MatrixXf;
using Vector = Eigen::VectorXf;

/**
 * @brief Generate random matrix
 */
Matrix random_matrix(int rows, int cols, unsigned int seed = 42) {
    std::mt19937 gen(seed);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    
    Matrix m(rows, cols);
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            m(i, j) = dist(gen);
        }
    }
    return m;
}

/**
 * @brief Standard attention implementation
 */
Matrix standard_attention(const Matrix& Q, const Matrix& K, const Matrix& V, float scale) {
    Matrix scores = (Q * K.transpose()) * scale;
    
    // Softmax per row
    for (int i = 0; i < scores.rows(); ++i) {
        float max_val = scores.row(i).maxCoeff();
        scores.row(i) = (scores.row(i).array() - max_val).exp();
        scores.row(i) /= scores.row(i).sum();
    }
    
    return scores * V;
}

/**
 * @brief Flash attention with blocking
 */
Matrix flash_attention(const Matrix& Q, const Matrix& K, const Matrix& V, 
                       float scale, int block_size = 64) {
    int seq_len = Q.rows();
    int head_dim = Q.cols();
    
    Matrix output = Matrix::Zero(seq_len, head_dim);
    Vector m = Vector::Constant(seq_len, -std::numeric_limits<float>::infinity());
    Vector l = Vector::Zero(seq_len);
    
    for (int j = 0; j < K.rows(); j += block_size) {
        int block_len = std::min(block_size, static_cast<int>(K.rows()) - j);
        
        Matrix K_j = K.middleRows(j, block_len);
        Matrix V_j = V.middleRows(j, block_len);
        
        Matrix S = (Q * K_j.transpose()) * scale;
        
        Vector m_j = S.rowwise().maxCoeff();
        Vector m_new = m.cwiseMax(m_j);
        
        Vector alpha = (m - m_new).array().exp();
        Vector beta = (m_j - m_new).array().exp();
        
        Matrix P = (S.colwise() - m_new).array().exp().matrix();
        
        Vector l_j = P.rowwise().sum();
        Vector l_new = alpha.cwiseProduct(l) + beta.cwiseProduct(l_j);
        
        output = (alpha.asDiagonal() * output) + (P * V_j);
        
        m = m_new;
        l = l_new;
    }
    
    // Normalize
    for (int i = 0; i < seq_len; ++i) {
        if (l(i) > 0) {
            output.row(i) /= l(i);
        }
    }
    
    return output;
}

// ============================================================================
// Benchmark Functions
// ============================================================================

static void BM_StandardAttention(benchmark::State& state) {
    int seq_len = state.range(0);
    int head_dim = state.range(1);
    float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
    
    Matrix Q = random_matrix(seq_len, head_dim);
    Matrix K = random_matrix(seq_len, head_dim);
    Matrix V = random_matrix(seq_len, head_dim);
    
    for (auto _ : state) {
        auto output = standard_attention(Q, K, V, scale);
        benchmark::DoNotOptimize(output);
    }
    
    state.SetItemsProcessed(state.iterations() * seq_len);
    state.SetBytesProcessed(state.iterations() * seq_len * head_dim * sizeof(float) * 4);
}

static void BM_FlashAttention(benchmark::State& state) {
    int seq_len = state.range(0);
    int head_dim = state.range(1);
    int block_size = state.range(2);
    float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
    
    Matrix Q = random_matrix(seq_len, head_dim);
    Matrix K = random_matrix(seq_len, head_dim);
    Matrix V = random_matrix(seq_len, head_dim);
    
    for (auto _ : state) {
        auto output = flash_attention(Q, K, V, scale, block_size);
        benchmark::DoNotOptimize(output);
    }
    
    state.SetItemsProcessed(state.iterations() * seq_len);
    state.SetBytesProcessed(state.iterations() * seq_len * head_dim * sizeof(float) * 4);
}

static void BM_MatMul(benchmark::State& state) {
    int m = state.range(0);
    int k = state.range(1);
    int n = state.range(2);
    
    Matrix A = random_matrix(m, k);
    Matrix B = random_matrix(k, n);
    
    for (auto _ : state) {
        Matrix C = A * B;
        benchmark::DoNotOptimize(C);
    }
    
    state.SetItemsProcessed(state.iterations());
    state.SetBytesProcessed(state.iterations() * (m * k + k * n + m * n) * sizeof(float));
}

static void BM_Softmax(benchmark::State& state) {
    int rows = state.range(0);
    int cols = state.range(1);
    
    Matrix m = random_matrix(rows, cols);
    
    for (auto _ : state) {
        Matrix result = m;
        for (int i = 0; i < result.rows(); ++i) {
            float max_val = result.row(i).maxCoeff();
            result.row(i) = (result.row(i).array() - max_val).exp();
            result.row(i) /= result.row(i).sum();
        }
        benchmark::DoNotOptimize(result);
    }
    
    state.SetItemsProcessed(state.iterations() * rows);
}

// Register benchmarks
BENCHMARK(BM_StandardAttention)
    ->Args({128, 64})    // seq=128, head_dim=64
    ->Args({256, 64})
    ->Args({512, 64})
    ->Args({1024, 64})
    ->Args({2048, 64})
    ->Unit(benchmark::kMicrosecond);

BENCHMARK(BM_FlashAttention)
    ->Args({128, 64, 32})    // seq, head_dim, block_size
    ->Args({256, 64, 64})
    ->Args({512, 64, 64})
    ->Args({1024, 64, 128})
    ->Args({2048, 64, 128})
    ->Unit(benchmark::kMicrosecond);

BENCHMARK(BM_MatMul)
    ->Args({512, 768, 768})    // Typical transformer sizes
    ->Args({512, 768, 3072})   // FFN hidden
    ->Args({512, 3072, 768})
    ->Unit(benchmark::kMicrosecond);

BENCHMARK(BM_Softmax)
    ->Args({512, 512})
    ->Args({1024, 1024})
    ->Args({2048, 2048})
    ->Unit(benchmark::kMicrosecond);

#endif // HAVE_EIGEN

} // namespace benchmarks
} // namespace optimization_core

BENCHMARK_MAIN();












