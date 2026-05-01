/**
 * @file bindings_sampling.cpp
 * @brief PyBind11 bindings for sampling and inference modules
 * 
 * Provides Python bindings for:
 * - Token sampling (Top-K, Top-P, Mirostat, Beam Search)
 * - Quantization utilities (INT8, INT4, FP16, BF16)
 * - SIMD operations (vectorized ops)
 * 
 * @author TruthGPT Team
 * @version 1.0.0
 */

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <pybind11/functional.h>

#include "../include/inference/sampling.hpp"
#include "../include/quantization/quantize.hpp"
#include "../include/simd/simd_ops.hpp"

namespace py = pybind11;
using namespace truthgpt;

// ═══════════════════════════════════════════════════════════════════════════════
// SAMPLING BINDINGS
// ═══════════════════════════════════════════════════════════════════════════════

void bind_sampling(py::module_& m) {
    auto sampling = m.def_submodule("sampling", "Token sampling utilities");
    
    // SamplingConfig
    py::class_<inference::SamplingConfig>(sampling, "SamplingConfig")
        .def(py::init<>())
        .def_readwrite("temperature", &inference::SamplingConfig::temperature)
        .def_readwrite("top_k", &inference::SamplingConfig::top_k)
        .def_readwrite("top_p", &inference::SamplingConfig::top_p)
        .def_readwrite("min_p", &inference::SamplingConfig::min_p)
        .def_readwrite("typical_p", &inference::SamplingConfig::typical_p)
        .def_readwrite("repetition_penalty", &inference::SamplingConfig::repetition_penalty)
        .def_readwrite("do_sample", &inference::SamplingConfig::do_sample)
        .def_readwrite("use_mirostat", &inference::SamplingConfig::use_mirostat)
        .def_readwrite("mirostat_tau", &inference::SamplingConfig::mirostat_tau)
        .def_readwrite("mirostat_eta", &inference::SamplingConfig::mirostat_eta)
        .def_readwrite("seed", &inference::SamplingConfig::seed)
        .def("__repr__", [](const inference::SamplingConfig& c) {
            return "<SamplingConfig temp=" + std::to_string(c.temperature) +
                   " top_k=" + std::to_string(c.top_k) +
                   " top_p=" + std::to_string(c.top_p) + ">";
        });
    
    // TokenSampler
    py::class_<inference::TokenSampler>(sampling, "TokenSampler")
        .def(py::init<const inference::SamplingConfig&>(),
             py::arg("config") = inference::SamplingConfig())
        .def("sample", [](inference::TokenSampler& self,
                         py::array_t<float> logits,
                         const std::vector<int>& past_tokens) {
            auto buf = logits.request();
            std::vector<float> logits_vec(
                static_cast<float*>(buf.ptr),
                static_cast<float*>(buf.ptr) + buf.size
            );
            return self.sample(logits_vec, past_tokens);
        }, py::arg("logits"), py::arg("past_tokens") = std::vector<int>(),
           "Sample next token from logits")
        .def("set_seed", &inference::TokenSampler::set_seed)
        .def("reset_mirostat", &inference::TokenSampler::reset_mirostat)
        .def("set_config", &inference::TokenSampler::set_config);
    
    // BeamHypothesis
    py::class_<inference::BeamHypothesis>(sampling, "BeamHypothesis")
        .def_readonly("tokens", &inference::BeamHypothesis::tokens)
        .def_readonly("score", &inference::BeamHypothesis::score)
        .def_readonly("is_finished", &inference::BeamHypothesis::is_finished);
    
    // BeamSearchDecoder
    py::class_<inference::BeamSearchDecoder>(sampling, "BeamSearchDecoder")
        .def(py::init<int, float>(),
             py::arg("num_beams") = 4,
             py::arg("length_penalty") = 1.0f)
        .def("init", &inference::BeamSearchDecoder::init)
        .def("step", [](inference::BeamSearchDecoder& self,
                       py::list all_logprobs_list,
                       int eos_token_id) {
            std::vector<std::vector<float>> all_logprobs;
            for (auto& item : all_logprobs_list) {
                auto arr = item.cast<py::array_t<float>>();
                auto buf = arr.request();
                std::vector<float> vec(
                    static_cast<float*>(buf.ptr),
                    static_cast<float*>(buf.ptr) + buf.size
                );
                all_logprobs.push_back(std::move(vec));
            }
            self.step(all_logprobs, eos_token_id);
        }, py::arg("all_logprobs"), py::arg("eos_token_id"))
        .def("best", &inference::BeamSearchDecoder::best,
             py::return_value_policy::reference_internal)
        .def("beams", &inference::BeamSearchDecoder::beams,
             py::return_value_policy::reference_internal)
        .def("all_finished", &inference::BeamSearchDecoder::all_finished);
}

// ═══════════════════════════════════════════════════════════════════════════════
// QUANTIZATION BINDINGS
// ═══════════════════════════════════════════════════════════════════════════════

void bind_quantization(py::module_& m) {
    auto quant = m.def_submodule("quantization", "Quantization utilities");
    
    // QuantType enum
    py::enum_<quantization::QuantType>(quant, "QuantType")
        .value("FP32", quantization::QuantType::FP32)
        .value("FP16", quantization::QuantType::FP16)
        .value("BF16", quantization::QuantType::BF16)
        .value("INT8", quantization::QuantType::INT8)
        .value("INT4", quantization::QuantType::INT4);
    
    // QuantParams
    py::class_<quantization::QuantParams>(quant, "QuantParams")
        .def_readwrite("scale", &quantization::QuantParams::scale)
        .def_readwrite("zero_point", &quantization::QuantParams::zero_point)
        .def_static("int8_symmetric", &quantization::QuantParams::int8_symmetric)
        .def_static("int8_asymmetric", &quantization::QuantParams::int8_asymmetric)
        .def_static("int4_symmetric", &quantization::QuantParams::int4_symmetric);
    
    // Quantization functions
    quant.def("quantize_int8", [](py::array_t<float> input, float scale) {
        auto buf = input.request();
        size_t n = buf.size;
        
        std::vector<int8_t> output(n);
        quantization::QuantParams params = quantization::QuantParams::int8_symmetric(scale * 127.0f);
        
        for (size_t i = 0; i < n; ++i) {
            output[i] = quantization::quantize_int8(
                static_cast<float*>(buf.ptr)[i], params);
        }
        
        return py::array_t<int8_t>(n, output.data());
    }, py::arg("input"), py::arg("scale"),
       "Quantize FP32 array to INT8");
    
    quant.def("dequantize_int8", [](py::array_t<int8_t> input, float scale) {
        auto buf = input.request();
        size_t n = buf.size;
        
        std::vector<float> output(n);
        quantization::QuantParams params = quantization::QuantParams::int8_symmetric(scale * 127.0f);
        
        for (size_t i = 0; i < n; ++i) {
            output[i] = quantization::dequantize_int8(
                static_cast<int8_t*>(buf.ptr)[i], params);
        }
        
        return py::array_t<float>(n, output.data());
    }, py::arg("input"), py::arg("scale"),
       "Dequantize INT8 array to FP32");
    
    // FP16 conversion
    quant.def("to_fp16", [](py::array_t<float> input) {
        auto buf = input.request();
        size_t n = buf.size;
        float* ptr = static_cast<float*>(buf.ptr);
        
        std::vector<uint16_t> output(n);
        for (size_t i = 0; i < n; ++i) {
            output[i] = quantization::float_to_fp16(ptr[i]);
        }
        
        return py::array_t<uint16_t>(n, output.data());
    }, py::arg("input"), "Convert FP32 to FP16");
    
    quant.def("from_fp16", [](py::array_t<uint16_t> input) {
        auto buf = input.request();
        size_t n = buf.size;
        uint16_t* ptr = static_cast<uint16_t*>(buf.ptr);
        
        std::vector<float> output(n);
        for (size_t i = 0; i < n; ++i) {
            output[i] = quantization::fp16_to_float(ptr[i]);
        }
        
        return py::array_t<float>(n, output.data());
    }, py::arg("input"), "Convert FP16 to FP32");
    
    // BF16 conversion
    quant.def("to_bf16", [](py::array_t<float> input) {
        auto buf = input.request();
        size_t n = buf.size;
        float* ptr = static_cast<float*>(buf.ptr);
        
        std::vector<uint16_t> output(n);
        for (size_t i = 0; i < n; ++i) {
            output[i] = quantization::float_to_bf16(ptr[i]);
        }
        
        return py::array_t<uint16_t>(n, output.data());
    }, py::arg("input"), "Convert FP32 to BF16");
    
    quant.def("from_bf16", [](py::array_t<uint16_t> input) {
        auto buf = input.request();
        size_t n = buf.size;
        uint16_t* ptr = static_cast<uint16_t*>(buf.ptr);
        
        std::vector<float> output(n);
        for (size_t i = 0; i < n; ++i) {
            output[i] = quantization::bf16_to_float(ptr[i]);
        }
        
        return py::array_t<float>(n, output.data());
    }, py::arg("input"), "Convert BF16 to FP32");
}

// ═══════════════════════════════════════════════════════════════════════════════
// SIMD OPERATIONS BINDINGS
// ═══════════════════════════════════════════════════════════════════════════════

void bind_simd(py::module_& m) {
    auto simd_mod = m.def_submodule("simd", "SIMD-optimized operations");
    
    // Element-wise operations
    simd_mod.def("vec_add", [](py::array_t<float> a, py::array_t<float> b) {
        auto buf_a = a.request();
        auto buf_b = b.request();
        
        if (buf_a.size != buf_b.size) {
            throw std::runtime_error("Array sizes must match");
        }
        
        size_t n = buf_a.size;
        py::array_t<float> result(n);
        auto buf_c = result.request();
        
        simd::vec_add(
            static_cast<float*>(buf_a.ptr),
            static_cast<float*>(buf_b.ptr),
            static_cast<float*>(buf_c.ptr),
            n
        );
        
        return result;
    }, py::arg("a"), py::arg("b"), "SIMD-accelerated element-wise addition");
    
    simd_mod.def("vec_mul", [](py::array_t<float> a, py::array_t<float> b) {
        auto buf_a = a.request();
        auto buf_b = b.request();
        
        if (buf_a.size != buf_b.size) {
            throw std::runtime_error("Array sizes must match");
        }
        
        size_t n = buf_a.size;
        py::array_t<float> result(n);
        auto buf_c = result.request();
        
        simd::vec_mul(
            static_cast<float*>(buf_a.ptr),
            static_cast<float*>(buf_b.ptr),
            static_cast<float*>(buf_c.ptr),
            n
        );
        
        return result;
    }, py::arg("a"), py::arg("b"), "SIMD-accelerated element-wise multiplication");
    
    simd_mod.def("vec_dot", [](py::array_t<float> a, py::array_t<float> b) {
        auto buf_a = a.request();
        auto buf_b = b.request();
        
        if (buf_a.size != buf_b.size) {
            throw std::runtime_error("Array sizes must match");
        }
        
        return simd::vec_dot(
            static_cast<float*>(buf_a.ptr),
            static_cast<float*>(buf_b.ptr),
            buf_a.size
        );
    }, py::arg("a"), py::arg("b"), "SIMD-accelerated dot product");
    
    simd_mod.def("vec_sum", [](py::array_t<float> a) {
        auto buf = a.request();
        return simd::vec_sum(static_cast<float*>(buf.ptr), buf.size);
    }, py::arg("a"), "SIMD-accelerated sum reduction");
    
    simd_mod.def("vec_max", [](py::array_t<float> a) {
        auto buf = a.request();
        return simd::vec_max(static_cast<float*>(buf.ptr), buf.size);
    }, py::arg("a"), "SIMD-accelerated max reduction");
    
    // Activation functions
    simd_mod.def("relu", [](py::array_t<float> input) {
        auto buf = input.request();
        size_t n = buf.size;
        py::array_t<float> result(n);
        auto buf_out = result.request();
        
        simd::vec_relu(
            static_cast<float*>(buf.ptr),
            static_cast<float*>(buf_out.ptr),
            n
        );
        
        return result;
    }, py::arg("input"), "SIMD-accelerated ReLU activation");
    
    simd_mod.def("gelu", [](py::array_t<float> input) {
        auto buf = input.request();
        size_t n = buf.size;
        py::array_t<float> result(n);
        auto buf_out = result.request();
        
        simd::vec_gelu(
            static_cast<float*>(buf.ptr),
            static_cast<float*>(buf_out.ptr),
            n
        );
        
        return result;
    }, py::arg("input"), "SIMD-accelerated GELU activation");
    
    simd_mod.def("silu", [](py::array_t<float> input) {
        auto buf = input.request();
        size_t n = buf.size;
        py::array_t<float> result(n);
        auto buf_out = result.request();
        
        simd::vec_silu(
            static_cast<float*>(buf.ptr),
            static_cast<float*>(buf_out.ptr),
            n
        );
        
        return result;
    }, py::arg("input"), "SIMD-accelerated SiLU/Swish activation");
    
    simd_mod.def("softmax", [](py::array_t<float> input) {
        auto buf = input.request();
        size_t n = buf.size;
        py::array_t<float> result(n);
        auto buf_out = result.request();
        
        // Copy input to output
        std::memcpy(buf_out.ptr, buf.ptr, n * sizeof(float));
        
        simd::vec_softmax(
            static_cast<float*>(buf.ptr),
            static_cast<float*>(buf_out.ptr),
            n
        );
        
        return result;
    }, py::arg("input"), "SIMD-accelerated softmax");
    
    // Layer normalization
    simd_mod.def("layer_norm", [](py::array_t<float> input,
                                  py::array_t<float> gamma,
                                  py::array_t<float> beta,
                                  float eps) {
        auto buf_in = input.request();
        auto buf_gamma = gamma.request();
        auto buf_beta = beta.request();
        size_t n = buf_in.size;
        
        if (buf_gamma.size != n || buf_beta.size != n) {
            throw std::runtime_error("Gamma and beta must match input size");
        }
        
        py::array_t<float> result(n);
        auto buf_out = result.request();
        
        simd::layer_norm(
            static_cast<float*>(buf_in.ptr),
            static_cast<float*>(buf_gamma.ptr),
            static_cast<float*>(buf_beta.ptr),
            static_cast<float*>(buf_out.ptr),
            n,
            eps
        );
        
        return result;
    }, py::arg("input"), py::arg("gamma"), py::arg("beta"),
       py::arg("eps") = 1e-5f, "SIMD-accelerated layer normalization");
    
    simd_mod.def("rms_norm", [](py::array_t<float> input,
                                py::array_t<float> weight,
                                float eps) {
        auto buf_in = input.request();
        auto buf_weight = weight.request();
        size_t n = buf_in.size;
        
        py::array_t<float> result(n);
        auto buf_out = result.request();
        
        simd::rms_norm(
            static_cast<float*>(buf_in.ptr),
            static_cast<float*>(buf_weight.ptr),
            static_cast<float*>(buf_out.ptr),
            n,
            eps
        );
        
        return result;
    }, py::arg("input"), py::arg("weight"),
       py::arg("eps") = 1e-5f, "SIMD-accelerated RMS normalization");
    
    // Capability info
    simd_mod.def("simd_capability", &simd::simd_capability,
                "Get SIMD capability string (AVX-512, AVX2, SSE4.1, or Scalar)");
}

// ═══════════════════════════════════════════════════════════════════════════════
// MODULE DEFINITION
// ═══════════════════════════════════════════════════════════════════════════════

PYBIND11_MODULE(truthgpt_cpp_ext, m) {
    m.doc() = "TruthGPT C++ Extensions - Sampling, Quantization, and SIMD operations";
    
    // Bind submodules
    bind_sampling(m);
    bind_quantization(m);
    bind_simd(m);
    
    // Version info
    m.attr("__version__") = "1.0.0";
    
    // Get system info
    m.def("get_info", []() {
        py::dict info;
        info["simd"] = simd::simd_capability();
        
#ifdef __AVX512F__
        info["avx512"] = true;
#else
        info["avx512"] = false;
#endif
        
#ifdef __AVX2__
        info["avx2"] = true;
#else
        info["avx2"] = false;
#endif
        
        return info;
    }, "Get system capabilities info");
}




