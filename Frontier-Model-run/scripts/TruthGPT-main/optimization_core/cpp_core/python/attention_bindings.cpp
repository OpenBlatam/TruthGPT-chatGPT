/**
 * @file attention_bindings.cpp
 * @brief PyBind11 bindings for attention module
 * 
 * Modular bindings for Flash Attention implementations.
 */

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <pybind11/functional.h>

#ifdef HAVE_EIGEN
#include <pybind11/eigen.h>
#endif

#include "../include/attention/flash_attention.hpp"
#include <sstream>

namespace py = pybind11;
using namespace optimization_core;

template<typename T>
std::vector<T> numpy_to_vector(py::array_t<T> arr) {
    auto buf = arr.request();
    if (buf.ndim == 0) return {};
    T* ptr = static_cast<T*>(buf.ptr);
    return std::vector<T>(ptr, ptr + buf.size);
}

template<typename T>
py::array_t<T> vector_to_numpy(const std::vector<T>& vec,
                               const std::vector<ssize_t>& shape) {
    ssize_t total_size = 1;
    for (auto s : shape) total_size *= s;
    
    if (total_size != static_cast<ssize_t>(vec.size())) {
        throw std::runtime_error("Shape mismatch");
    }
    
    py::array_t<T> result(shape);
    auto buf = result.request();
    std::memcpy(buf.ptr, vec.data(), vec.size() * sizeof(T));
    return result;
}

void register_attention_module(py::module_& m) {
    auto attention_module = m.def_submodule("attention",
        "High-performance attention implementations");
    
    py::enum_<attention::AttentionPattern>(attention_module, "AttentionPattern",
        "Attention pattern types")
        .value("Full", attention::AttentionPattern::Full)
        .value("Causal", attention::AttentionPattern::Causal)
        .value("SlidingWindow", attention::AttentionPattern::SlidingWindow)
        .value("BlockSparse", attention::AttentionPattern::BlockSparse)
        .value("Strided", attention::AttentionPattern::Strided)
        .value("Local", attention::AttentionPattern::Local)
        .value("BigBird", attention::AttentionPattern::BigBird)
        .export_values();
    
    py::enum_<attention::PositionEncoding>(attention_module, "PositionEncoding",
        "Position encoding types")
        .value("None", attention::PositionEncoding::None)
        .value("RoPE", attention::PositionEncoding::RoPE)
        .value("ALiBi", attention::PositionEncoding::ALiBi)
        .value("Relative", attention::PositionEncoding::Relative)
        .export_values();
    
    py::class_<attention::FlashAttentionConfig>(attention_module, "FlashAttentionConfig",
        R"doc(
            Configuration for Flash Attention.
            
            Example:
                >>> config = FlashAttentionConfig(d_model=768, n_heads=12)
                >>> attn = FlashAttentionCPU(config)
        )doc")
        .def(py::init<>())
        .def(py::init([](int d_model, int n_heads, int n_kv_heads, int head_dim,
                        int max_seq_len, float dropout, bool use_causal_mask,
                        int window_size) {
            attention::FlashAttentionConfig config;
            config.d_model = d_model;
            config.n_heads = n_heads;
            config.n_kv_heads = n_kv_heads > 0 ? n_kv_heads : n_heads;
            config.head_dim = head_dim > 0 ? head_dim : d_model / n_heads;
            config.max_seq_len = max_seq_len;
            config.dropout = dropout;
            config.pattern = use_causal_mask ? attention::AttentionPattern::Causal 
                                            : attention::AttentionPattern::Full;
            config.window_size = window_size;
            return config;
        }),
        py::arg("d_model") = 768,
        py::arg("n_heads") = 12,
        py::arg("n_kv_heads") = -1,
        py::arg("head_dim") = -1,
        py::arg("max_seq_len") = 8192,
        py::arg("dropout") = 0.0f,
        py::arg("use_causal_mask") = false,
        py::arg("window_size") = 512)
        .def_readwrite("d_model", &attention::FlashAttentionConfig::d_model)
        .def_readwrite("n_heads", &attention::FlashAttentionConfig::n_heads)
        .def_readwrite("n_kv_heads", &attention::FlashAttentionConfig::n_kv_heads)
        .def_readwrite("head_dim", &attention::FlashAttentionConfig::head_dim)
        .def_readwrite("max_seq_len", &attention::FlashAttentionConfig::max_seq_len)
        .def_readwrite("dropout", &attention::FlashAttentionConfig::dropout)
        .def_readwrite("pattern", &attention::FlashAttentionConfig::pattern)
        .def_readwrite("position_encoding", &attention::FlashAttentionConfig::position_encoding)
        .def_readwrite("window_size", &attention::FlashAttentionConfig::window_size)
        .def_readwrite("rope_theta", &attention::FlashAttentionConfig::rope_theta)
        .def("validate", &attention::FlashAttentionConfig::validate)
        .def("get_softmax_scale", &attention::FlashAttentionConfig::get_softmax_scale)
        .def("is_gqa", &attention::FlashAttentionConfig::is_gqa)
        .def_static("llama_7b", &attention::FlashAttentionConfig::llama_7b)
        .def_static("llama_70b", &attention::FlashAttentionConfig::llama_70b)
        .def_static("mistral_7b", &attention::FlashAttentionConfig::mistral_7b)
        .def("__repr__", [](const attention::FlashAttentionConfig& c) {
            std::ostringstream ss;
            ss << "FlashAttentionConfig(d_model=" << c.d_model
               << ", n_heads=" << c.n_heads << ")";
            return ss.str();
        });
    
    py::class_<attention::AttentionMask>(attention_module, "AttentionMask",
        "Flexible attention mask")
        .def(py::init<>())
        .def_static("causal", &attention::AttentionMask::causal,
            py::arg("seq_len"))
        .def_static("sliding_window", &attention::AttentionMask::sliding_window,
            py::arg("seq_len"), py::arg("window_size"), py::arg("global_tokens") = 0)
        .def_static("custom", [](py::array_t<float> mask, int seq_q, int seq_k) {
            return attention::AttentionMask::custom(numpy_to_vector(mask), seq_q, seq_k);
        }, py::arg("mask"), py::arg("seq_q"), py::arg("seq_k"))
        .def("is_masked", &attention::AttentionMask::is_masked,
            py::arg("i"), py::arg("j"))
        .def("get_value", &attention::AttentionMask::get_value,
            py::arg("i"), py::arg("j"));

#ifdef HAVE_EIGEN
    py::class_<attention::FlashAttentionCPU>(attention_module, "FlashAttentionCPU",
        R"doc(
            CPU-optimized Flash Attention using Eigen.
            
            Provides 5-10x speedup over PyTorch CPU attention.
        )doc")
        .def(py::init<const attention::FlashAttentionConfig&>(), py::arg("config"))
        .def("forward", [](attention::FlashAttentionCPU& self,
                          py::array_t<float> query,
                          py::array_t<float> key,
                          py::array_t<float> value,
                          int batch_size,
                          int seq_len,
                          std::optional<attention::AttentionMask> mask,
                          bool return_attention_weights) {
            
            auto result = self.forward(
                numpy_to_vector(query),
                numpy_to_vector(key),
                numpy_to_vector(value),
                batch_size, seq_len, mask, return_attention_weights
            );
            
            py::dict ret;
            ret["output"] = vector_to_numpy(result.output,
                {result.batch_size, result.seq_len, result.d_model});
            ret["compute_time_ms"] = result.compute_time_ms;
            ret["memory_bytes"] = result.memory_used_bytes;
            
            if (result.attention_weights.has_value()) {
                ret["attention_weights"] = vector_to_numpy(result.attention_weights.value(),
                    {result.batch_size * self.config().n_heads, result.seq_len, result.seq_len});
            }
            
            return ret;
        },
        py::arg("query"),
        py::arg("key"),
        py::arg("value"),
        py::arg("batch_size"),
        py::arg("seq_len"),
        py::arg("mask") = py::none(),
        py::arg("return_attention_weights") = false)
        .def("set_weights", [](attention::FlashAttentionCPU& self,
                               py::array_t<float> wq,
                               py::array_t<float> wk,
                               py::array_t<float> wv,
                               py::array_t<float> wo) {
            self.set_weights(
                numpy_to_vector(wq),
                numpy_to_vector(wk),
                numpy_to_vector(wv),
                numpy_to_vector(wo)
            );
        },
        py::arg("wq"), py::arg("wk"), py::arg("wv"), py::arg("wo"))
        .def_property_readonly("config", &attention::FlashAttentionCPU::config);
#endif
}












