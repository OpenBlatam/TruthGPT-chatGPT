/**
 * @file bindings_refactored.cpp
 * @brief Refactored PyBind11 bindings using new modular architecture
 */

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <pybind11/functional.h>

#include "../include/optimization_core.hpp"

namespace py = pybind11;
using namespace optimization_core;

// ============================================================================
// Type Conversions
// ============================================================================

template<typename T>
std::vector<T> numpy_to_vector(py::array_t<T> arr) {
    auto buf = arr.request();
    T* ptr = static_cast<T*>(buf.ptr);
    return std::vector<T>(ptr, ptr + buf.size);
}

template<typename T>
py::array_t<T> vector_to_numpy(const std::vector<T>& vec,
                               const std::vector<py::ssize_t>& shape = {}) {
    if (shape.empty()) {
        return py::array_t<T>(vec.size(), vec.data());
    }
    return py::array_t<T>(shape, vec.data());
}

// ============================================================================
// Module Definition
// ============================================================================

PYBIND11_MODULE(_cpp_core, m) {
    m.doc() = R"doc(
        optimization_core C++ Extensions (Refactored)
        =============================================
        
        High-performance implementations with clean Python bindings.
        
        Modules:
        - attention: Flash attention, sparse attention
        - memory: KV cache with multiple eviction strategies
        - inference: Token sampling, beam search
    )doc";
    
    // Version info
    m.attr("__version__") = Version::string;
    m.def("version", &Version::full, "Get full version string");
    m.def("available_backends", &available_backends, "Get list of available backends");
    m.def("has_backend", &has_backend, py::arg("name"), "Check if backend is available");
    
    // ========================================================================
    // Common Types
    // ========================================================================
    
    py::class_<Shape>(m, "Shape", "Tensor shape")
        .def(py::init<>())
        .def(py::init([](py::list dims) {
            std::vector<usize> d;
            for (auto item : dims) d.push_back(item.cast<usize>());
            return Shape(d);
        }))
        .def("rank", &Shape::rank)
        .def("numel", &Shape::numel)
        .def("__len__", &Shape::rank)
        .def("__getitem__", [](const Shape& s, usize i) { return s[i]; })
        .def("__repr__", &Shape::to_string);
    
    py::class_<Timer>(m, "Timer", "High-resolution timer")
        .def(py::init<>())
        .def("reset", &Timer::reset)
        .def("elapsed_ms", &Timer::elapsed_ms)
        .def("elapsed_us", &Timer::elapsed_us)
        .def("elapsed_s", &Timer::elapsed_s);
    
    // ========================================================================
    // Attention Module
    // ========================================================================
    
    auto attn = m.def_submodule("attention", "Attention mechanisms");
    
    py::class_<attention::AttentionConfig>(attn, "AttentionConfig",
        R"doc(
            Attention configuration with builder pattern.
            
            Example:
                config = AttentionConfig()
                config.with_heads(12).with_head_dim(64).with_flash(128)
        )doc")
        .def(py::init<>())
        .def_readwrite("num_heads", &attention::AttentionConfig::num_heads)
        .def_readwrite("head_dim", &attention::AttentionConfig::head_dim)
        .def_readwrite("dropout", &attention::AttentionConfig::dropout)
        .def_readwrite("use_flash", &attention::AttentionConfig::use_flash)
        .def_readwrite("block_size", &attention::AttentionConfig::block_size)
        .def_readwrite("use_causal_mask", &attention::AttentionConfig::use_causal_mask)
        .def("with_heads", &attention::AttentionConfig::with_heads, py::return_value_policy::reference)
        .def("with_head_dim", &attention::AttentionConfig::with_head_dim, py::return_value_policy::reference)
        .def("with_dropout", &attention::AttentionConfig::with_dropout, py::return_value_policy::reference)
        .def("with_flash", &attention::AttentionConfig::with_flash, py::return_value_policy::reference)
        .def("with_causal", &attention::AttentionConfig::with_causal, py::return_value_policy::reference)
        .def("get_scale", &attention::AttentionConfig::get_scale)
        .def("d_model", &attention::AttentionConfig::d_model)
        .def("validate", &attention::AttentionConfig::validate);
    
    py::class_<attention::AttentionStats>(attn, "AttentionStats")
        .def_readonly("total_tokens", &attention::AttentionStats::total_tokens)
        .def_readonly("attention_computations", &attention::AttentionStats::attention_computations)
        .def_readonly("memory_peak_mb", &attention::AttentionStats::memory_peak_mb)
        .def_readonly("compute_time_ms", &attention::AttentionStats::compute_time_ms)
        .def_static("compute", &attention::AttentionStats::compute);
    
#ifdef HAVE_EIGEN
    py::class_<attention::ScaledDotProductAttention, attention::IAttention,
               std::shared_ptr<attention::ScaledDotProductAttention>>(
        attn, "ScaledDotProductAttention",
        "Standard scaled dot-product attention")
        .def(py::init<attention::AttentionConfig>(),
             py::arg("config") = attention::AttentionConfig())
        .def("forward", [](attention::ScaledDotProductAttention& self,
                          py::array_t<f32> q, py::array_t<f32> k, py::array_t<f32> v,
                          usize batch, usize seq, std::optional<py::array_t<f32>> mask) {
            auto qv = numpy_to_vector(q);
            auto kv = numpy_to_vector(k);
            auto vv = numpy_to_vector(v);
            std::optional<std::vector<f32>> mv;
            if (mask) mv = numpy_to_vector(*mask);
            
            auto out = self.forward(qv, kv, vv, batch, seq, mv);
            return vector_to_numpy(out, {static_cast<py::ssize_t>(batch * seq), 
                                         static_cast<py::ssize_t>(self.get_stats().total_tokens / (batch * seq))});
        }, py::arg("query"), py::arg("key"), py::arg("value"),
           py::arg("batch_size"), py::arg("seq_len"), py::arg("mask") = py::none())
        .def("get_stats", &attention::ScaledDotProductAttention::get_stats)
        .def("reset_stats", &attention::ScaledDotProductAttention::reset_stats);
    
    py::class_<attention::FlashAttention, attention::IAttention,
               std::shared_ptr<attention::FlashAttention>>(
        attn, "FlashAttention",
        "Memory-efficient flash attention with block processing")
        .def(py::init<attention::AttentionConfig>(),
             py::arg("config") = attention::AttentionConfig())
        .def("forward", [](attention::FlashAttention& self,
                          py::array_t<f32> q, py::array_t<f32> k, py::array_t<f32> v,
                          usize batch, usize seq, std::optional<py::array_t<f32>> mask) {
            auto qv = numpy_to_vector(q);
            auto kv = numpy_to_vector(k);
            auto vv = numpy_to_vector(v);
            std::optional<std::vector<f32>> mv;
            if (mask) mv = numpy_to_vector(*mask);
            
            auto out = self.forward(qv, kv, vv, batch, seq, mv);
            return vector_to_numpy(out);
        }, py::arg("query"), py::arg("key"), py::arg("value"),
           py::arg("batch_size"), py::arg("seq_len"), py::arg("mask") = py::none())
        .def("get_stats", &attention::FlashAttention::get_stats);
#endif
    
    // Mask utilities
    attn.def("create_causal_mask", &attention::mask::create_causal,
             py::arg("seq_len"), "Create causal attention mask");
    attn.def("create_padding_mask", &attention::mask::create_padding,
             py::arg("lengths"), py::arg("max_len"), "Create padding mask from lengths");
    
    // ========================================================================
    // Memory Module  
    // ========================================================================
    
    auto mem = m.def_submodule("memory", "Memory management");
    
    py::enum_<memory::EvictionStrategy>(mem, "EvictionStrategy")
        .value("LRU", memory::EvictionStrategy::LRU, "Least Recently Used")
        .value("LFU", memory::EvictionStrategy::LFU, "Least Frequently Used")
        .value("FIFO", memory::EvictionStrategy::FIFO, "First In First Out")
        .value("Adaptive", memory::EvictionStrategy::Adaptive, "Adaptive (LRU + LFU)")
        .export_values();
    
    py::class_<memory::CacheConfig>(mem, "CacheConfig",
        "KV Cache configuration with builder pattern")
        .def(py::init<>())
        .def_readwrite("max_size", &memory::CacheConfig::max_size)
        .def_readwrite("eviction_strategy", &memory::CacheConfig::eviction_strategy)
        .def_readwrite("enable_compression", &memory::CacheConfig::enable_compression)
        .def_readwrite("compression_threshold", &memory::CacheConfig::compression_threshold)
        .def("with_size", &memory::CacheConfig::with_size, py::return_value_policy::reference)
        .def("with_strategy", &memory::CacheConfig::with_strategy, py::return_value_policy::reference)
        .def("with_compression", &memory::CacheConfig::with_compression, 
             py::arg("enable"), py::arg("threshold") = 1024,
             py::return_value_policy::reference);
    
    py::class_<memory::CacheStats>(mem, "CacheStats")
        .def_property_readonly("hit_count", [](const memory::CacheStats& s) { 
            return s.hit_count.load(); 
        })
        .def_property_readonly("miss_count", [](const memory::CacheStats& s) { 
            return s.miss_count.load(); 
        })
        .def_property_readonly("eviction_count", [](const memory::CacheStats& s) { 
            return s.eviction_count.load(); 
        })
        .def_property_readonly("current_size", [](const memory::CacheStats& s) { 
            return s.current_size.load(); 
        })
        .def("hit_rate", &memory::CacheStats::hit_rate)
        .def("compression_ratio", &memory::CacheStats::compression_ratio);
    
    py::class_<memory::KVCache>(mem, "KVCache",
        "High-performance KV cache for LLM inference")
        .def(py::init<memory::CacheConfig>(), py::arg("config") = memory::CacheConfig())
        .def("get", [](memory::KVCache& self, usize layer, usize pos, const std::string& tag) {
            auto result = self.get(layer, pos, tag);
            if (result) {
                return py::cast(py::bytes(reinterpret_cast<const char*>(result->data()), 
                                          result->size()));
            }
            return py::cast(py::none());
        }, py::arg("layer_idx"), py::arg("position"), py::arg("tag") = "")
        .def("put", [](memory::KVCache& self, usize layer, usize pos, 
                      py::bytes data, const std::string& tag) {
            std::string str = data;
            std::vector<u8> vec(str.begin(), str.end());
            self.put(layer, pos, std::move(vec), tag);
        }, py::arg("layer_idx"), py::arg("position"), py::arg("data"), py::arg("tag") = "")
        .def("remove", &memory::KVCache::remove,
             py::arg("layer_idx"), py::arg("position"), py::arg("tag") = "")
        .def("clear", &memory::KVCache::clear)
        .def("size", &memory::KVCache::size)
        .def("max_size", &memory::KVCache::max_size)
        .def("empty", &memory::KVCache::empty)
        .def("full", &memory::KVCache::full)
        .def("hit_rate", &memory::KVCache::hit_rate)
        .def_property_readonly("stats", &memory::KVCache::stats)
        .def("__len__", &memory::KVCache::size);
    
    py::class_<memory::ConcurrentKVCache>(mem, "ConcurrentKVCache",
        "Thread-safe KV cache wrapper")
        .def(py::init<memory::CacheConfig>(), py::arg("config") = memory::CacheConfig())
        .def("get", [](memory::ConcurrentKVCache& self, usize layer, usize pos, 
                      const std::string& tag) {
            auto result = self.get(layer, pos, tag);
            if (result) {
                return py::cast(py::bytes(reinterpret_cast<const char*>(result->data()),
                                          result->size()));
            }
            return py::cast(py::none());
        }, py::arg("layer_idx"), py::arg("position"), py::arg("tag") = "")
        .def("put", [](memory::ConcurrentKVCache& self, usize layer, usize pos,
                      py::bytes data, const std::string& tag) {
            std::string str = data;
            std::vector<u8> vec(str.begin(), str.end());
            self.put(layer, pos, std::move(vec), tag);
        }, py::arg("layer_idx"), py::arg("position"), py::arg("data"), py::arg("tag") = "")
        .def("clear", &memory::ConcurrentKVCache::clear)
        .def("size", &memory::ConcurrentKVCache::size)
        .def("hit_rate", &memory::ConcurrentKVCache::hit_rate);
    
    // ========================================================================
    // Inference Module
    // ========================================================================
    
    auto inf = m.def_submodule("inference", "Inference engine");
    
    py::class_<inference::GenerationConfig>(inf, "GenerationConfig",
        "Generation configuration with builder pattern and presets")
        .def(py::init<>())
        .def_readwrite("max_new_tokens", &inference::GenerationConfig::max_new_tokens)
        .def_readwrite("temperature", &inference::GenerationConfig::temperature)
        .def_readwrite("top_p", &inference::GenerationConfig::top_p)
        .def_readwrite("top_k", &inference::GenerationConfig::top_k)
        .def_readwrite("repetition_penalty", &inference::GenerationConfig::repetition_penalty)
        .def_readwrite("do_sample", &inference::GenerationConfig::do_sample)
        .def_readwrite("num_beams", &inference::GenerationConfig::num_beams)
        .def_readwrite("eos_token_id", &inference::GenerationConfig::eos_token_id)
        .def("with_max_tokens", &inference::GenerationConfig::with_max_tokens, 
             py::return_value_policy::reference)
        .def("with_temperature", &inference::GenerationConfig::with_temperature,
             py::return_value_policy::reference)
        .def("with_top_p", &inference::GenerationConfig::with_top_p,
             py::return_value_policy::reference)
        .def("with_top_k", &inference::GenerationConfig::with_top_k,
             py::return_value_policy::reference)
        .def("with_sampling", &inference::GenerationConfig::with_sampling,
             py::return_value_policy::reference)
        .def_static("greedy", &inference::GenerationConfig::greedy, "Create greedy config")
        .def_static("sampling", &inference::GenerationConfig::sampling,
                   py::arg("temp") = 0.8f, py::arg("top_p") = 0.9f, 
                   "Create sampling config")
        .def_static("beam", &inference::GenerationConfig::beam,
                   py::arg("num_beams") = 4, "Create beam search config");
    
    py::class_<inference::GenerationResult>(inf, "GenerationResult")
        .def_readonly("token_ids", &inference::GenerationResult::token_ids)
        .def_readonly("logprobs", &inference::GenerationResult::logprobs)
        .def_readonly("total_logprob", &inference::GenerationResult::total_logprob)
        .def_readonly("generation_time_ms", &inference::GenerationResult::generation_time_ms)
        .def_readonly("tokens_generated", &inference::GenerationResult::tokens_generated)
        .def("tokens_per_second", &inference::GenerationResult::tokens_per_second);
    
    py::class_<inference::TokenSampler>(inf, "TokenSampler", "Token sampling strategies")
        .def(py::init<u32>(), py::arg("seed") = 42)
        .def("sample", [](inference::TokenSampler& self, py::array_t<f32> logits,
                         const inference::GenerationConfig& config) {
            return self.sample(numpy_to_vector(logits), config);
        }, py::arg("logits"), py::arg("config") = inference::GenerationConfig())
        .def("set_seed", &inference::TokenSampler::set_seed);
    
    py::class_<inference::InferenceEngine>(inf, "InferenceEngine",
        "High-performance inference engine")
        .def(py::init<u32>(), py::arg("seed") = 42)
        .def("generate", [](inference::InferenceEngine& self,
                           py::list input_ids,
                           py::function forward_fn,
                           const inference::GenerationConfig& config) {
            std::vector<i32> ids;
            for (auto item : input_ids) ids.push_back(item.cast<i32>());
            
            auto cpp_forward = [&forward_fn](const std::vector<i32>& tokens) {
                py::list py_tokens;
                for (auto t : tokens) py_tokens.append(t);
                py::array_t<f32> result = forward_fn(py_tokens).cast<py::array_t<f32>>();
                return numpy_to_vector(result);
            };
            
            return self.generate(ids, cpp_forward, config);
        }, py::arg("input_ids"), py::arg("forward_fn"), 
           py::arg("config") = inference::GenerationConfig())
        .def("set_seed", &inference::InferenceEngine::set_seed);
    
    // Sampling utilities
    inf.def("softmax", [](py::array_t<f32> logits) {
        return vector_to_numpy(inference::sampling::softmax(numpy_to_vector(logits)));
    }, py::arg("logits"), "Compute softmax");
    
    inf.def("greedy_sample", [](py::array_t<f32> logits) {
        return inference::sampling::greedy(numpy_to_vector(logits));
    }, py::arg("logits"), "Greedy sampling");
}












