/**
 * @file memory_bindings.cpp
 * @brief PyBind11 bindings for memory module
 * 
 * Modular bindings for KV Cache and memory management.
 */

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <pybind11/functional.h>

#include "../include/memory/kv_cache.hpp"
#include <sstream>
#include <iomanip>

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

void register_memory_module(py::module_& m) {
    auto memory_module = m.def_submodule("memory",
        "High-performance memory management and caching");
    
    py::enum_<memory::EvictionStrategy>(memory_module, "EvictionStrategy",
        "Cache eviction strategies")
        .value("LRU", memory::EvictionStrategy::LRU)
        .value("LFU", memory::EvictionStrategy::LFU)
        .value("FIFO", memory::EvictionStrategy::FIFO)
        .value("S3FIFO", memory::EvictionStrategy::S3FIFO)
        .value("ARC", memory::EvictionStrategy::ARC)
        .value("Adaptive", memory::EvictionStrategy::Adaptive)
        .value("TwoQ", memory::EvictionStrategy::TwoQ)
        .value("None", memory::EvictionStrategy::None)
        .export_values();
    
    py::enum_<memory::CompressionAlgorithm>(memory_module, "CompressionAlgorithm",
        "Compression algorithms")
        .value("None", memory::CompressionAlgorithm::None)
        .value("LZ4", memory::CompressionAlgorithm::LZ4)
        .value("LZ4_HC", memory::CompressionAlgorithm::LZ4_HC)
        .value("ZSTD", memory::CompressionAlgorithm::ZSTD)
        .value("ZSTD_Fast", memory::CompressionAlgorithm::ZSTD_Fast)
        .value("ZSTD_High", memory::CompressionAlgorithm::ZSTD_High)
        .export_values();
    
    py::class_<memory::KVCacheConfig>(memory_module, "KVCacheConfig",
        R"doc(
            Configuration for KV Cache.
            
            Example:
                >>> config = KVCacheConfig(max_cache_size=4*1024*1024*1024)
                >>> cache = UltraKVCache(config)
        )doc")
        .def(py::init<>())
        .def(py::init([](size_t max_cache_size, size_t max_entries,
                        float eviction_threshold, float eviction_target,
                        memory::EvictionStrategy strategy,
                        bool use_compression,
                        memory::CompressionAlgorithm compression_algorithm,
                        size_t compression_threshold,
                        int num_shards) {
            memory::KVCacheConfig config;
            config.max_cache_size = max_cache_size;
            config.max_entries = max_entries;
            config.eviction_threshold = eviction_threshold;
            config.eviction_target = eviction_target;
            config.eviction_strategy = strategy;
            config.use_compression = use_compression;
            config.compression_algorithm = compression_algorithm;
            config.compression_threshold = compression_threshold;
            config.num_shards = num_shards;
            return config;
        }),
        py::arg("max_cache_size") = 8ULL * 1024 * 1024 * 1024,
        py::arg("max_entries") = 1000000,
        py::arg("eviction_threshold") = 0.85f,
        py::arg("eviction_target") = 0.70f,
        py::arg("eviction_strategy") = memory::EvictionStrategy::S3FIFO,
        py::arg("use_compression") = true,
        py::arg("compression_algorithm") = memory::CompressionAlgorithm::LZ4,
        py::arg("compression_threshold") = 4096,
        py::arg("num_shards") = 32)
        .def_readwrite("max_cache_size", &memory::KVCacheConfig::max_cache_size)
        .def_readwrite("max_entries", &memory::KVCacheConfig::max_entries)
        .def_readwrite("eviction_threshold", &memory::KVCacheConfig::eviction_threshold)
        .def_readwrite("eviction_target", &memory::KVCacheConfig::eviction_target)
        .def_readwrite("eviction_strategy", &memory::KVCacheConfig::eviction_strategy)
        .def_readwrite("use_compression", &memory::KVCacheConfig::use_compression)
        .def_readwrite("compression_algorithm", &memory::KVCacheConfig::compression_algorithm)
        .def_readwrite("compression_threshold", &memory::KVCacheConfig::compression_threshold)
        .def_readwrite("num_shards", &memory::KVCacheConfig::num_shards)
        .def("validate", &memory::KVCacheConfig::validate)
        .def_static("inference_optimized", &memory::KVCacheConfig::inference_optimized,
            py::arg("memory_budget_gb") = 8)
        .def_static("long_context", &memory::KVCacheConfig::long_context,
            py::arg("memory_budget_gb") = 32)
        .def("__repr__", [](const memory::KVCacheConfig& c) {
            std::ostringstream ss;
            ss << "KVCacheConfig(max=" << c.max_cache_size / (1024*1024*1024) << "GB"
               << ", entries=" << c.max_entries << ")";
            return ss.str();
        });
    
    py::class_<memory::CacheStats>(memory_module, "CacheStats",
        "Cache performance statistics")
        .def(py::init<>())
        .def_property_readonly("hit_count", [](const memory::CacheStats& s) {
            return s.hit_count.load();
        })
        .def_property_readonly("miss_count", [](const memory::CacheStats& s) {
            return s.miss_count.load();
        })
        .def_property_readonly("eviction_count", [](const memory::CacheStats& s) {
            return s.eviction_count.load();
        })
        .def_property_readonly("compression_count", [](const memory::CacheStats& s) {
            return s.compression_count.load();
        })
        .def_property_readonly("current_memory_bytes", [](const memory::CacheStats& s) {
            return s.current_memory_bytes.load();
        })
        .def_property_readonly("peak_memory_bytes", [](const memory::CacheStats& s) {
            return s.peak_memory_bytes.load();
        })
        .def_property_readonly("entry_count", [](const memory::CacheStats& s) {
            return s.current_entry_count.load();
        })
        .def("hit_rate", &memory::CacheStats::hit_rate)
        .def("avg_get_latency_us", &memory::CacheStats::avg_get_latency_us)
        .def("avg_put_latency_us", &memory::CacheStats::avg_put_latency_us)
        .def("to_dict", &memory::CacheStats::to_map)
        .def("reset", &memory::CacheStats::reset)
        .def("__repr__", [](const memory::CacheStats& s) {
            std::ostringstream ss;
            ss << "CacheStats(hit_rate=" << std::fixed << std::setprecision(2) 
               << (s.hit_rate() * 100) << "%, entries=" << s.current_entry_count.load()
               << ", mem=" << s.current_memory_bytes.load() / (1024*1024) << "MB)";
            return ss.str();
        });
    
    py::class_<memory::UltraKVCache<float>>(memory_module, "UltraKVCache",
        R"doc(
            High-performance KV Cache for LLM inference.
            
            Features:
            - Lock-free concurrent access with sharding
            - Multiple eviction strategies
            - Integrated compression
            
            Example:
                >>> cache = UltraKVCache(KVCacheConfig())
                >>> cache.put(layer_idx=0, position=42, key_state=k, value_state=v)
        )doc")
        .def(py::init<const memory::KVCacheConfig&>(),
            py::arg("config") = memory::KVCacheConfig())
        .def("get", [](memory::UltraKVCache<float>& self,
                      int layer_idx, int position,
                      const std::string& key) -> py::object {
            auto result = self.get(layer_idx, position, key);
            if (result.has_value()) {
                auto& state = result.value();
                py::dict ret;
                
                std::vector<ssize_t> shape = {
                    state.batch_size, state.n_heads, state.seq_len, state.head_dim
                };
                
                if (!state.key_state.empty()) {
                    ret["key"] = vector_to_numpy(state.key_state, shape);
                    ret["value"] = vector_to_numpy(state.value_state, shape);
                }
                ret["layer_idx"] = state.layer_idx;
                ret["position"] = state.position;
                ret["access_count"] = state.metadata.access_count;
                ret["is_compressed"] = state.metadata.is_compressed;
                return ret;
            }
            return py::none();
        },
        py::arg("layer_idx"), py::arg("position"), py::arg("key") = "")
        .def("put", [](memory::UltraKVCache<float>& self,
                      int layer_idx, int position,
                      py::array_t<float> key_state,
                      py::array_t<float> value_state,
                      const std::string& key,
                      float priority) {
            memory::KVState<float> state;
            state.key_state = numpy_to_vector(key_state);
            state.value_state = numpy_to_vector(value_state);
            
            auto k_shape = key_state.request();
            if (k_shape.ndim >= 4) {
                state.batch_size = static_cast<int>(k_shape.shape[0]);
                state.n_heads = static_cast<int>(k_shape.shape[1]);
                state.seq_len = static_cast<int>(k_shape.shape[2]);
                state.head_dim = static_cast<int>(k_shape.shape[3]);
            }
            
            self.put(layer_idx, position, std::move(state), key, priority);
        },
        py::arg("layer_idx"), py::arg("position"),
        py::arg("key_state"), py::arg("value_state"),
        py::arg("key") = "", py::arg("priority") = 1.0f)
        .def("remove", &memory::UltraKVCache<float>::remove,
            py::arg("layer_idx"), py::arg("position"), py::arg("key") = "")
        .def("pin", &memory::UltraKVCache<float>::pin,
            py::arg("layer_idx"), py::arg("position"), py::arg("key") = "")
        .def("unpin", &memory::UltraKVCache<float>::unpin,
            py::arg("layer_idx"), py::arg("position"), py::arg("key") = "")
        .def("contains", &memory::UltraKVCache<float>::contains,
            py::arg("layer_idx"), py::arg("position"), py::arg("key") = "")
        .def("clear", &memory::UltraKVCache<float>::clear)
        .def("size", &memory::UltraKVCache<float>::size)
        .def("memory_usage", &memory::UltraKVCache<float>::memory_usage)
        .def("max_size", &memory::UltraKVCache<float>::max_size)
        .def("empty", &memory::UltraKVCache<float>::empty)
        .def_property_readonly("stats", &memory::UltraKVCache<float>::stats,
            py::return_value_policy::reference_internal)
        .def("reset_stats", &memory::UltraKVCache<float>::reset_stats)
        .def("__len__", &memory::UltraKVCache<float>::size)
        .def("__bool__", [](const memory::UltraKVCache<float>& c) {
            return !c.empty();
        })
        .def("__repr__", [](const memory::UltraKVCache<float>& c) {
            std::ostringstream ss;
            ss << "UltraKVCache(entries=" << c.size()
               << ", mem=" << c.memory_usage() / (1024*1024) << "MB"
               << "/" << c.max_size() / (1024*1024*1024) << "GB)";
            return ss.str();
        });
}












