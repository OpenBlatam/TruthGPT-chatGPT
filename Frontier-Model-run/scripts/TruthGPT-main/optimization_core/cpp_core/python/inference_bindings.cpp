/**
 * @file inference_bindings.cpp
 * @brief PyBind11 bindings for inference module
 * 
 * Modular bindings for token sampling and inference engine.
 */

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <pybind11/functional.h>

#include "../include/inference/engine.hpp"
#include "../include/inference/sampling.hpp"
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
py::array_t<T> vector_to_numpy(const std::vector<T>& vec) {
    return py::array_t<T>(vec.size(), vec.data());
}

void register_inference_module(py::module_& m) {
    auto inference_module = m.def_submodule("inference",
        "High-performance inference engine");
    
    py::class_<inference::GenerationConfig>(inference_module, "GenerationConfig",
        R"doc(
            Generation configuration with builder pattern and presets.
            
            Example:
                >>> config = GenerationConfig.sampling(temp=0.8, top_p=0.9)
                >>> config.with_max_tokens(100)
        )doc")
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
        .def_static("greedy", &inference::GenerationConfig::greedy)
        .def_static("sampling", &inference::GenerationConfig::sampling,
                   py::arg("temp") = 0.8f, py::arg("top_p") = 0.9f)
        .def_static("beam", &inference::GenerationConfig::beam,
                   py::arg("num_beams") = 4)
        .def("__repr__", [](const inference::GenerationConfig& c) {
            std::ostringstream ss;
            ss << "GenerationConfig(max_tokens=" << c.max_new_tokens
               << ", temp=" << c.temperature
               << ", top_p=" << c.top_p << ")";
            return ss.str();
        });
    
    py::class_<inference::GenerationResult>(inference_module, "GenerationResult",
        "Result from token generation")
        .def_readonly("token_ids", &inference::GenerationResult::token_ids)
        .def_readonly("logprobs", &inference::GenerationResult::logprobs)
        .def_readonly("total_logprob", &inference::GenerationResult::total_logprob)
        .def_readonly("generation_time_ms", &inference::GenerationResult::generation_time_ms)
        .def_readonly("tokens_generated", &inference::GenerationResult::tokens_generated)
        .def("tokens_per_second", &inference::GenerationResult::tokens_per_second)
        .def("__repr__", [](const inference::GenerationResult& r) {
            std::ostringstream ss;
            ss << "GenerationResult(tokens=" << r.tokens_generated
               << ", time=" << r.generation_time_ms << "ms"
               << ", speed=" << r.tokens_per_second() << " tok/s)";
            return ss.str();
        });
    
    py::class_<inference::TokenSampler>(inference_module, "TokenSampler",
        "Token sampling strategies")
        .def(py::init<uint32_t>(), py::arg("seed") = 42)
        .def("sample", [](inference::TokenSampler& self, 
                         py::array_t<float> logits,
                         const inference::GenerationConfig& config) {
            return self.sample(numpy_to_vector(logits), config);
        }, py::arg("logits"), py::arg("config") = inference::GenerationConfig())
        .def("greedy", [](inference::TokenSampler& self, py::array_t<float> logits) {
            auto vec = numpy_to_vector(logits);
            return std::distance(vec.begin(), std::max_element(vec.begin(), vec.end()));
        }, py::arg("logits"))
        .def("set_seed", &inference::TokenSampler::set_seed);
    
    py::class_<inference::InferenceEngine>(inference_module, "InferenceEngine",
        R"doc(
            High-performance inference engine.
            
            Example:
                >>> engine = InferenceEngine(seed=42)
                >>> result = engine.generate(input_ids, forward_fn, config)
        )doc")
        .def(py::init<uint32_t>(), py::arg("seed") = 42)
        .def("generate", [](inference::InferenceEngine& self,
                           py::list input_ids,
                           py::function forward_fn,
                           const inference::GenerationConfig& config) {
            std::vector<int32_t> ids;
            for (auto item : input_ids) ids.push_back(item.cast<int32_t>());
            
            auto cpp_forward = [&forward_fn](const std::vector<int32_t>& tokens) {
                py::list py_tokens;
                for (auto t : tokens) py_tokens.append(t);
                py::array_t<float> result = forward_fn(py_tokens).cast<py::array_t<float>>();
                return numpy_to_vector(result);
            };
            
            return self.generate(ids, cpp_forward, config);
        }, py::arg("input_ids"), py::arg("forward_fn"), 
           py::arg("config") = inference::GenerationConfig())
        .def("set_seed", &inference::InferenceEngine::set_seed);
    
    inference_module.def("softmax", [](py::array_t<float> logits) {
        auto vec = numpy_to_vector(logits);
        float max_val = *std::max_element(vec.begin(), vec.end());
        float sum = 0.0f;
        for (auto& v : vec) {
            v = std::exp(v - max_val);
            sum += v;
        }
        for (auto& v : vec) v /= sum;
        return vector_to_numpy(vec);
    }, py::arg("logits"), "Compute softmax");
    
    inference_module.def("greedy_sample", [](py::array_t<float> logits) {
        auto vec = numpy_to_vector(logits);
        return static_cast<int>(std::distance(vec.begin(), 
                                              std::max_element(vec.begin(), vec.end())));
    }, py::arg("logits"), "Greedy sampling");
    
    inference_module.def("top_k_sample", [](py::array_t<float> logits, 
                                           int k, uint32_t seed) {
        auto vec = numpy_to_vector(logits);
        std::vector<std::pair<float, int>> indexed;
        for (size_t i = 0; i < vec.size(); ++i) {
            indexed.emplace_back(vec[i], static_cast<int>(i));
        }
        std::partial_sort(indexed.begin(), indexed.begin() + k, indexed.end(),
                         [](const auto& a, const auto& b) { return a.first > b.first; });
        
        float sum = 0.0f;
        for (int i = 0; i < k; ++i) {
            indexed[i].first = std::exp(indexed[i].first);
            sum += indexed[i].first;
        }
        
        float r = static_cast<float>(seed % 1000000) / 1000000.0f * sum;
        float cumsum = 0.0f;
        for (int i = 0; i < k; ++i) {
            cumsum += indexed[i].first;
            if (cumsum >= r) {
                return indexed[i].second;
            }
        }
        return indexed[k-1].second;
    }, py::arg("logits"), py::arg("k") = 50, py::arg("seed") = 42,
       "Top-k sampling");
}












