#pragma once

/**
 * @file types.hpp
 * @brief Common type definitions and utilities
 */

#include <cstdint>
#include <chrono>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <variant>
#include <vector>

namespace optimization_core {

// ============================================================================
// Type Aliases
// ============================================================================

using f32 = float;
using f64 = double;
using i32 = int32_t;
using i64 = int64_t;
using u32 = uint32_t;
using u64 = uint64_t;
using usize = size_t;

using Clock = std::chrono::high_resolution_clock;
using TimePoint = Clock::time_point;
using Duration = std::chrono::duration<double>;
using Milliseconds = std::chrono::duration<double, std::milli>;
using Microseconds = std::chrono::duration<double, std::micro>;

// ============================================================================
// Result Type (Rust-like)
// ============================================================================

template<typename T, typename E = std::string>
class Result {
public:
    static Result ok(T value) { return Result(std::move(value), true); }
    static Result err(E error) { return Result(std::move(error), false); }
    
    bool is_ok() const { return is_ok_; }
    bool is_err() const { return !is_ok_; }
    
    T& unwrap() {
        if (!is_ok_) throw std::runtime_error("Called unwrap on error result");
        return std::get<T>(data_);
    }
    
    const T& unwrap() const {
        if (!is_ok_) throw std::runtime_error("Called unwrap on error result");
        return std::get<T>(data_);
    }
    
    T unwrap_or(T default_value) const {
        return is_ok_ ? std::get<T>(data_) : default_value;
    }
    
    const E& error() const {
        if (is_ok_) throw std::runtime_error("Called error on ok result");
        return std::get<E>(data_);
    }
    
    template<typename F>
    auto map(F&& f) const -> Result<decltype(f(std::declval<T>())), E> {
        using U = decltype(f(std::declval<T>()));
        if (is_ok_) {
            return Result<U, E>::ok(f(std::get<T>(data_)));
        }
        return Result<U, E>::err(std::get<E>(data_));
    }

private:
    Result(T value, bool) : data_(std::move(value)), is_ok_(true) {}
    Result(E error, bool) : data_(std::move(error)), is_ok_(false) {}
    
    std::variant<T, E> data_;
    bool is_ok_;
};

// ============================================================================
// Tensor Shape
// ============================================================================

struct Shape {
    std::vector<usize> dims;
    
    Shape() = default;
    Shape(std::initializer_list<usize> d) : dims(d) {}
    explicit Shape(std::vector<usize> d) : dims(std::move(d)) {}
    
    usize rank() const { return dims.size(); }
    
    usize numel() const {
        if (dims.empty()) return 0;
        usize n = 1;
        for (auto d : dims) n *= d;
        return n;
    }
    
    usize operator[](usize i) const { return dims[i]; }
    usize& operator[](usize i) { return dims[i]; }
    
    bool operator==(const Shape& other) const { return dims == other.dims; }
    bool operator!=(const Shape& other) const { return dims != other.dims; }
    
    std::string to_string() const {
        std::string s = "[";
        for (usize i = 0; i < dims.size(); ++i) {
            s += std::to_string(dims[i]);
            if (i + 1 < dims.size()) s += ", ";
        }
        return s + "]";
    }
};

// ============================================================================
// Data Type Enum
// ============================================================================

enum class DType {
    Float32,
    Float16,
    BFloat16,
    Int32,
    Int64,
    Int8,
    UInt8,
};

inline usize dtype_size(DType dtype) {
    switch (dtype) {
        case DType::Float32: return 4;
        case DType::Float16: return 2;
        case DType::BFloat16: return 2;
        case DType::Int32: return 4;
        case DType::Int64: return 8;
        case DType::Int8: return 1;
        case DType::UInt8: return 1;
        default: return 0;
    }
}

// ============================================================================
// Timer Utility
// ============================================================================

class Timer {
public:
    Timer() : start_(Clock::now()) {}
    
    void reset() { start_ = Clock::now(); }
    
    double elapsed_ms() const {
        return std::chrono::duration<double, std::milli>(Clock::now() - start_).count();
    }
    
    double elapsed_us() const {
        return std::chrono::duration<double, std::micro>(Clock::now() - start_).count();
    }
    
    double elapsed_s() const {
        return std::chrono::duration<double>(Clock::now() - start_).count();
    }

private:
    TimePoint start_;
};

// ============================================================================
// Scope Guard (RAII)
// ============================================================================

template<typename F>
class ScopeGuard {
public:
    explicit ScopeGuard(F&& f) : func_(std::forward<F>(f)), active_(true) {}
    ~ScopeGuard() { if (active_) func_(); }
    
    ScopeGuard(const ScopeGuard&) = delete;
    ScopeGuard& operator=(const ScopeGuard&) = delete;
    
    ScopeGuard(ScopeGuard&& other) noexcept 
        : func_(std::move(other.func_)), active_(other.active_) {
        other.active_ = false;
    }
    
    void dismiss() { active_ = false; }

private:
    F func_;
    bool active_;
};

template<typename F>
ScopeGuard<F> make_scope_guard(F&& f) {
    return ScopeGuard<F>(std::forward<F>(f));
}

} // namespace optimization_core












