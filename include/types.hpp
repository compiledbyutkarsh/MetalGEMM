#pragma once

#include <cstdint>
#include <cstddef>
#include <string>
#include <vector>
#include <chrono>
#include <functional>

using u8    = uint8_t;
using u16   = uint16_t;
using u32   = uint32_t;
using u64   = uint64_t;
using i32   = int32_t;
using i64   = int64_t;
using f32   = float;
using f64   = double;

namespace MetalGEMM {

enum class Backend {
    CPU_NAIVE,
    CPU_ACCELERATE,
    METAL_GPU
};

struct GEMMConfig {
    size_t  M;
    size_t  N;
    size_t  K;
    f32     alpha;
    f32     beta;
    bool    transpose_a;
    bool    transpose_b;
    Backend backend;

    GEMMConfig(size_t m, size_t n, size_t k,
               f32 a = 1.0f, f32 b = 0.0f,
               bool ta = false, bool tb = false,
               Backend back = Backend::METAL_GPU)
        : M(m), N(n), K(k), alpha(a), beta(b),
          transpose_a(ta), transpose_b(tb), backend(back) {}
};

struct BenchmarkResult {
    std::string backend_name;
    size_t      M, N, K;
    f64         time_ms;
    f64         gflops;
    f64         bandwidth_gb;
    bool        correct;

    BenchmarkResult() : M(0), N(0), K(0), time_ms(0),
                        gflops(0), bandwidth_gb(0), correct(false) {}
};

using Timer = std::chrono::high_resolution_clock;
using TimePoint = std::chrono::time_point<Timer>;

inline f64 elapsed_ms(TimePoint start, TimePoint end) {
    return std::chrono::duration<f64, std::milli>(end - start).count();
}

inline f64 compute_gflops(size_t M, size_t N, size_t K, f64 time_ms) {
    f64 flops = 2.0 * M * N * K;
    return (flops / (time_ms * 1e-3)) / 1e9;
}

} // namespace MetalGEMM
