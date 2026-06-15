#pragma once

#include "types.hpp"
#include "matrix.hpp"

namespace MetalGEMM {

class GEMMEngine {
public:
    GEMMEngine();
    ~GEMMEngine();

    void multiply(const Matrix &A, const Matrix &B, Matrix &C, const GEMMConfig &config);

    BenchmarkResult benchmark(size_t M, size_t N, size_t K, Backend backend, int warmup = 3, int runs = 10);
    void            run_all_benchmarks(size_t M, size_t N, size_t K);
    void            verify_correctness(size_t M, size_t N, size_t K);

    bool is_metal_available() const { return metal_available_; }

private:
    void cpu_naive(const Matrix &A, const Matrix &B, Matrix &C, const GEMMConfig &cfg);
    void cpu_accelerate(const Matrix &A, const Matrix &B, Matrix &C, const GEMMConfig &cfg);
    void metal_gpu(const Matrix &A, const Matrix &B, Matrix &C, const GEMMConfig &cfg);

    bool metal_available_;
    void *metal_ctx_;
};

} // namespace MetalGEMM
