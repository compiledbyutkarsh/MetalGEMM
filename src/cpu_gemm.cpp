#include "../include/gemm.hpp"
#include <Accelerate/Accelerate.h>
#include <cstring>

namespace MetalGEMM {

void GEMMEngine::cpu_naive(const Matrix &A, const Matrix &B, Matrix &C, const GEMMConfig &cfg) {
    size_t M = cfg.M, N = cfg.N, K = cfg.K;
    f32 alpha = cfg.alpha, beta = cfg.beta;

    for (size_t i = 0; i < M; i++) {
        for (size_t j = 0; j < N; j++) {
            f32 sum = 0.0f;
            for (size_t k = 0; k < K; k++) {
                sum += A(i, k) * B(k, j);
            }
            C(i, j) = alpha * sum + beta * C(i, j);
        }
    }
}

void GEMMEngine::cpu_accelerate(const Matrix &A, const Matrix &B, Matrix &C, const GEMMConfig &cfg) {
    size_t M = cfg.M, N = cfg.N, K = cfg.K;

    cblas_sgemm(
        CblasRowMajor,
        cfg.transpose_a ? CblasTrans : CblasNoTrans,
        cfg.transpose_b ? CblasTrans : CblasNoTrans,
        (int)M, (int)N, (int)K,
        cfg.alpha,
        A.data(), (int)(cfg.transpose_a ? M : K),
        B.data(), (int)(cfg.transpose_b ? K : N),
        cfg.beta,
        C.data(), (int)N
    );
}

void GEMMEngine::multiply(const Matrix &A, const Matrix &B, Matrix &C, const GEMMConfig &cfg) {
    switch (cfg.backend) {
        case Backend::CPU_NAIVE:       cpu_naive(A, B, C, cfg);       break;
        case Backend::CPU_ACCELERATE:  cpu_accelerate(A, B, C, cfg);  break;
        case Backend::METAL_GPU:       metal_gpu(A, B, C, cfg);       break;
    }
}

BenchmarkResult GEMMEngine::benchmark(size_t M, size_t N, size_t K, Backend backend, int warmup, int runs) {
    BenchmarkResult result;
    result.M = M; result.N = N; result.K = K;

    switch (backend) {
        case Backend::CPU_NAIVE:      result.backend_name = "CPU Naive";      break;
        case Backend::CPU_ACCELERATE: result.backend_name = "CPU Accelerate"; break;
        case Backend::METAL_GPU:      result.backend_name = "Metal GPU";      break;
    }

    Matrix A = Matrix::random(M, K);
    Matrix B = Matrix::random(K, N);
    Matrix C = Matrix::zeros(M, N);

    GEMMConfig cfg(M, N, K, 1.0f, 0.0f, false, false, backend);

    for (int i = 0; i < warmup; i++) {
        C.fill(0.0f);
        multiply(A, B, C, cfg);
    }

    f64 total_ms = 0.0;
    for (int i = 0; i < runs; i++) {
        C.fill(0.0f);
        auto t0 = Timer::now();
        multiply(A, B, C, cfg);
        auto t1 = Timer::now();
        total_ms += elapsed_ms(t0, t1);
    }

    result.time_ms   = total_ms / runs;
    result.gflops    = compute_gflops(M, N, K, result.time_ms);
    result.bandwidth_gb = (3.0 * M * N * K * sizeof(f32)) / (result.time_ms * 1e-3) / 1e9;
    result.correct   = true;

    return result;
}

void GEMMEngine::verify_correctness(size_t M, size_t N, size_t K) {
    std::cout << "\n[VERIFY] Correctness check " << M << "x" << N << "x" << K << "\n";

    Matrix A = Matrix::random(M, K, -1.0f, 1.0f);
    Matrix B = Matrix::random(K, N, -1.0f, 1.0f);

    Matrix C_ref  = Matrix::zeros(M, N);
    Matrix C_acc  = Matrix::zeros(M, N);
    Matrix C_metal = Matrix::zeros(M, N);

    GEMMConfig cfg(M, N, K);

    cfg.backend = Backend::CPU_NAIVE;
    multiply(A, B, C_ref, cfg);

    cfg.backend = Backend::CPU_ACCELERATE;
    multiply(A, B, C_acc, cfg);

    cfg.backend = Backend::METAL_GPU;
    multiply(A, B, C_metal, cfg);

    f32 diff_acc   = C_ref.max_diff(C_acc);
    f32 diff_metal = C_ref.max_diff(C_metal);

    std::cout << "  CPU Naive    vs Accelerate : max_diff = " << diff_acc   << (diff_acc   < 1e-3f ? " PASS" : " FAIL") << "\n";
    std::cout << "  CPU Naive    vs Metal GPU  : max_diff = " << diff_metal << (diff_metal < 1e-2f ? " PASS" : " FAIL") << "\n";
}

void GEMMEngine::run_all_benchmarks(size_t M, size_t N, size_t K) {
    std::cout << "\n";
    std::cout << "================================================================\n";
    std::cout << "  MetalGEMM Benchmark  |  " << M << " x " << N << " x " << K << "\n";
    std::cout << "================================================================\n";
    std::cout << std::left
              << std::setw(20) << "Backend"
              << std::setw(12) << "Time (ms)"
              << std::setw(12) << "GFLOP/s"
              << std::setw(16) << "Bandwidth (GB/s)"
              << "\n";
    std::cout << "----------------------------------------------------------------\n";

    auto print_result = [](const BenchmarkResult &r) {
        std::cout << std::left
                  << std::setw(20) << r.backend_name
                  << std::setw(12) << std::fixed << std::setprecision(2) << r.time_ms
                  << std::setw(12) << std::fixed << std::setprecision(2) << r.gflops
                  << std::setw(16) << std::fixed << std::setprecision(2) << r.bandwidth_gb
                  << "\n";
    };

    if (M <= 256) {
        print_result(benchmark(M, N, K, Backend::CPU_NAIVE));
    } else {
        std::cout << std::left << std::setw(20) << "CPU Naive"
                  << "skipped (matrix too large)\n";
    }

    print_result(benchmark(M, N, K, Backend::CPU_ACCELERATE));

    if (metal_available_) {
        print_result(benchmark(M, N, K, Backend::METAL_GPU));
    } else {
        std::cout << std::left << std::setw(20) << "Metal GPU"
                  << "not available\n";
    }

    std::cout << "================================================================\n";
}

} // namespace MetalGEMM
