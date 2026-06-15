# 🧠 MetalGEMM

> GPU-accelerated General Matrix Multiplication (GEMM) built from scratch in C++ using Apple Metal and Accelerate framework.

![Language](https://img.shields.io/badge/Language-C%2B%2B17-blue?style=flat-square)
![GPU](https://img.shields.io/badge/GPU-Apple%20Metal-orange?style=flat-square)
![CPU](https://img.shields.io/badge/CPU-Accelerate%20BLAS-lightgrey?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)

---

## 💡 What is MetalGEMM?

MetalGEMM is a from-scratch implementation of high-performance matrix multiplication targeting Apple hardware. It implements three backends — a naive CPU baseline, Apple's Accelerate BLAS, and a custom Metal GPU compute kernel — and benchmarks them head-to-head.

GEMM (General Matrix Multiplication) is the core operation behind deep learning, scientific computing, and graphics. This project explores how far you can push hardware with tiled execution, SIMD vectorization, and GPU parallelism.

---

## 📊 Benchmark Results

Tested on MacBook Pro 2019 | AMD Radeon Pro 5500M | Intel Core i7

```
Matrix: 2048 x 2048 x 2048
================================================================
Backend             Time (ms)   GFLOP/s     Bandwidth (GB/s)
----------------------------------------------------------------
CPU Naive           ~50000      ~0.34       ~2.06
CPU Accelerate      62.21       276.18      1657.06
Metal GPU           505.58      33.98       203.88
================================================================
```

```
Matrix: 256 x 256 x 256
================================================================
Backend             Time (ms)   GFLOP/s     Bandwidth (GB/s)
----------------------------------------------------------------
CPU Naive           24.99       1.34        8.06
CPU Accelerate      0.24        137.25      823.49
Metal GPU           1.30        25.90       155.43
================================================================
```

---

## 🚀 Features

### 🖥️ Three Backends
- **CPU Naive** — baseline triple-loop implementation for correctness reference
- **CPU Accelerate** — Apple's BLAS via the Accelerate framework (cblas_sgemm)
- **Metal GPU** — custom compute kernels running on the discrete/integrated GPU

### 🧩 Metal Compute Kernels
- `gemm_naive` — straightforward GPU parallelism, one thread per output element
- `gemm_tiled` — threadgroup shared memory tiling for cache efficiency (16x16 tiles)
- `gemm_simd` — float4 SIMD vectorization for 4x throughput per thread

### ✅ Correctness Verification
- All backends verified against CPU naive reference
- Max diff tolerance: 1e-3 (float32 rounding expected)
- Tested at 64x64, 128x128, 256x256

### ⏱️ Benchmarking
- Warmup runs before timing
- Average over multiple runs
- Reports GFLOP/s and memory bandwidth (GB/s)

---

## 📁 Project Structure

```
MetalGEMM/
├── include/
│   ├── types.hpp       # Base types, GEMMConfig, BenchmarkResult
│   ├── matrix.hpp      # Matrix class
│   └── gemm.hpp        # GEMMEngine interface
├── src/
│   ├── main.cpp        # Entry point, CLI, benchmark runner
│   ├── matrix.cpp      # Matrix implementation
│   ├── cpu_gemm.cpp    # CPU backends (naive + Accelerate)
│   └── metal_gemm.mm   # Metal GPU backend (Objective-C++)
├── shaders/
│   └── gemm.metal      # Metal compute kernels
└── CMakeLists.txt
```

---

## 🛠️ Build Requirements

| Tool | Purpose |
|------|---------|
| macOS 12+ | Metal compute support |
| Xcode CLT | clang++ / metal compiler |
| cmake 3.16+ | Build system |

No external dependencies — Metal and Accelerate are built into macOS.

---

## 🔨 Building

```bash
git clone https://github.com/compiledbyutkarsh/MetalGEMM
cd MetalGEMM
mkdir build && cd build
cmake ..
make -j4
```

---

## 💻 Usage

```bash
# Run default benchmark + correctness check
./metalgemm

# Custom matrix size benchmark
./metalgemm -m 2048 -n 2048 -k 2048 -b

# Correctness verification only
./metalgemm -v

# Full benchmark suite
./metalgemm -b
```

---

## 📌 Roadmap

- [ ] Half precision (fp16) Metal kernels
- [ ] Batched GEMM support
- [ ] Matrix transpose optimization
- [ ] Apple Silicon Neural Engine integration
- [ ] Python bindings via pybind11

---

## 📜 License

MIT License - free to use, study, and build upon.

---

<p align="center">Made with 🧠 by <a href="https://github.com/compiledbyutkarsh">compiledbyutkarsh</a></p>
