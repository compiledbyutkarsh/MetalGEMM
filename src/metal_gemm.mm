#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#include "../include/gemm.hpp"
#include "../include/matrix.hpp"
#include <iostream>
#include <stdexcept>

namespace MetalGEMM {

struct MetalContext {
    id<MTLDevice>              device;
    id<MTLCommandQueue>        queue;
    id<MTLLibrary>             library;
    id<MTLComputePipelineState> naive_pso;
    id<MTLComputePipelineState> tiled_pso;
    id<MTLComputePipelineState> simd_pso;
};

static MetalContext *init_metal() {
    MetalContext *ctx = new MetalContext();

    ctx->device = MTLCreateSystemDefaultDevice();
    if (!ctx->device) {
        delete ctx;
        return nullptr;
    }

    ctx->queue = [ctx->device newCommandQueue];

    NSString *shader_path = [[NSBundle mainBundle] pathForResource:@"gemm" ofType:@"metal"];
    NSString *source = nullptr;

    if (shader_path) {
        source = [NSString stringWithContentsOfFile:shader_path
                                           encoding:NSUTF8StringEncoding
                                              error:nil];
    }

    if (!source) {
        NSString *src_dir = [NSString stringWithUTF8String:
            (std::string(__FILE__).substr(0, std::string(__FILE__).rfind('/')) + "/../shaders/gemm.metal").c_str()
        ];
        source = [NSString stringWithContentsOfFile:src_dir
                                           encoding:NSUTF8StringEncoding
                                              error:nil];
    }

    if (!source) {
        std::cerr << "[Metal] Could not load shader source\n";
        delete ctx;
        return nullptr;
    }

    NSError *error = nil;
    MTLCompileOptions *opts = [[MTLCompileOptions alloc] init];
    opts.fastMathEnabled = YES;

    ctx->library = [ctx->device newLibraryWithSource:source options:opts error:&error];
    if (!ctx->library) {
        std::cerr << "[Metal] Shader compile error: "
                  << [[error localizedDescription] UTF8String] << "\n";
        delete ctx;
        return nullptr;
    }

    auto make_pso = [&](const char *name) -> id<MTLComputePipelineState> {
        id<MTLFunction> fn = [ctx->library newFunctionWithName:[NSString stringWithUTF8String:name]];
        if (!fn) {
            std::cerr << "[Metal] Function not found: " << name << "\n";
            return nil;
        }
        NSError *pso_err = nil;
        auto pso = [ctx->device newComputePipelineStateWithFunction:fn error:&pso_err];
        if (!pso) {
            std::cerr << "[Metal] PSO error: " << [[pso_err localizedDescription] UTF8String] << "\n";
        }
        return pso;
    };

    ctx->naive_pso = make_pso("gemm_naive");
    ctx->tiled_pso = make_pso("gemm_tiled");
    ctx->simd_pso  = make_pso("gemm_simd");

    if (!ctx->naive_pso || !ctx->tiled_pso || !ctx->simd_pso) {
        delete ctx;
        return nullptr;
    }

    return ctx;
}

void metal_multiply(MetalContext *ctx,
                    const Matrix &A, const Matrix &B, Matrix &C,
                    const GEMMConfig &cfg,
                    id<MTLComputePipelineState> pso) {
    size_t M = cfg.M, N = cfg.N, K = cfg.K;

    id<MTLBuffer> buf_a = [ctx->device newBufferWithBytes:A.data()
                                                   length:A.bytes()
                                                  options:MTLResourceStorageModeShared];
    id<MTLBuffer> buf_b = [ctx->device newBufferWithBytes:B.data()
                                                   length:B.bytes()
                                                  options:MTLResourceStorageModeShared];
    id<MTLBuffer> buf_c = [ctx->device newBufferWithBytes:C.data()
                                                   length:C.bytes()
                                                  options:MTLResourceStorageModeShared];

    uint um = (uint)M, un = (uint)N, uk = (uint)K;
    float alpha = cfg.alpha, beta = cfg.beta;

    id<MTLBuffer> buf_m     = [ctx->device newBufferWithBytes:&um    length:4 options:MTLResourceStorageModeShared];
    id<MTLBuffer> buf_n     = [ctx->device newBufferWithBytes:&un    length:4 options:MTLResourceStorageModeShared];
    id<MTLBuffer> buf_k     = [ctx->device newBufferWithBytes:&uk    length:4 options:MTLResourceStorageModeShared];
    id<MTLBuffer> buf_alpha = [ctx->device newBufferWithBytes:&alpha length:4 options:MTLResourceStorageModeShared];
    id<MTLBuffer> buf_beta  = [ctx->device newBufferWithBytes:&beta  length:4 options:MTLResourceStorageModeShared];

    id<MTLCommandBuffer>      cmd = [ctx->queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];

    [enc setComputePipelineState:pso];
    [enc setBuffer:buf_a   offset:0 atIndex:0];
    [enc setBuffer:buf_b   offset:0 atIndex:1];
    [enc setBuffer:buf_c   offset:0 atIndex:2];
    [enc setBuffer:buf_m   offset:0 atIndex:3];
    [enc setBuffer:buf_n   offset:0 atIndex:4];
    [enc setBuffer:buf_k   offset:0 atIndex:5];
    [enc setBuffer:buf_alpha offset:0 atIndex:6];
    [enc setBuffer:buf_beta  offset:0 atIndex:7];

    MTLSize threads_per_group = MTLSizeMake(16, 16, 1);
    MTLSize grid_size = MTLSizeMake(
        (N + 15) / 16,
        (M + 15) / 16,
        1
    );

    [enc dispatchThreadgroups:grid_size threadsPerThreadgroup:threads_per_group];
    [enc endEncoding];
    [cmd commit];
    [cmd waitUntilCompleted];

    memcpy(C.data(), [buf_c contents], C.bytes());
}

GEMMEngine::GEMMEngine() : metal_available_(false), metal_ctx_(nullptr) {
    MetalContext *ctx = init_metal();
    if (ctx) {
        metal_ctx_       = ctx;
        metal_available_ = true;
        std::cout << "[Metal] Device: "
                  << [((MetalContext*)metal_ctx_)->device.name UTF8String]
                  << "\n";
    } else {
        std::cerr << "[Metal] Not available, falling back to CPU\n";
    }
}

GEMMEngine::~GEMMEngine() {
    if (metal_ctx_) {
        delete (MetalContext*)metal_ctx_;
        metal_ctx_ = nullptr;
    }
}

void GEMMEngine::metal_gpu(const Matrix &A, const Matrix &B, Matrix &C, const GEMMConfig &cfg) {
    if (!metal_available_) {
        cpu_accelerate(A, B, C, cfg);
        return;
    }
    MetalContext *ctx = (MetalContext*)metal_ctx_;
    metal_multiply(ctx, A, B, C, cfg, ctx->tiled_pso);
}

} // namespace MetalGEMM
