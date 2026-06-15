#include <metal_stdlib>
using namespace metal;

kernel void gemm_naive(
    device const float *A     [[ buffer(0) ]],
    device const float *B     [[ buffer(1) ]],
    device       float *C     [[ buffer(2) ]],
    constant     uint  &M     [[ buffer(3) ]],
    constant     uint  &N     [[ buffer(4) ]],
    constant     uint  &K     [[ buffer(5) ]],
    constant     float &alpha [[ buffer(6) ]],
    constant     float &beta  [[ buffer(7) ]],
    uint2 gid [[ thread_position_in_grid ]]
) {
    uint row = gid.y;
    uint col = gid.x;

    if (row >= M || col >= N) return;

    float sum = 0.0f;
    for (uint k = 0; k < K; k++) {
        sum += A[row * K + k] * B[k * N + col];
    }

    C[row * N + col] = alpha * sum + beta * C[row * N + col];
}

constant uint TILE_SIZE = 16;

kernel void gemm_tiled(
    device const float *A     [[ buffer(0) ]],
    device const float *B     [[ buffer(1) ]],
    device       float *C     [[ buffer(2) ]],
    constant     uint  &M     [[ buffer(3) ]],
    constant     uint  &N     [[ buffer(4) ]],
    constant     uint  &K     [[ buffer(5) ]],
    constant     float &alpha [[ buffer(6) ]],
    constant     float &beta  [[ buffer(7) ]],
    uint2 gid  [[ thread_position_in_grid ]],
    uint2 lid  [[ thread_position_in_threadgroup ]],
    uint2 grid [[ threadgroups_per_grid ]]
) {
    threadgroup float tileA[16][16];
    threadgroup float tileB[16][16];

    uint row = gid.y;
    uint col = gid.x;

    float acc = 0.0f;

    uint num_tiles = (K + TILE_SIZE - 1) / TILE_SIZE;

    for (uint t = 0; t < num_tiles; t++) {
        uint a_col = t * TILE_SIZE + lid.x;
        uint b_row = t * TILE_SIZE + lid.y;

        tileA[lid.y][lid.x] = (row < M && a_col < K) ? A[row * K + a_col] : 0.0f;
        tileB[lid.y][lid.x] = (b_row < K && col < N) ? B[b_row * N + col] : 0.0f;

        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint k = 0; k < TILE_SIZE; k++) {
            acc += tileA[lid.y][k] * tileB[k][lid.x];
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (row < M && col < N) {
        C[row * N + col] = alpha * acc + beta * C[row * N + col];
    }
}

kernel void gemm_simd(
    device const float *A     [[ buffer(0) ]],
    device const float *B     [[ buffer(1) ]],
    device       float *C     [[ buffer(2) ]],
    constant     uint  &M     [[ buffer(3) ]],
    constant     uint  &N     [[ buffer(4) ]],
    constant     uint  &K     [[ buffer(5) ]],
    constant     float &alpha [[ buffer(6) ]],
    constant     float &beta  [[ buffer(7) ]],
    uint2 gid [[ thread_position_in_grid ]]
) {
    uint row = gid.y;
    uint col = gid.x;

    if (row >= M || col >= N) return;

    float4 acc = float4(0.0f);
    uint k = 0;

    for (; k + 4 <= K; k += 4) {
        float4 a = float4(A[row * K + k],
                          A[row * K + k + 1],
                          A[row * K + k + 2],
                          A[row * K + k + 3]);
        float4 b = float4(B[(k + 0) * N + col],
                          B[(k + 1) * N + col],
                          B[(k + 2) * N + col],
                          B[(k + 3) * N + col]);
        acc += a * b;
    }

    float sum = acc.x + acc.y + acc.z + acc.w;
    for (; k < K; k++) {
        sum += A[row * K + k] * B[k * N + col];
    }

    C[row * N + col] = alpha * sum + beta * C[row * N + col];
}
