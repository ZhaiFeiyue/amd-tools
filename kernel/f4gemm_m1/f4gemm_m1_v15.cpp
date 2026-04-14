// FP4 GEMM M=1 v15: v12 base + fully unrolled K-loop
// Each wave does 14 K-iters (K=7168, 4 waves, 56 total / 4 = 14)
// Fully unroll with #pragma unroll for maximum load-MFMA overlap

#include <hip/hip_runtime.h>
#include <hip/hip_bfloat16.h>

typedef int v8i __attribute__((ext_vector_type(8)));
typedef float v4f __attribute__((ext_vector_type(4)));

__device__ __forceinline__ int shuffled_scale_offset(int r, int c, int s_n_pad) {
    int i0 = r >> 5;
    int i1 = (r >> 4) & 1;
    int i2 = r & 15;
    int j0 = c >> 3;
    int j1 = (c >> 2) & 1;
    int j2 = c & 3;
    return i0 * (s_n_pad << 5) + (j0 << 8) + (j2 << 6) + (i2 << 2) + (j1 << 1) + i1;
}

extern "C"
__global__ __launch_bounds__(256, 2)
void f4gemm_m1_v15(
    hip_bfloat16* __restrict__ D,
    const uint8_t* __restrict__ A,
    const uint8_t* __restrict__ B_shuffled,
    const uint8_t* __restrict__ ScaleA,
    const uint8_t* __restrict__ ScaleB,
    int M, int N, int K,
    int stride_A, int stride_B, int stride_D,
    int stride_SA, int stride_SB
) {
    int n_tile = blockIdx.x;
    int tid = threadIdx.x;
    int wave_id = tid / 64;
    int lane = tid % 64;
    int m_index = lane & 15;
    int k_row_group = lane >> 4;

    int n_start = n_tile * 16;
    if (n_start >= N) return;

    int b_n = n_start + m_index;
    bool valid_n = (b_n < N);

    int n_block = n_start >> 4;
    int K_bytes = K >> 1;
    int K_blocks = K_bytes >> 5;
    int b_nblock_stride = K_blocks << 9;
    int b_base = n_block * b_nblock_stride
               + ((k_row_group >> 1) << 9)
               + ((k_row_group & 1) << 8)
               + (m_index << 4);
    int b_ki_stride = 1024;
    int a_grp_off = k_row_group << 4;

    int total_k = K >> 7;
    int k_begin = (total_k * wave_id) >> 2;
    int k_end = (total_k * (wave_id + 1)) >> 2;

    v4f c = {0.0f, 0.0f, 0.0f, 0.0f};

    int b_off = b_base + k_begin * b_ki_stride;
    int a_off = k_begin * 64 + a_grp_off;

    #pragma unroll
    for (int ki = k_begin; ki < k_end; ki++) {
        const uint32_t* ap = (const uint32_t*)(A + a_off);
        const uint32_t* bp = (const uint32_t*)(B_shuffled + b_off);
        v8i a = {(int)ap[0], (int)ap[1], (int)ap[2], (int)ap[3], 0, 0, 0, 0};
        v8i b = {(int)bp[0], (int)bp[1], (int)bp[2], (int)bp[3], 0, 0, 0, 0};

        int sc = ki * 4 + k_row_group;
        int32_t sca = (int32_t)ScaleA[sc];
        int32_t scb = valid_n ? (int32_t)ScaleB[shuffled_scale_offset(b_n, sc, stride_SB)] : 0;

        c = __builtin_amdgcn_mfma_scale_f32_16x16x128_f8f6f4(a, b, c, 4, 4, 0, sca, 0, scb);

        a_off += 64;
        b_off += b_ki_stride;
    }

    __shared__ float lds[4 * 16];
    if (lane < 16) lds[wave_id * 16 + lane] = c[0];
    __syncthreads();

    if (wave_id == 0 && lane < 16) {
        float sum = lds[lane] + lds[16 + lane] + lds[32 + lane] + lds[48 + lane];
        int out_n = n_start + lane;
        if (out_n < N) D[out_n] = hip_bfloat16(sum);
    }
}

extern "C" void launch_f4gemm_m1(
    void* D, void* A, void* B, void* ScaleA, void* ScaleB,
    int M, int N, int K,
    int stride_A, int stride_B, int stride_D,
    int stride_SA, int stride_SB
) {
    dim3 grid((N + 15) / 16, 1, 1);
    dim3 block(256, 1, 1);
    hipLaunchKernelGGL(f4gemm_m1_v15, grid, block, 0, 0,
        (hip_bfloat16*)D, (uint8_t*)A, (uint8_t*)B,
        (uint8_t*)ScaleA, (uint8_t*)ScaleB,
        M, N, K, stride_A, stride_B, stride_D, stride_SA, stride_SB);
    (void)hipDeviceSynchronize();
}
