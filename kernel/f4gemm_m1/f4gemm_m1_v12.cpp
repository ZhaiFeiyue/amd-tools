// FP4 GEMM M=1 v12: v11 with optimized address computation
// Eliminated all division/modulo from shuffled_b_offset by precomputing
// constants outside the K-loop. Only additions inside the loop.
//
// Key insight: k_byte = ki * 64 + k_row_group * 16
//   k_block = ki * 2 + (k_row_group >> 1)
//   k_half = k_row_group & 1
//   k_local = 0 (always aligned to 16 bytes)
//
// So B flat offset = n_block * B_NBLOCK_STRIDE
//                  + (ki * 2 + (k_row_group >> 1)) * 512
//                  + (k_row_group & 1) * 256
//                  + m_index * 16
// The ki-dependent part is just ki * 1024, which increments by 1024 each iter.

#include <hip/hip_runtime.h>
#include <hip/hip_bfloat16.h>

typedef int v8i __attribute__((ext_vector_type(8)));
typedef float v4f __attribute__((ext_vector_type(4)));

__device__ __forceinline__ int shuffled_scale_offset(int r, int c, int s_n_pad) {
    int i0 = r >> 5;         // r / 32
    int i1 = (r >> 4) & 1;   // (r%32) / 16
    int i2 = r & 15;          // r % 16
    int j0 = c >> 3;          // c / 8
    int j1 = (c >> 2) & 1;   // (c%8) / 4
    int j2 = c & 3;           // c % 4
    return i0 * (s_n_pad << 5) + (j0 << 8) + (j2 << 6) + (i2 << 2) + (j1 << 1) + i1;
}

extern "C"
__global__ __launch_bounds__(256, 2)
void f4gemm_m1_v12(
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

    // Precompute B offset constants (NO division in loop)
    int n_block = n_start >> 4;
    int K_bytes = K >> 1;
    int K_blocks = K_bytes >> 5;  // K_bytes / 32
    int b_nblock_stride = K_blocks * 512;  // K_blocks * 2 * 16 * 16
    int b_base = n_block * b_nblock_stride
               + ((k_row_group >> 1) << 9)   // (k_row_group/2) * 512
               + ((k_row_group & 1) << 8)    // (k_row_group%2) * 256
               + (m_index << 4);             // m_index * 16
    int b_ki_stride = 1024;  // Each ki increments k_block by 2 → 2 * 512 = 1024

    // A offset (linear, not shuffled)
    int a_grp_off = (k_row_group << 4);  // k_row_group * 16 bytes (32 fp4 = 16 bytes)

    int total_k = K >> 7;  // K / 128
    int k_begin = (total_k * wave_id) >> 2;
    int k_end = (total_k * (wave_id + 1)) >> 2;

    v4f c = {0.0f, 0.0f, 0.0f, 0.0f};

    int b_off = b_base + k_begin * b_ki_stride;
    int a_off = k_begin * 64 + a_grp_off;

    int ki = k_begin;
    for (; ki + 1 < k_end; ki += 2) {
        // --- Iter 0 ---
        const uint32_t* ap0 = (const uint32_t*)(A + a_off);
        v8i a0 = {(int)ap0[0], (int)ap0[1], (int)ap0[2], (int)ap0[3], 0, 0, 0, 0};

        const uint32_t* bp0 = (const uint32_t*)(B_shuffled + b_off);
        v8i b0 = {(int)bp0[0], (int)bp0[1], (int)bp0[2], (int)bp0[3], 0, 0, 0, 0};

        int sc0 = ki * 4 + k_row_group;
        int32_t sca0 = (int32_t)ScaleA[sc0];
        int32_t scb0 = valid_n ? (int32_t)ScaleB[shuffled_scale_offset(b_n, sc0, stride_SB)] : 0;

        // --- Iter 1 ---
        int a_off1 = a_off + 64;
        int b_off1 = b_off + b_ki_stride;

        const uint32_t* ap1 = (const uint32_t*)(A + a_off1);
        v8i a1 = {(int)ap1[0], (int)ap1[1], (int)ap1[2], (int)ap1[3], 0, 0, 0, 0};

        const uint32_t* bp1 = (const uint32_t*)(B_shuffled + b_off1);
        v8i b1 = {(int)bp1[0], (int)bp1[1], (int)bp1[2], (int)bp1[3], 0, 0, 0, 0};

        int sc1 = sc0 + 4;
        int32_t sca1 = (int32_t)ScaleA[sc1];
        int32_t scb1 = valid_n ? (int32_t)ScaleB[shuffled_scale_offset(b_n, sc1, stride_SB)] : 0;

        c = __builtin_amdgcn_mfma_scale_f32_16x16x128_f8f6f4(
            a0, b0, c, 4, 4, 0, sca0, 0, scb0);
        c = __builtin_amdgcn_mfma_scale_f32_16x16x128_f8f6f4(
            a1, b1, c, 4, 4, 0, sca1, 0, scb1);

        a_off += 128;     // 2 * 64
        b_off += b_ki_stride * 2;
    }
    for (; ki < k_end; ki++) {
        const uint32_t* ap = (const uint32_t*)(A + a_off);
        v8i a = {(int)ap[0], (int)ap[1], (int)ap[2], (int)ap[3], 0, 0, 0, 0};

        const uint32_t* bp = (const uint32_t*)(B_shuffled + b_off);
        v8i b = {(int)bp[0], (int)bp[1], (int)bp[2], (int)bp[3], 0, 0, 0, 0};

        int sc = ki * 4 + k_row_group;
        int32_t sca = (int32_t)ScaleA[sc];
        int32_t scb = valid_n ? (int32_t)ScaleB[shuffled_scale_offset(b_n, sc, stride_SB)] : 0;

        c = __builtin_amdgcn_mfma_scale_f32_16x16x128_f8f6f4(
            a, b, c, 4, 4, 0, sca, 0, scb);

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
    hipLaunchKernelGGL(f4gemm_m1_v12, grid, block, 0, 0,
        (hip_bfloat16*)D, (uint8_t*)A, (uint8_t*)B,
        (uint8_t*)ScaleA, (uint8_t*)ScaleB,
        M, N, K, stride_A, stride_B, stride_D, stride_SA, stride_SB);
    (void)hipDeviceSynchronize();
}
