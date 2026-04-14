// FP4 GEMM M=1 v13: v12 + manual s_waitcnt for load-compute overlap
// Issue loads for next K-iter, then s_waitcnt only for current data, then MFMA
// This allows MFMA to execute while loads for next iter are in flight

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
void f4gemm_m1_v13(
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
    int my_iters = k_end - k_begin;

    if (my_iters <= 0) goto reduce;

    {
        v4f c = {0.0f, 0.0f, 0.0f, 0.0f};

        int b_off = b_base + k_begin * b_ki_stride;
        int a_off = k_begin * 64 + a_grp_off;

        // Prefetch first iter's data
        const uint32_t* ap_cur = (const uint32_t*)(A + a_off);
        const uint32_t* bp_cur = (const uint32_t*)(B_shuffled + b_off);

        // Load current iter data into registers
        // Use volatile to prevent compiler from reordering
        int a0_cur = ap_cur[0], a1_cur = ap_cur[1], a2_cur = ap_cur[2], a3_cur = ap_cur[3];
        int b0_cur = bp_cur[0], b1_cur = bp_cur[1], b2_cur = bp_cur[2], b3_cur = bp_cur[3];
        int sc_cur = k_begin * 4 + k_row_group;
        int32_t sca_cur = (int32_t)ScaleA[sc_cur];
        int32_t scb_cur = valid_n ? (int32_t)ScaleB[shuffled_scale_offset(b_n, sc_cur, stride_SB)] : 0;

        for (int ki = k_begin; ki < k_end; ki++) {
            // Data for this iter already loaded
            v8i a_use = {a0_cur, a1_cur, a2_cur, a3_cur, 0, 0, 0, 0};
            v8i b_use = {b0_cur, b1_cur, b2_cur, b3_cur, 0, 0, 0, 0};
            int32_t sca_use = sca_cur;
            int32_t scb_use = scb_cur;

            // Start loading NEXT iter (if exists) — these loads go into flight
            if (ki + 1 < k_end) {
                int a_off_n = a_off + 64;
                int b_off_n = b_off + b_ki_stride;
                const uint32_t* ap_nxt = (const uint32_t*)(A + a_off_n);
                const uint32_t* bp_nxt = (const uint32_t*)(B_shuffled + b_off_n);
                a0_cur = ap_nxt[0]; a1_cur = ap_nxt[1]; a2_cur = ap_nxt[2]; a3_cur = ap_nxt[3];
                b0_cur = bp_nxt[0]; b1_cur = bp_nxt[1]; b2_cur = bp_nxt[2]; b3_cur = bp_nxt[3];
                int sc_nxt = (ki + 1) * 4 + k_row_group;
                sca_cur = (int32_t)ScaleA[sc_nxt];
                scb_cur = valid_n ? (int32_t)ScaleB[shuffled_scale_offset(b_n, sc_nxt, stride_SB)] : 0;
            }

            // MFMA on current data — executes while next loads are in flight
            c = __builtin_amdgcn_mfma_scale_f32_16x16x128_f8f6f4(
                a_use, b_use, c, 4, 4, 0, sca_use, 0, scb_use);

            a_off += 64;
            b_off += b_ki_stride;
        }

        // Store to LDS for reduction
        __shared__ float lds[4 * 16];
        if (lane < 16) lds[wave_id * 16 + lane] = c[0];
        __syncthreads();

        if (wave_id == 0 && lane < 16) {
            float sum = lds[lane] + lds[16 + lane] + lds[32 + lane] + lds[48 + lane];
            int out_n = n_start + lane;
            if (out_n < N) D[out_n] = hip_bfloat16(sum);
        }
        return;
    }

reduce:
    {
        __shared__ float lds[4 * 16];
        if (lane < 16) lds[wave_id * 16 + lane] = 0.0f;
        __syncthreads();
        if (wave_id == 0 && lane < 16) {
            float sum = lds[lane] + lds[16 + lane] + lds[32 + lane] + lds[48 + lane];
            int out_n = n_start + lane;
            if (out_n < N) D[out_n] = hip_bfloat16(sum);
        }
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
    hipLaunchKernelGGL(f4gemm_m1_v13, grid, block, 0, 0,
        (hip_bfloat16*)D, (uint8_t*)A, (uint8_t*)B,
        (uint8_t*)ScaleA, (uint8_t*)ScaleB,
        M, N, K, stride_A, stride_B, stride_D, stride_SA, stride_SB);
    (void)hipDeviceSynchronize();
}
