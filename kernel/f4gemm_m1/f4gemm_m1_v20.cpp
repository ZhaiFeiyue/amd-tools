// v20: Isolate optimization #5 — large N-tile (128), 8 MFMAs per K-iter
// Single wave per WG, no LDS, no K-split, direct global load
// Purpose: measure how much A-data reuse across 8 MFMAs helps

#include <hip/hip_runtime.h>
#include <hip/hip_bfloat16.h>

typedef int v8i __attribute__((ext_vector_type(8)));
typedef float v4f __attribute__((ext_vector_type(4)));

extern "C"
__global__ __launch_bounds__(64, 4)
void f4gemm_m1_v20(
    hip_bfloat16* __restrict__ D,
    const uint8_t* __restrict__ A,
    const uint8_t* __restrict__ B,
    const uint8_t* __restrict__ ScaleA,
    const uint8_t* __restrict__ ScaleB,
    int M, int N, int K,
    int stride_A, int stride_B, int stride_D,
    int stride_SA, int stride_SB
) {
    int n_tile = blockIdx.x;
    int lane = threadIdx.x;
    int m_index = lane & 15;
    int k_row_group = lane >> 4;
    int n_start = n_tile * 128;
    if (n_start >= N) return;

    int a_grp_off = k_row_group << 4;
    int k_iters = K >> 7;

    // 8 subtiles, each 16 N-columns. Process sequentially.
    for (int sub = 0; sub < 8; sub++) {
        int bn = n_start + sub * 16 + m_index;
        bool vn = (bn < N);
        int b_base = vn ? (bn * stride_B) : 0;
        int sb_off = vn ? (bn * stride_SB) : 0;

        v4f c = {0.0f, 0.0f, 0.0f, 0.0f};

        for (int ki = 0; ki < k_iters; ki++) {
            int koff = ki * 64 + a_grp_off;

            const uint32_t* ap = (const uint32_t*)(A + koff);
            v8i a = {(int)ap[0], (int)ap[1], (int)ap[2], (int)ap[3], 0, 0, 0, 0};

            v8i b = {0, 0, 0, 0, 0, 0, 0, 0};
            if (vn) {
                const uint32_t* bp = (const uint32_t*)(B + b_base + koff);
                b[0] = bp[0]; b[1] = bp[1]; b[2] = bp[2]; b[3] = bp[3];
            }

            int sc = ki * 4 + k_row_group;
            int32_t sca = (int32_t)ScaleA[sc];
            int32_t scb = vn ? (int32_t)ScaleB[sb_off + sc] : 0;

            c = __builtin_amdgcn_mfma_scale_f32_16x16x128_f8f6f4(
                a, b, c, 4, 4, 0, sca, 0, scb);
        }

        if (lane < 16) {
            int on = n_start + sub * 16 + lane;
            if (on < N) D[on] = hip_bfloat16(c[0]);
        }
    }
}

extern "C" void launch_f4gemm_m1(
    void* D, void* A, void* B, void* ScaleA, void* ScaleB,
    int M, int N, int K,
    int stride_A, int stride_B, int stride_D,
    int stride_SA, int stride_SB
) {
    dim3 grid((N + 127) / 128, 1, 1);
    dim3 block(64, 1, 1);
    hipLaunchKernelGGL(f4gemm_m1_v20, grid, block, 0, 0,
        (hip_bfloat16*)D, (uint8_t*)A, (uint8_t*)B,
        (uint8_t*)ScaleA, (uint8_t*)ScaleB,
        M, N, K, stride_A, stride_B, stride_D, stride_SA, stride_SB);
    (void)hipDeviceSynchronize();
}
