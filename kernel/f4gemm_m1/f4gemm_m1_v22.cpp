// v22: #5 + K-split — 8 MFMAs per K-iter + 4-wave intra-WG K-split + LDS reduce
// N-tile=128, 4 waves per WG, each wave does 8 MFMAs per K-iter on K/4 range
// This is v10 structure (4-wave K-split) but each wave processes 8 N-subtiles

#include <hip/hip_runtime.h>
#include <hip/hip_bfloat16.h>

typedef int v8i __attribute__((ext_vector_type(8)));
typedef float v4f __attribute__((ext_vector_type(4)));

extern "C"
__global__ __launch_bounds__(256, 1)
void f4gemm_m1_v22(
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
    int tid = threadIdx.x;
    int wave_id = tid >> 6;
    int lane = tid & 63;
    int m_index = lane & 15;
    int k_row_group = lane >> 4;
    int n_start = n_tile * 128;
    if (n_start >= N) return;

    int a_grp_off = k_row_group << 4;
    int total_k = K >> 7;
    int k_begin = (total_k * wave_id) >> 2;
    int k_end = (total_k * (wave_id + 1)) >> 2;

    // B bases for 8 subtiles
    int bn[8], bb[8], sb[8];
    for (int s = 0; s < 8; s++) {
        bn[s] = n_start + s * 16 + m_index;
        bb[s] = bn[s] * stride_B;
        sb[s] = bn[s] * stride_SB;
    }

    v4f c0={0,0,0,0}, c1={0,0,0,0}, c2={0,0,0,0}, c3={0,0,0,0};
    v4f c4={0,0,0,0}, c5={0,0,0,0}, c6={0,0,0,0}, c7={0,0,0,0};

    for (int ki = k_begin; ki < k_end; ki++) {
        int koff = ki * 64 + a_grp_off;
        int sc = ki * 4 + k_row_group;

        const uint32_t* ap = (const uint32_t*)(A + koff);
        v8i a = {(int)ap[0], (int)ap[1], (int)ap[2], (int)ap[3], 0, 0, 0, 0};
        int32_t sca = (int32_t)ScaleA[sc];

        #define MFMA_SUB(idx, acc) { \
            const uint32_t* bp = (const uint32_t*)(B + bb[idx] + koff); \
            v8i bv = {(int)bp[0], (int)bp[1], (int)bp[2], (int)bp[3], 0, 0, 0, 0}; \
            int32_t scb = (int32_t)ScaleB[sb[idx] + sc]; \
            acc = __builtin_amdgcn_mfma_scale_f32_16x16x128_f8f6f4(a, bv, acc, 4, 4, 0, sca, 0, scb); \
        }

        MFMA_SUB(0, c0) MFMA_SUB(1, c1) MFMA_SUB(2, c2) MFMA_SUB(3, c3)
        MFMA_SUB(4, c4) MFMA_SUB(5, c5) MFMA_SUB(6, c6) MFMA_SUB(7, c7)

        #undef MFMA_SUB
    }

    // LDS reduce across 4 waves, per subtile
    __shared__ float lds[8][4 * 16];  // [8 subtiles][4 waves × 16 lanes]

    if (lane < 16) {
        lds[0][wave_id * 16 + lane] = c0[0];
        lds[1][wave_id * 16 + lane] = c1[0];
        lds[2][wave_id * 16 + lane] = c2[0];
        lds[3][wave_id * 16 + lane] = c3[0];
        lds[4][wave_id * 16 + lane] = c4[0];
        lds[5][wave_id * 16 + lane] = c5[0];
        lds[6][wave_id * 16 + lane] = c6[0];
        lds[7][wave_id * 16 + lane] = c7[0];
    }
    __syncthreads();

    // Wave 0 reduces and writes
    if (wave_id == 0 && lane < 16) {
        for (int s = 0; s < 8; s++) {
            float sum = lds[s][lane] + lds[s][16+lane] + lds[s][32+lane] + lds[s][48+lane];
            int on = n_start + s * 16 + lane;
            if (on < N) D[on] = hip_bfloat16(sum);
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
    dim3 block(256, 1, 1);
    hipLaunchKernelGGL(f4gemm_m1_v22, grid, block, 0, 0,
        (hip_bfloat16*)D, (uint8_t*)A, (uint8_t*)B,
        (uint8_t*)ScaleA, (uint8_t*)ScaleB,
        M, N, K, stride_A, stride_B, stride_D, stride_SA, stride_SB);
    (void)hipDeviceSynchronize();
}
