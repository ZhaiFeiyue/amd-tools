// FP4 GEMM M=1 v19: Measure CK optimizations independently
// Approach: single wave per WG, N-tile=128, 8 MFMAs per K-iter
// Each wave loads A once, then loads 8 different B subtiles and does 8 MFMAs
// No LDS, no barriers — rely on L1/L2 cache for B data locality
// K-split across workgroups via grid_y for parallelism

#include <hip/hip_runtime.h>
#include <hip/hip_bfloat16.h>

typedef int v8i __attribute__((ext_vector_type(8)));
typedef float v4f __attribute__((ext_vector_type(4)));

extern "C"
__global__ __launch_bounds__(64, 4)
void f4gemm_m1_v19(
    hip_bfloat16* __restrict__ D,
    const uint8_t* __restrict__ A,
    const uint8_t* __restrict__ B,
    const uint8_t* __restrict__ ScaleA,
    const uint8_t* __restrict__ ScaleB,
    int M, int N, int K,
    int stride_A, int stride_B, int stride_D,
    int stride_SA, int stride_SB
) {
    int n_tile = blockIdx.x;     // which 128-col block
    int k_split = blockIdx.y;    // K-split partition (0..3)
    int lane = threadIdx.x;      // 0..63
    int m_index = lane & 15;
    int k_row_group = lane >> 4;

    int n_start = n_tile * 128;
    if (n_start >= N) return;

    // 8 accumulators for 8 N-subtiles of 16 columns each
    v4f c0={0,0,0,0}, c1={0,0,0,0}, c2={0,0,0,0}, c3={0,0,0,0};
    v4f c4={0,0,0,0}, c5={0,0,0,0}, c6={0,0,0,0}, c7={0,0,0,0};

    int a_grp_off = k_row_group << 4;
    int total_k = K >> 7;
    int k_begin = (total_k * k_split) >> 2;  // 4-way K-split
    int k_end = (total_k * (k_split + 1)) >> 2;

    // Precompute B base offsets for 8 subtiles
    int bn[8];
    bool vn[8];
    int sb_base[8];
    for (int s = 0; s < 8; s++) {
        bn[s] = n_start + s * 16 + m_index;
        vn[s] = (bn[s] < N);
        sb_base[s] = bn[s] * stride_SB;
    }

    for (int ki = k_begin; ki < k_end; ki++) {
        // Load A once (shared across all 8 MFMAs)
        const uint32_t* ap = (const uint32_t*)(A + ki * 64 + a_grp_off);
        v8i a_data = {(int)ap[0], (int)ap[1], (int)ap[2], (int)ap[3], 0, 0, 0, 0};

        int sc = ki * 4 + k_row_group;
        int32_t sca = (int32_t)ScaleA[sc];
        int k_byte = ki * 64 + (k_row_group << 4);

        // 8 MFMAs, each loading B from a different N-subtile
        // A is reused 8 times!
        #define DO_MFMA(idx, acc) { \
            v8i b = {0,0,0,0,0,0,0,0}; \
            if (vn[idx]) { \
                const uint32_t* bp = (const uint32_t*)(B + bn[idx] * stride_B + k_byte); \
                b[0] = bp[0]; b[1] = bp[1]; b[2] = bp[2]; b[3] = bp[3]; \
            } \
            int32_t scb = vn[idx] ? (int32_t)ScaleB[sb_base[idx] + sc] : 0; \
            acc = __builtin_amdgcn_mfma_scale_f32_16x16x128_f8f6f4( \
                a_data, b, acc, 4, 4, 0, sca, 0, scb); \
        }

        DO_MFMA(0, c0)
        DO_MFMA(1, c1)
        DO_MFMA(2, c2)
        DO_MFMA(3, c3)
        DO_MFMA(4, c4)
        DO_MFMA(5, c5)
        DO_MFMA(6, c6)
        DO_MFMA(7, c7)

        #undef DO_MFMA
    }

    // Write output: atomicAdd for K-split reduction
    if (lane < 16) {
        #define WRITE_OUT(idx, acc) { \
            int on = n_start + idx * 16 + lane; \
            if (on < N) { \
                if (gridDim.y == 1) \
                    D[on] = hip_bfloat16(acc[0]); \
                else { \
                    /* Need float atomicAdd — use workspace */ \
                } \
            } \
        }
        // For simplicity, only handle k_split=0 writes (single-pass mode)
        if (k_split == 0 && gridDim.y == 1) {
            if (n_start + 0*16 + lane < N) D[n_start + 0*16 + lane] = hip_bfloat16(c0[0]);
            if (n_start + 1*16 + lane < N) D[n_start + 1*16 + lane] = hip_bfloat16(c1[0]);
            if (n_start + 2*16 + lane < N) D[n_start + 2*16 + lane] = hip_bfloat16(c2[0]);
            if (n_start + 3*16 + lane < N) D[n_start + 3*16 + lane] = hip_bfloat16(c3[0]);
            if (n_start + 4*16 + lane < N) D[n_start + 4*16 + lane] = hip_bfloat16(c4[0]);
            if (n_start + 5*16 + lane < N) D[n_start + 5*16 + lane] = hip_bfloat16(c5[0]);
            if (n_start + 6*16 + lane < N) D[n_start + 6*16 + lane] = hip_bfloat16(c6[0]);
            if (n_start + 7*16 + lane < N) D[n_start + 7*16 + lane] = hip_bfloat16(c7[0]);
        }
        #undef WRITE_OUT
    }
}

extern "C" void launch_f4gemm_m1(
    void* D, void* A, void* B, void* ScaleA, void* ScaleB,
    int M, int N, int K,
    int stride_A, int stride_B, int stride_D,
    int stride_SA, int stride_SB
) {
    // No K-split for now (single pass)
    dim3 grid((N + 127) / 128, 1, 1);
    dim3 block(64, 1, 1);  // 1 wave per WG
    hipLaunchKernelGGL(f4gemm_m1_v19, grid, block, 0, 0,
        (hip_bfloat16*)D, (uint8_t*)A, (uint8_t*)B,
        (uint8_t*)ScaleA, (uint8_t*)ScaleB,
        M, N, K, stride_A, stride_B, stride_D, stride_SA, stride_SB);
    (void)hipDeviceSynchronize();
}
