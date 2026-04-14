// FP4 GEMM M=1 v18: LDS staging + N-tile=128 + K-split=4
// Grid: (N/128, 4, 1), each WG processes 128 N-cols × K/4 K-elements
// AtomicAdd to float32 workspace, then convert to bf16

#include <hip/hip_runtime.h>
#include <hip/hip_bfloat16.h>

typedef int v8i __attribute__((ext_vector_type(8)));
typedef float v4f __attribute__((ext_vector_type(4)));

extern "C"
__global__ __launch_bounds__(256, 1)
void f4gemm_m1_v18(
    float* __restrict__ D_f32,       // [N] float32 accumulator
    const uint8_t* __restrict__ A,
    const uint8_t* __restrict__ B,   // UNSHUFFLED [N, K/2]
    const uint8_t* __restrict__ ScaleA,
    const uint8_t* __restrict__ ScaleB,
    int M, int N, int K,
    int stride_A, int stride_B, int stride_D,
    int stride_SA, int stride_SB
) {
    int n_tile = blockIdx.x;
    int k_split = blockIdx.y;
    int tid = threadIdx.x;
    int wave_id = tid >> 6;
    int lane = tid & 63;
    int m_index = lane & 15;
    int k_row_group = lane >> 4;

    int n_start = n_tile * 128;
    if (n_start >= N) return;

    __shared__ uint8_t lds_B[128 * 64];  // 8KB

    int n_sub0 = wave_id * 2;
    int n_sub1 = wave_id * 2 + 1;
    int b_n0 = n_start + n_sub0 * 16 + m_index;
    int b_n1 = n_start + n_sub1 * 16 + m_index;
    bool vn0 = (b_n0 < N);
    bool vn1 = (b_n1 < N);

    v4f c0 = {0.0f, 0.0f, 0.0f, 0.0f};
    v4f c1 = {0.0f, 0.0f, 0.0f, 0.0f};

    int a_grp_off = k_row_group << 4;
    int total_k = K >> 7;
    int k_begin = (total_k * k_split) >> 2;
    int k_end = (total_k * (k_split + 1)) >> 2;

    int load_n0 = tid >> 1;
    int load_k_off0 = (tid & 1) << 5;

    for (int ki = k_begin; ki < k_end; ki++) {
        int k_byte_base = ki * 64;

        // Cooperative load B into LDS
        {
            int src_n = n_start + load_n0;
            uint32_t* dst = (uint32_t*)(lds_B + load_n0 * 64 + load_k_off0);
            if (src_n < N) {
                const uint32_t* src = (const uint32_t*)(B + src_n * stride_B + k_byte_base + load_k_off0);
                dst[0] = src[0]; dst[1] = src[1]; dst[2] = src[2]; dst[3] = src[3];
                dst[4] = src[4]; dst[5] = src[5]; dst[6] = src[6]; dst[7] = src[7];
            } else {
                dst[0] = 0; dst[1] = 0; dst[2] = 0; dst[3] = 0;
                dst[4] = 0; dst[5] = 0; dst[6] = 0; dst[7] = 0;
            }
        }
        __syncthreads();

        const uint32_t* ap = (const uint32_t*)(A + ki * 64 + a_grp_off);
        v8i a_data = {(int)ap[0], (int)ap[1], (int)ap[2], (int)ap[3], 0, 0, 0, 0};

        int sc = ki * 4 + k_row_group;
        int32_t sca = (int32_t)ScaleA[sc];

        // Tile 0
        {
            const uint32_t* lp = (const uint32_t*)(lds_B + (n_sub0 * 16 + m_index) * 64 + (k_row_group << 4));
            v8i b0 = {(int)lp[0], (int)lp[1], (int)lp[2], (int)lp[3], 0, 0, 0, 0};
            int32_t scb = vn0 ? (int32_t)ScaleB[b_n0 * stride_SB + sc] : 0;
            c0 = __builtin_amdgcn_mfma_scale_f32_16x16x128_f8f6f4(a_data, b0, c0, 4, 4, 0, sca, 0, scb);
        }
        // Tile 1
        {
            const uint32_t* lp = (const uint32_t*)(lds_B + (n_sub1 * 16 + m_index) * 64 + (k_row_group << 4));
            v8i b1 = {(int)lp[0], (int)lp[1], (int)lp[2], (int)lp[3], 0, 0, 0, 0};
            int32_t scb = vn1 ? (int32_t)ScaleB[b_n1 * stride_SB + sc] : 0;
            c1 = __builtin_amdgcn_mfma_scale_f32_16x16x128_f8f6f4(a_data, b1, c1, 4, 4, 0, sca, 0, scb);
        }

        __syncthreads();
    }

    // AtomicAdd partial results
    if (lane < 16) {
        int on0 = n_start + n_sub0 * 16 + lane;
        if (on0 < N) atomicAdd(&D_f32[on0], c0[0]);
        int on1 = n_start + n_sub1 * 16 + lane;
        if (on1 < N) atomicAdd(&D_f32[on1], c1[0]);
    }
}

extern "C"
__global__ void cvt_f32_bf16(hip_bfloat16* D, const float* S, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) D[i] = hip_bfloat16(S[i]);
}

static float* g_f32 = nullptr;
static int g_f32_n = 0;

extern "C" void launch_f4gemm_m1(
    void* D, void* A, void* B, void* ScaleA, void* ScaleB,
    int M, int N, int K,
    int stride_A, int stride_B, int stride_D,
    int stride_SA, int stride_SB
) {
    if (g_f32_n < N) {
        if (g_f32) (void)hipFree(g_f32);
        (void)hipMalloc(&g_f32, N * sizeof(float));
        g_f32_n = N;
    }
    (void)hipMemsetAsync(g_f32, 0, N * sizeof(float), 0);

    dim3 grid((N + 127) / 128, 4, 1);  // K_SPLIT = 4
    dim3 block(256, 1, 1);
    hipLaunchKernelGGL(f4gemm_m1_v18, grid, block, 0, 0,
        g_f32, (uint8_t*)A, (uint8_t*)B,
        (uint8_t*)ScaleA, (uint8_t*)ScaleB,
        M, N, K, stride_A, stride_B, stride_D, stride_SA, stride_SB);

    dim3 cg((N + 255) / 256, 1, 1);
    hipLaunchKernelGGL(cvt_f32_bf16, cg, dim3(256), 0, 0,
        (hip_bfloat16*)D, g_f32, N);
    (void)hipDeviceSynchronize();
}
