// FP4 GEMM M=1 v17: LDS with simplified linear addressing
// LDS layout: B_lds[128][64] = 8192 bytes, simple row-major
// Cooperative load: thread tid loads 32 contiguous bytes from shuffled B
// No complex address math — just linear offset + stride

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
__global__ __launch_bounds__(256, 1)
void f4gemm_m1_v17(
    hip_bfloat16* __restrict__ D,
    const uint8_t* __restrict__ A,
    const uint8_t* __restrict__ B,       // UNSHUFFLED [N, K/2]
    const uint8_t* __restrict__ ScaleA,  // unshuffled
    const uint8_t* __restrict__ ScaleB,  // shuffled
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

    // LDS: B[128][64] row-major. Each N-row has 64 bytes (128 fp4 per K-iter)
    __shared__ uint8_t lds_B[128 * 64];

    // Each wave handles 2 N-subtiles of 16
    int n_sub0 = wave_id * 2;
    int n_sub1 = wave_id * 2 + 1;
    int b_n0 = n_start + n_sub0 * 16 + m_index;
    int b_n1 = n_start + n_sub1 * 16 + m_index;
    bool vn0 = (b_n0 < N);
    bool vn1 = (b_n1 < N);

    v4f c0 = {0.0f, 0.0f, 0.0f, 0.0f};
    v4f c1 = {0.0f, 0.0f, 0.0f, 0.0f};

    int a_grp_off = k_row_group << 4;
    int k_iters = K >> 7;

    // Precompute cooperative load parameters
    // 8192 bytes / 256 threads = 32 bytes each = 2 rows of 16 bytes
    // tid → loads 2 chunks of 16 bytes:
    //   chunk 0: lds_B[row0][col0..col0+15]
    //   chunk 1: lds_B[row1][col1..col1+15]
    // Simple mapping: tid*32 = flat LDS offset
    //   row = (tid*32) / 64 = tid / 2
    //   col = (tid*32) % 64 = (tid & 1) * 32
    int load_n0 = tid >> 1;           // 0..127
    int load_k_off0 = (tid & 1) << 5; // 0 or 32

    for (int ki = 0; ki < k_iters; ki++) {
        int k_byte_base = ki * 64;

        // Cooperative load: each thread loads 32 bytes of B into LDS
        // B is UNSHUFFLED [N, K/2] row-major
        {
            int src_n = n_start + load_n0;
            if (src_n < N) {
                const uint32_t* src = (const uint32_t*)(B + src_n * stride_B + k_byte_base + load_k_off0);
                uint32_t* dst = (uint32_t*)(lds_B + load_n0 * 64 + load_k_off0);
                dst[0] = src[0]; dst[1] = src[1]; dst[2] = src[2]; dst[3] = src[3];
                dst[4] = src[4]; dst[5] = src[5]; dst[6] = src[6]; dst[7] = src[7];
            } else {
                uint32_t* dst = (uint32_t*)(lds_B + load_n0 * 64 + load_k_off0);
                dst[0] = 0; dst[1] = 0; dst[2] = 0; dst[3] = 0;
                dst[4] = 0; dst[5] = 0; dst[6] = 0; dst[7] = 0;
            }
        }
        __syncthreads();

        // Load A (broadcast, not shuffled)
        const uint32_t* ap = (const uint32_t*)(A + ki * 64 + a_grp_off);
        v8i a_data = {(int)ap[0], (int)ap[1], (int)ap[2], (int)ap[3], 0, 0, 0, 0};

        // Read B from LDS for each subtile
        // Subtile s: N-rows [s*16 .. s*16+15], K-bytes [k_row_group*16 .. +15]
        // LDS offset = (n_sub*16 + m_index) * 64 + k_row_group * 16
        {
            int lds_off0 = (n_sub0 * 16 + m_index) * 64 + (k_row_group << 4);
            const uint32_t* lp0 = (const uint32_t*)(lds_B + lds_off0);
            v8i b0 = {(int)lp0[0], (int)lp0[1], (int)lp0[2], (int)lp0[3], 0, 0, 0, 0};

            int sc = ki * 4 + k_row_group;
            int32_t sca = (int32_t)ScaleA[sc];
            int32_t scb0 = vn0 ? (int32_t)ScaleB[b_n0 * stride_SB + sc] : 0;

            c0 = __builtin_amdgcn_mfma_scale_f32_16x16x128_f8f6f4(
                a_data, b0, c0, 4, 4, 0, sca, 0, scb0);
        }
        {
            int lds_off1 = (n_sub1 * 16 + m_index) * 64 + (k_row_group << 4);
            const uint32_t* lp1 = (const uint32_t*)(lds_B + lds_off1);
            v8i b1 = {(int)lp1[0], (int)lp1[1], (int)lp1[2], (int)lp1[3], 0, 0, 0, 0};

            int sc = ki * 4 + k_row_group;
            int32_t scb1 = vn1 ? (int32_t)ScaleB[b_n1 * stride_SB + sc] : 0;
            int32_t sca = (int32_t)ScaleA[sc];

            c1 = __builtin_amdgcn_mfma_scale_f32_16x16x128_f8f6f4(
                a_data, b1, c1, 4, 4, 0, sca, 0, scb1);
        }

        __syncthreads();
    }

    // Output
    if (lane < 16) {
        int on0 = n_start + n_sub0 * 16 + lane;
        if (on0 < N) D[on0] = hip_bfloat16(c0[0]);
        int on1 = n_start + n_sub1 * 16 + lane;
        if (on1 < N) D[on1] = hip_bfloat16(c1[0]);
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
    hipLaunchKernelGGL(f4gemm_m1_v17, grid, block, 0, 0,
        (hip_bfloat16*)D, (uint8_t*)A, (uint8_t*)B,
        (uint8_t*)ScaleA, (uint8_t*)ScaleB,
        M, N, K, stride_A, stride_B, stride_D, stride_SA, stride_SB);
    (void)hipDeviceSynchronize();
}
