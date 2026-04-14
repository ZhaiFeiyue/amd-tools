// v23: Direct-to-LDS load via __builtin_amdgcn_global_load_lds
//
// IMPORTANT: global_load_lds_dwordx4 uses m0 as LDS BASE offset.
// Each lane in a wave writes at m0 + lane * 16 bytes.
// So one call loads 64 lanes × 16 bytes = 1024 bytes per wave.
// With 4 waves and 2 calls each: 4 × 2 × 1024 = 8192 bytes total.
//
// LDS layout: B[128 N-rows][64 K-bytes] = 8192 bytes, row-major.
// Each row = 64 bytes = 4 dwordx4.
// 8192 / 1024 = 8 "strips" of 1024 bytes.
// Wave w, call c: loads strip (w*2+c), which covers rows [strip*16 .. strip*16+15]
//   at columns [lane*... no, that's wrong. Let me think again.]
//
// Actually: each lane provides a VGPR address for the global source.
// The LDS destination is m0 + lane * 16.
// So lane 0 → LDS[m0], lane 1 → LDS[m0+16], ..., lane 63 → LDS[m0+63*16]
//
// We want LDS[row][col], row-major, 64 bytes per row.
// 1024 bytes = 16 rows × 64 bytes.
// Lane l → row = l / 4, col = (l % 4) * 16
// LDS offset = row * 64 + col = (l/4)*64 + (l%4)*16
// But global_load_lds gives LDS offset = m0 + l * 16 (linear!), not row-major.
//
// Solution: set m0 for each strip, and have each lane load the right B address.
// Strip s (0..7): covers LDS bytes [s*1024 .. s*1024+1023]
//   = rows [s*16 .. s*16+15], all 64 K-bytes
// In linear LDS: LDS[s*1024 + l*16]
//   = LDS row = s*16 + l/4, col = (l%4)*16
// Lane l's global source: B[(n_start + s*16 + l/4) * stride_B + k_byte + (l%4)*16]
//
// So for each strip: m0 = s * 1024
// Lane l loads from: B_src = B + (n_start + s*16 + l/4)*stride_B + k_byte + (l%4)*16
// Each wave does 2 strips. Wave w does strips 2w and 2w+1.

#include <hip/hip_runtime.h>
#include <hip/hip_bfloat16.h>

typedef int v8i __attribute__((ext_vector_type(8)));
typedef float v4f __attribute__((ext_vector_type(4)));

extern "C"
__global__ __launch_bounds__(256, 1)
void f4gemm_m1_v23(
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

    __shared__ uint8_t lds_B[8192];  // [128][64] row-major = linear 8192 bytes

    int n_sub0 = wave_id * 2;
    int n_sub1 = wave_id * 2 + 1;
    int bn0 = n_start + n_sub0 * 16 + m_index;
    int bn1 = n_start + n_sub1 * 16 + m_index;
    bool vn0 = (bn0 < N), vn1 = (bn1 < N);

    v4f c0 = {0,0,0,0}, c1 = {0,0,0,0};

    int a_grp_off = k_row_group << 4;
    int k_iters = K >> 7;

    // Precompute per-lane load parameters for cooperative load
    // Lane l in a wave: loads row_in_strip = l/4, col_in_strip = (l%4)*16
    int lane_row = lane >> 2;     // 0..15
    int lane_col = (lane & 3) << 4; // 0, 16, 32, 48

    // Each wave does 2 strips. Wave w → strips 2w and 2w+1
    int strip0 = wave_id * 2;
    int strip1 = wave_id * 2 + 1;

    // Global source row for strip s, this lane:
    // n_global = n_start + s*16 + lane_row
    // B offset = n_global * stride_B + k_byte + lane_col
    int src_n0 = n_start + strip0 * 16 + lane_row;
    int src_n1 = n_start + strip1 * 16 + lane_row;
    int src_base0 = src_n0 * stride_B;
    int src_base1 = src_n1 * stride_B;

    // LDS offset for strip s: m0 = s * 1024
    // But we need to tell the compiler the LDS address.
    // global_load_lds dst must be lds_B + strip*1024 (the base for this strip)
    // The builtin will set m0 = &lds_B[strip*1024] (relative to LDS start)

    for (int ki = 0; ki < k_iters; ki++) {
        int k_byte = ki * 64;

        // Step 1: Each wave loads 2 strips (2 × 1024 = 2048 bytes)
        // Strip 0 for this wave
        {
            const uint8_t* src = B + src_base0 + k_byte + lane_col;
            uint8_t* dst = lds_B + strip0 * 1024;
            if (src_n0 < N) {
                __builtin_amdgcn_global_load_lds(src, dst, 16, 0, 0);
            }
        }
        // Strip 1 for this wave
        {
            const uint8_t* src = B + src_base1 + k_byte + lane_col;
            uint8_t* dst = lds_B + strip1 * 1024;
            if (src_n1 < N) {
                __builtin_amdgcn_global_load_lds(src, dst, 16, 0, 0);
            }
        }
        __syncthreads();

        // Step 2: Load A
        const uint32_t* ap = (const uint32_t*)(A + k_byte + a_grp_off);
        v8i a = {(int)ap[0], (int)ap[1], (int)ap[2], (int)ap[3], 0, 0, 0, 0};
        int sc = ki * 4 + k_row_group;
        int32_t sca = (int32_t)ScaleA[sc];

        // Step 3: Read from LDS and execute MFMAs
        // LDS is linear [8192 bytes]. Row r, col c → offset r*64+c.
        // But we loaded linearly: strip s at offset s*1024, lane l at offset l*16.
        // So LDS[s*1024 + l*16] = B data for (strip_row=s*16+l/4, strip_col=(l%4)*16)
        // = B[n_start + s*16 + l/4, k_byte + (l%4)*16]
        //
        // For MFMA read: need B[n_sub*16 + m_index, k_byte + k_row_group*16]
        // = B at row (n_sub*16 + m_index), col (k_row_group*16)
        // In our LDS linear layout: this is at strip n_sub, 
        //   lane_in_strip = m_index * 4 + k_row_group
        // LDS offset = n_sub * 1024 + (m_index * 4 + k_row_group) * 16
        {
            int lds_off0 = n_sub0 * 1024 + (m_index * 4 + k_row_group) * 16;
            const uint32_t* lp0 = (const uint32_t*)(lds_B + lds_off0);
            v8i b0 = {(int)lp0[0], (int)lp0[1], (int)lp0[2], (int)lp0[3], 0, 0, 0, 0};
            int32_t scb0 = vn0 ? (int32_t)ScaleB[bn0 * stride_SB + sc] : 0;
            c0 = __builtin_amdgcn_mfma_scale_f32_16x16x128_f8f6f4(a, b0, c0, 4, 4, 0, sca, 0, scb0);
        }
        {
            int lds_off1 = n_sub1 * 1024 + (m_index * 4 + k_row_group) * 16;
            const uint32_t* lp1 = (const uint32_t*)(lds_B + lds_off1);
            v8i b1 = {(int)lp1[0], (int)lp1[1], (int)lp1[2], (int)lp1[3], 0, 0, 0, 0};
            int32_t scb1 = vn1 ? (int32_t)ScaleB[bn1 * stride_SB + sc] : 0;
            c1 = __builtin_amdgcn_mfma_scale_f32_16x16x128_f8f6f4(a, b1, c1, 4, 4, 0, sca, 0, scb1);
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
    hipLaunchKernelGGL(f4gemm_m1_v23, grid, block, 0, 0,
        (hip_bfloat16*)D, (uint8_t*)A, (uint8_t*)B,
        (uint8_t*)ScaleA, (uint8_t*)ScaleB,
        M, N, K, stride_A, stride_B, stride_D, stride_SA, stride_SB);
    (void)hipDeviceSynchronize();
}
