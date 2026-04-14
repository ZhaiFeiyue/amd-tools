// FP4 GEMM M=1 v11: Preshuffled B data + shuffled scales
// Same input format as CK/ASM for apples-to-apples comparison
//
// B data: shuffle_weight(w, layout=(16,16))
//   Address: s_row = n_start + (k_iter*4 + k_row_group)*2 + m_index/8
//            s_col = (m_index%8)*16 + dk
//   → 16 lanes read consecutive 16-byte chunks = COALESCED
//
// Scales: e8m0_shuffle() permutation
//   For original scale at (r, c):
//     i0=r/32, i1=(r%32)/16, i2=r%16, j0=c/8, j1=(c%8)/4, j2=c%4
//     off = 256*j0 + 64*j2 + 4*i2 + 2*j1 + i1
//     flat_idx = i0 * 32 * s_n_pad + off
//   Read as ScaleX_flat[flat_idx]

#include <hip/hip_runtime.h>
#include <hip/hip_bfloat16.h>

typedef int v8i __attribute__((ext_vector_type(8)));
typedef float v4f __attribute__((ext_vector_type(4)));

__device__ __forceinline__ int shuffled_scale_offset(int r, int c, int s_n_pad) {
    int i0 = r / 32;
    int i1 = (r % 32) / 16;
    int i2 = r % 16;
    int j0 = c / 8;
    int j1 = (c % 8) / 4;
    int j2 = c % 4;
    int off = 256 * j0 + 64 * j2 + 4 * i2 + 2 * j1 + i1;
    return i0 * 32 * s_n_pad + off;
}

// Compute flat byte offset for shuffle_weight(w, layout=(16,16)) with fp4x2 (uint8)
// Original: w[n, k] where n=0..N-1, k=0..K_bytes-1
// Permuted: [N//16, K_bytes//32, 2, 16, 16] with dims (n_block, k_block, k_half, n_local, k_local)
// Flat offset in shuffled buffer:
__device__ __forceinline__ int shuffled_b_offset(int n_block, int m_index, int k_byte, int K_bytes) {
    int k_block = k_byte / 32;
    int k_rem = k_byte % 32;
    int k_half = k_rem / 16;
    int k_local = k_rem % 16;
    int n_local = m_index;
    // After permute [n_block, k_block, k_half, n_local, k_local]:
    int K_blocks = K_bytes / 32;
    int flat = n_block * (K_blocks * 2 * 16 * 16)
             + k_block * (2 * 16 * 16)
             + k_half * (16 * 16)
             + n_local * 16
             + k_local;
    return flat;
}

extern "C"
__global__ __launch_bounds__(256, 2)
void f4gemm_m1_v11(
    hip_bfloat16* __restrict__ D,
    const uint8_t* __restrict__ A,            // [M, K/2] UNSHUFFLED
    const uint8_t* __restrict__ B_shuffled,   // [N, K/2] PRESHUFFLED
    const uint8_t* __restrict__ ScaleA,       // [M, K/32] UNSHUFFLED (flat)
    const uint8_t* __restrict__ ScaleB,       // shuffled, flat byte array
    int M, int N, int K,
    int stride_A, int stride_B, int stride_D,
    int stride_SA, int stride_SB              // stride_SA = K/32, stride_SB = padded K/32
) {
    int n_tile = blockIdx.x;
    int tid = threadIdx.x;
    int wave_id = tid / 64;
    int lane = tid % 64;
    int m_index = lane % 16;
    int k_row_group = lane / 16;

    int n_start = n_tile * 16;
    if (n_start >= N) return;

    int b_n = n_start + m_index;  // original N-row for this lane
    bool valid_n = (b_n < N);

    // B shuffled addressing uses shuffled_b_offset()
    int n_block = n_start / 16;
    int K_bytes = K / 2;

    // A offset (NOT shuffled)
    int k_grp_off = (k_row_group * 32) / 2;

    int total_k = K / 128;
    int k_begin = (total_k * wave_id) / 4;
    int k_end = (total_k * (wave_id + 1)) / 4;

    v4f c = {0.0f, 0.0f, 0.0f, 0.0f};

    int ki = k_begin;
    for (; ki + 1 < k_end; ki += 2) {
        // --- Iter 0 ---
        const uint32_t* ap0 = (const uint32_t*)(A + ki * 64 + k_grp_off);
        v8i a0 = {(int)ap0[0], (int)ap0[1], (int)ap0[2], (int)ap0[3], 0, 0, 0, 0};

        // B: k_byte for this lane = ki*64 + k_row_group*16 (= start of 16-byte chunk)
        int b_k_byte0 = ki * 64 + k_row_group * 16;
        int b_off0 = shuffled_b_offset(n_block, m_index, b_k_byte0, K_bytes);
        v8i b0 = {0, 0, 0, 0, 0, 0, 0, 0};
        {
            const uint32_t* bp0 = (const uint32_t*)(B_shuffled + b_off0);
            b0[0] = bp0[0]; b0[1] = bp0[1]; b0[2] = bp0[2]; b0[3] = bp0[3];
        }

        int scale_c0 = ki * 4 + k_row_group;
        int32_t sca0 = (int32_t)ScaleA[scale_c0];
        int32_t scb0 = valid_n ? (int32_t)ScaleB[shuffled_scale_offset(b_n, scale_c0, stride_SB)] : 0;

        // --- Iter 1 ---
        const uint32_t* ap1 = (const uint32_t*)(A + (ki+1) * 64 + k_grp_off);
        v8i a1 = {(int)ap1[0], (int)ap1[1], (int)ap1[2], (int)ap1[3], 0, 0, 0, 0};

        int b_k_byte1 = (ki+1) * 64 + k_row_group * 16;
        int b_off1 = shuffled_b_offset(n_block, m_index, b_k_byte1, K_bytes);
        v8i b1 = {0, 0, 0, 0, 0, 0, 0, 0};
        {
            const uint32_t* bp1 = (const uint32_t*)(B_shuffled + b_off1);
            b1[0] = bp1[0]; b1[1] = bp1[1]; b1[2] = bp1[2]; b1[3] = bp1[3];
        }

        int scale_c1 = (ki+1) * 4 + k_row_group;
        int32_t sca1 = (int32_t)ScaleA[scale_c1];
        int32_t scb1 = valid_n ? (int32_t)ScaleB[shuffled_scale_offset(b_n, scale_c1, stride_SB)] : 0;

        c = __builtin_amdgcn_mfma_scale_f32_16x16x128_f8f6f4(
            a0, b0, c, 4, 4, 0, sca0, 0, scb0);
        c = __builtin_amdgcn_mfma_scale_f32_16x16x128_f8f6f4(
            a1, b1, c, 4, 4, 0, sca1, 0, scb1);
    }
    for (; ki < k_end; ki++) {
        const uint32_t* ap = (const uint32_t*)(A + ki * 64 + k_grp_off);
        v8i a = {(int)ap[0], (int)ap[1], (int)ap[2], (int)ap[3], 0, 0, 0, 0};

        int b_k_byte = ki * 64 + k_row_group * 16;
        int b_off = shuffled_b_offset(n_block, m_index, b_k_byte, K_bytes);
        v8i b = {0, 0, 0, 0, 0, 0, 0, 0};
        {
            const uint32_t* bp = (const uint32_t*)(B_shuffled + b_off);
            b[0] = bp[0]; b[1] = bp[1]; b[2] = bp[2]; b[3] = bp[3];
        }

        int scale_c = ki * 4 + k_row_group;
        int32_t sca = (int32_t)ScaleA[scale_c];
        int32_t scb = valid_n ? (int32_t)ScaleB[shuffled_scale_offset(b_n, scale_c, stride_SB)] : 0;

        c = __builtin_amdgcn_mfma_scale_f32_16x16x128_f8f6f4(
            a, b, c, 4, 4, 0, sca, 0, scb);
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
    hipLaunchKernelGGL(f4gemm_m1_v11, grid, block, 0, 0,
        (hip_bfloat16*)D, (uint8_t*)A, (uint8_t*)B,
        (uint8_t*)ScaleA, (uint8_t*)ScaleB,
        M, N, K, stride_A, stride_B, stride_D, stride_SA, stride_SB);
    (void)hipDeviceSynchronize();
}
