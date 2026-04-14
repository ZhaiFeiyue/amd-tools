// FP4 GEMM M=1 v16: CK-style LDS staging + large N-tile (128) + multi-MFMA per load
//
// Key insight from profiling CK:
//   CK loads B[128×128_fp4] into 16KB LDS once,
//   then does 8 MFMAs from LDS (each covers 16 N-columns),
//   achieving 8x data reuse per VMEM load.
//
// Our design:
//   - 256 threads (4 waves), N-tile = 128
//   - Each K-iter: cooperatively load B[128_N × 64_bytes] = 8192 bytes into LDS
//   - Then each wave reads its 2 MFMA tiles from LDS (fast ~20 cycle latency)
//   - 4 waves × 2 tiles = 8 MFMAs per K-iter
//   - A is broadcast (16 bytes per wave, same data for all 8 MFMAs)
//   - Preshuffled B: cooperative load is naturally coalesced

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
void f4gemm_m1_v16(
    hip_bfloat16* __restrict__ D,
    const uint8_t* __restrict__ A,
    const uint8_t* __restrict__ B_shuffled,
    const uint8_t* __restrict__ ScaleA,
    const uint8_t* __restrict__ ScaleB,
    int M, int N, int K,
    int stride_A, int stride_B, int stride_D,
    int stride_SA, int stride_SB
) {
    int n_tile = blockIdx.x;   // which 128-col tile
    int tid = threadIdx.x;     // 0..255
    int wave_id = tid / 64;    // 0..3
    int lane = tid % 64;
    int m_index = lane & 15;
    int k_row_group = lane >> 4;

    int n_start = n_tile * 128;
    if (n_start >= N) return;

    // LDS: B data for one K-iter = 128 N-rows × 64 bytes = 8192 bytes
    // Stored as flat bytes in LDS
    __shared__ uint8_t lds_B[8192];

    // Each wave handles 2 N-subtiles of 16 columns: wave_id*2 and wave_id*2+1
    int n_sub0 = wave_id * 2;       // 0,2,4,6
    int n_sub1 = wave_id * 2 + 1;   // 1,3,5,7
    int b_n0 = n_start + n_sub0 * 16 + m_index;
    int b_n1 = n_start + n_sub1 * 16 + m_index;
    bool valid_n0 = (b_n0 < N);
    bool valid_n1 = (b_n1 < N);

    // Precompute shuffled B address base for cooperative loading
    int n_block = n_start >> 4;  // which 16-row block (for shuffle addressing)
    int K_bytes = K >> 1;
    int K_blocks = K_bytes >> 5;
    int b_nblock_stride = K_blocks << 9;  // K_blocks * 512

    // Accumulators: 2 tiles per wave
    v4f c0 = {0.0f, 0.0f, 0.0f, 0.0f};
    v4f c1 = {0.0f, 0.0f, 0.0f, 0.0f};

    int a_grp_off = k_row_group << 4;
    int k_iters = K >> 7;  // K / 128

    for (int ki = 0; ki < k_iters; ki++) {
        // =====================================================
        // Step 1: Cooperative load B[128_N × 64_bytes] into LDS
        // 8192 bytes / 256 threads = 32 bytes per thread = 8 dwords
        // =====================================================
        {
            // Each thread loads 32 bytes (8 dwords) of shuffled B
            // Thread tid → LDS offset tid * 32
            // B source: shuffled layout, we load contiguous chunks
            // For shuffled B with N-tile of 128: there are 8 N-blocks of 16 rows
            // Each N-block × K-iter has a specific shuffled offset

            // Simple approach: load B_shuffled linearly from the right location
            // The shuffled B for N-tile n_start, K-iter ki spans 8 N-blocks × 1 K-group
            // Each (n_block, k_block, k_half) chunk = 256 bytes (16 n_local × 16 k_local)
            // Per K-iter (128 fp4 = 64 bytes = 4 k_groups): 8 n_blocks × 4 k_groups = 32 chunks
            // 32 chunks × 256 bytes = 8192 bytes ✓

            // Shuffled flat address for chunk (n_blk_local, k_group):
            //   base = (n_start/16 + n_blk_local) * b_nblock_stride
            //        + (ki*4 + k_group) * 512 (= k_block * 512 + k_half * 256 combined)
            //   But k_group 0..3 maps to:
            //     k_group 0 → k_block=ki*2+(0>>1)=ki*2, k_half=0 → offset ki*2*512 + 0 = ki*1024
            //     k_group 1 → k_block=ki*2+0, k_half=1 → offset ki*1024 + 256
            //     k_group 2 → k_block=ki*2+1, k_half=0 → offset ki*1024 + 512
            //     k_group 3 → k_block=ki*2+1, k_half=1 → offset ki*1024 + 768

            // Total per n_block per K-iter: 4 groups × 256 bytes = 1024 bytes
            // 8 n_blocks × 1024 = 8192 bytes

            // Thread tid loads bytes [tid*32 .. tid*32+31]
            // Map to (n_blk_local, k_group, n_local, k_local):
            int lds_off = tid * 32;
            int n_blk_local = lds_off / 1024;        // 0..7
            int within_nblk = lds_off % 1024;
            int k_group = within_nblk / 256;          // 0..3
            int within_chunk = within_nblk % 256;
            // within_chunk = n_local * 16 + k_local, load 32 bytes = 2 rows of 16 bytes

            // Shuffled B source offset
            int src_nblock = n_block + n_blk_local;  // actual n_block in shuffled B
            // k_group → (k_block_offset, k_half):
            //   k_group 0 → ki*1024 + 0
            //   k_group 1 → ki*1024 + 256
            //   k_group 2 → ki*1024 + 512
            //   k_group 3 → ki*1024 + 768
            int b_src = src_nblock * b_nblock_stride + ki * 1024 + k_group * 256 + within_chunk;

            // Load 32 bytes (8 dwords) from global to LDS
            const uint32_t* src_ptr = (const uint32_t*)(B_shuffled + b_src);
            uint32_t* dst_ptr = (uint32_t*)(lds_B + lds_off);
            dst_ptr[0] = src_ptr[0]; dst_ptr[1] = src_ptr[1];
            dst_ptr[2] = src_ptr[2]; dst_ptr[3] = src_ptr[3];
            dst_ptr[4] = src_ptr[4]; dst_ptr[5] = src_ptr[5];
            dst_ptr[6] = src_ptr[6]; dst_ptr[7] = src_ptr[7];
        }
        __syncthreads();

        // =====================================================
        // Step 2: Load A (same for all 8 MFMAs, broadcast)
        // =====================================================
        int a_off = ki * 64 + a_grp_off;
        const uint32_t* ap = (const uint32_t*)(A + a_off);
        v8i a_data = {(int)ap[0], (int)ap[1], (int)ap[2], (int)ap[3], 0, 0, 0, 0};

        // =====================================================
        // Step 3: Each wave reads 2 tiles from LDS and does 2 MFMAs
        // =====================================================
        // N-subtile s: B data at LDS offset s*1024 (1024 bytes per 16-N-row subtile)
        // Within subtile: lane needs 16 bytes at offset depending on (m_index, k_row_group)
        // LDS layout matches shuffled B: [n_local=16][k_local=16] per (k_group, n_blk_local)
        // For MFMA lane (m_index, k_row_group):
        //   LDS offset = n_sub * 1024 + k_row_group * 256 + m_index * 16

        // Tile 0
        {
            int lds_off0 = n_sub0 * 1024 + k_row_group * 256 + m_index * 16;
            const uint32_t* lp0 = (const uint32_t*)(lds_B + lds_off0);
            v8i b0 = {(int)lp0[0], (int)lp0[1], (int)lp0[2], (int)lp0[3], 0, 0, 0, 0};

            int sc0 = ki * 4 + k_row_group;
            int32_t sca0 = (int32_t)ScaleA[sc0];
            int32_t scb0 = valid_n0 ? (int32_t)ScaleB[shuffled_scale_offset(b_n0, sc0, stride_SB)] : 0;

            c0 = __builtin_amdgcn_mfma_scale_f32_16x16x128_f8f6f4(
                a_data, b0, c0, 4, 4, 0, sca0, 0, scb0);
        }

        // Tile 1
        {
            int lds_off1 = n_sub1 * 1024 + k_row_group * 256 + m_index * 16;
            const uint32_t* lp1 = (const uint32_t*)(lds_B + lds_off1);
            v8i b1 = {(int)lp1[0], (int)lp1[1], (int)lp1[2], (int)lp1[3], 0, 0, 0, 0};

            int sc1 = ki * 4 + k_row_group;
            int32_t sca1 = (int32_t)ScaleA[sc1];
            int32_t scb1 = valid_n1 ? (int32_t)ScaleB[shuffled_scale_offset(b_n1, sc1, stride_SB)] : 0;

            c1 = __builtin_amdgcn_mfma_scale_f32_16x16x128_f8f6f4(
                a_data, b1, c1, 4, 4, 0, sca1, 0, scb1);
        }

        __syncthreads();  // ensure LDS can be reused
    }

    // =====================================================
    // Write output: each wave writes 2 × 16 = 32 N-columns
    // =====================================================
    if (lane < 16) {
        int out_n0 = n_start + n_sub0 * 16 + lane;
        if (out_n0 < N) D[out_n0] = hip_bfloat16(c0[0]);

        int out_n1 = n_start + n_sub1 * 16 + lane;
        if (out_n1 < N) D[out_n1] = hip_bfloat16(c1[0]);
    }
}

extern "C" void launch_f4gemm_m1(
    void* D, void* A, void* B, void* ScaleA, void* ScaleB,
    int M, int N, int K,
    int stride_A, int stride_B, int stride_D,
    int stride_SA, int stride_SB
) {
    dim3 grid((N + 127) / 128, 1, 1);  // N/128 workgroups
    dim3 block(256, 1, 1);
    hipLaunchKernelGGL(f4gemm_m1_v16, grid, block, 0, 0,
        (hip_bfloat16*)D, (uint8_t*)A, (uint8_t*)B,
        (uint8_t*)ScaleA, (uint8_t*)ScaleB,
        M, N, K, stride_A, stride_B, stride_D, stride_SA, stride_SB);
    (void)hipDeviceSynchronize();
}
