// v21: #5 — large N-tile, 8 MFMAs per K-iter (interleaved)
// K-loop outside, 8 MFMAs inside. A loaded once, B loaded 8 times per K-iter.
// Single wave, no LDS, no K-split.

#include <hip/hip_runtime.h>
#include <hip/hip_bfloat16.h>

typedef int v8i __attribute__((ext_vector_type(8)));
typedef float v4f __attribute__((ext_vector_type(4)));

extern "C"
__global__ __launch_bounds__(64, 4)
void f4gemm_m1_v21(
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

    // Precompute B base for 8 subtiles
    int bn0 = n_start + 0*16 + m_index, bn1 = n_start + 1*16 + m_index;
    int bn2 = n_start + 2*16 + m_index, bn3 = n_start + 3*16 + m_index;
    int bn4 = n_start + 4*16 + m_index, bn5 = n_start + 5*16 + m_index;
    int bn6 = n_start + 6*16 + m_index, bn7 = n_start + 7*16 + m_index;

    int bb0 = bn0*stride_B, bb1 = bn1*stride_B, bb2 = bn2*stride_B, bb3 = bn3*stride_B;
    int bb4 = bn4*stride_B, bb5 = bn5*stride_B, bb6 = bn6*stride_B, bb7 = bn7*stride_B;

    int sb0 = bn0*stride_SB, sb1 = bn1*stride_SB, sb2 = bn2*stride_SB, sb3 = bn3*stride_SB;
    int sb4 = bn4*stride_SB, sb5 = bn5*stride_SB, sb6 = bn6*stride_SB, sb7 = bn7*stride_SB;

    v4f c0={0,0,0,0}, c1={0,0,0,0}, c2={0,0,0,0}, c3={0,0,0,0};
    v4f c4={0,0,0,0}, c5={0,0,0,0}, c6={0,0,0,0}, c7={0,0,0,0};

    for (int ki = 0; ki < k_iters; ki++) {
        int koff = ki * 64 + a_grp_off;
        int sc = ki * 4 + k_row_group;

        // Load A once
        const uint32_t* ap = (const uint32_t*)(A + koff);
        v8i a = {(int)ap[0], (int)ap[1], (int)ap[2], (int)ap[3], 0, 0, 0, 0};
        int32_t sca = (int32_t)ScaleA[sc];

        // 8 MFMAs, each loading different B subtile
        // B loads should hit L1/L2 cache for A; B itself is different each time
        const uint32_t *p0, *p1, *p2, *p3, *p4, *p5, *p6, *p7;
        p0=(const uint32_t*)(B+bb0+koff); p1=(const uint32_t*)(B+bb1+koff);
        p2=(const uint32_t*)(B+bb2+koff); p3=(const uint32_t*)(B+bb3+koff);
        p4=(const uint32_t*)(B+bb4+koff); p5=(const uint32_t*)(B+bb5+koff);
        p6=(const uint32_t*)(B+bb6+koff); p7=(const uint32_t*)(B+bb7+koff);

        v8i b0={(int)p0[0],(int)p0[1],(int)p0[2],(int)p0[3],0,0,0,0};
        v8i b1={(int)p1[0],(int)p1[1],(int)p1[2],(int)p1[3],0,0,0,0};
        v8i b2={(int)p2[0],(int)p2[1],(int)p2[2],(int)p2[3],0,0,0,0};
        v8i b3={(int)p3[0],(int)p3[1],(int)p3[2],(int)p3[3],0,0,0,0};
        v8i b4={(int)p4[0],(int)p4[1],(int)p4[2],(int)p4[3],0,0,0,0};
        v8i b5={(int)p5[0],(int)p5[1],(int)p5[2],(int)p5[3],0,0,0,0};
        v8i b6={(int)p6[0],(int)p6[1],(int)p6[2],(int)p6[3],0,0,0,0};
        v8i b7={(int)p7[0],(int)p7[1],(int)p7[2],(int)p7[3],0,0,0,0};

        int32_t s0=(int32_t)ScaleB[sb0+sc], s1=(int32_t)ScaleB[sb1+sc];
        int32_t s2=(int32_t)ScaleB[sb2+sc], s3=(int32_t)ScaleB[sb3+sc];
        int32_t s4=(int32_t)ScaleB[sb4+sc], s5=(int32_t)ScaleB[sb5+sc];
        int32_t s6=(int32_t)ScaleB[sb6+sc], s7=(int32_t)ScaleB[sb7+sc];

        c0 = __builtin_amdgcn_mfma_scale_f32_16x16x128_f8f6f4(a, b0, c0, 4, 4, 0, sca, 0, s0);
        c1 = __builtin_amdgcn_mfma_scale_f32_16x16x128_f8f6f4(a, b1, c1, 4, 4, 0, sca, 0, s1);
        c2 = __builtin_amdgcn_mfma_scale_f32_16x16x128_f8f6f4(a, b2, c2, 4, 4, 0, sca, 0, s2);
        c3 = __builtin_amdgcn_mfma_scale_f32_16x16x128_f8f6f4(a, b3, c3, 4, 4, 0, sca, 0, s3);
        c4 = __builtin_amdgcn_mfma_scale_f32_16x16x128_f8f6f4(a, b4, c4, 4, 4, 0, sca, 0, s4);
        c5 = __builtin_amdgcn_mfma_scale_f32_16x16x128_f8f6f4(a, b5, c5, 4, 4, 0, sca, 0, s5);
        c6 = __builtin_amdgcn_mfma_scale_f32_16x16x128_f8f6f4(a, b6, c6, 4, 4, 0, sca, 0, s6);
        c7 = __builtin_amdgcn_mfma_scale_f32_16x16x128_f8f6f4(a, b7, c7, 4, 4, 0, sca, 0, s7);
    }

    if (lane < 16) {
        if (n_start+ 0*16+lane<N) D[n_start+ 0*16+lane]=hip_bfloat16(c0[0]);
        if (n_start+ 1*16+lane<N) D[n_start+ 1*16+lane]=hip_bfloat16(c1[0]);
        if (n_start+ 2*16+lane<N) D[n_start+ 2*16+lane]=hip_bfloat16(c2[0]);
        if (n_start+ 3*16+lane<N) D[n_start+ 3*16+lane]=hip_bfloat16(c3[0]);
        if (n_start+ 4*16+lane<N) D[n_start+ 4*16+lane]=hip_bfloat16(c4[0]);
        if (n_start+ 5*16+lane<N) D[n_start+ 5*16+lane]=hip_bfloat16(c5[0]);
        if (n_start+ 6*16+lane<N) D[n_start+ 6*16+lane]=hip_bfloat16(c6[0]);
        if (n_start+ 7*16+lane<N) D[n_start+ 7*16+lane]=hip_bfloat16(c7[0]);
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
    hipLaunchKernelGGL(f4gemm_m1_v21, grid, block, 0, 0,
        (hip_bfloat16*)D, (uint8_t*)A, (uint8_t*)B,
        (uint8_t*)ScaleA, (uint8_t*)ScaleB,
        M, N, K, stride_A, stride_B, stride_D, stride_SA, stride_SB);
    (void)hipDeviceSynchronize();
}
