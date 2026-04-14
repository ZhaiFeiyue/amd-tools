# CK Optimization Breakdown for M=1 FP4 GEMM

## CK Profile vs Our Best (v14)

| Metric | CK (id=14 sk=2) | Our v14 | Gap |
|--------|-----------------|---------|-----|
| TFLOPS | 4.19 | 2.34 | 1.79x |
| Waves | 512 | 1024 | CK fewer |
| Grid (WG) | 32 | 256 | CK 8x fewer |
| VGPR | 44 | 36 | CK uses more |
| LDS | 16384 B | 256 B | CK 64x more |
| VALU/wave | 220.5 | 258 | CK 14% fewer |
| SALU/wave | 169 | 89 | CK 90% more |
| VMEM_RD/wave | 50 | 56 | CK 11% fewer |
| LDS/wave | 44 | 2 | CK 22x more |
| Wait/Busy | 294% | 561% | CK 1.9x better |
| SQ_WAIT total | 1.51M | 2.87M | CK 1.9x less wait |

## CK's Optimizations (identified from profile + source analysis)

### 1. Large N-tile (128 vs 16)
- CK: tile_N=128, 8 MFMA tiles in N direction per K-iter
- v14: tile_N=16, 1 MFMA per K-iter per wave
- Impact: B data loaded once, reused 8x → fewer VMEM loads per MFMA

### 2. LDS staging with buffer_load_lds
- CK: `buffer_load ... offen lds` instruction — global → LDS directly (no VGPR)
- v14: `global_load` → VGPR → MFMA (no LDS)
- Impact: LDS read latency ~20 cycles vs VMEM ~400 cycles after first load

### 3. Double-buffered LDS (2 LDS buffers)
- CK: LDS = 16KB = 2 × 8KB buffers (pipeline V3)
- Loads K-iter N+1 into buffer 1 while computing from buffer 0
- Only 1 barrier per K-iter (not 2)
- v14: no LDS, 0 barriers

### 4. K-split (splitK=2 → KBatch=4)
- CK: Grid_z = 4, each WG does K/4 iters
- v14: 4-wave internal K-split (same approach)
- Similar contribution

### 5. Preshuffled B layout (coalesced cooperative load)
- CK: shuffle_weight makes 256 threads' loads perfectly coalesced
- v14: preshuffled B with direct load (coalesced per-lane)
- v14 already has this

### 6. ThreadGroupTensorSliceTransfer_DirectLoad
- CK: all 256 threads cooperatively load B tile in one shot
- Uses `buffer_load_dwordx4 ... offen lds` with m0 register for LDS offset
- This is the key instruction we can't replicate in HIP C++

## Estimated Individual Contributions

To measure each optimization's contribution independently, we need
controlled experiments isolating each variable.
