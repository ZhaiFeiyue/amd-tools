# AMD MI355X Interconnect Traffic Monitor & Analysis

Multi-node RDMA NIC and xGMI traffic monitor with Chrome Trace output, plus performance characterization of MI355X interconnects.

## Architecture

```
┌─────────────────────────────────────────────┐
│  start.sh (orchestrator)                    │
│  ├─ --script rdma_monitor.py (NIC)          │
│  └─ --script xgmi_monitor.py (xGMI)        │
│      ├─ monitor.py --master (local)         │
│      ├─ ssh → monitor.py --node 0           │
│      └─ ssh → monitor.py --node 1           │
└─────────────────────────────────────────────┘
```

- **Master**: TCP server on the launch machine, aggregates data, writes Chrome Trace JSON.
- **Slaves**: One per node, read sysfs counters, send rates to master via TCP.
- **Time reference**: All timestamps use the master's clock.

## Files

| File | Description |
|------|-------------|
| `rdma_monitor.py` | RDMA NIC traffic monitor (master/slave via `--master` / `--node N`) |
| `xgmi_monitor.py` | xGMI inter-GPU traffic monitor (binary sysfs `gpu_metrics` parsing) |
| `start.sh` | Deploys monitor to nodes via scp, launches master + slaves. `--script` selects monitor type |
| `perftest.sh` | Full RDMA test: containers → ib_write_bw → monitor → trace |

## Quick Start

### RDMA NIC Monitor

```bash
./start.sh --nodes node0,node1 --interval 0.1 --output rdma_trace.json
# Ctrl-C to stop and save
```

### xGMI Monitor

```bash
./start.sh --script xgmi_monitor.py --nodes node0 --interval 0.02 --output xgmi_trace.json
```

### Full RDMA perftest

```bash
./perftest.sh \
    --nodes node0,node1 \
    --image rocm/pytorch-private:tag \
    --duration 30 --interval 0.1 \
    --nic-index 0 --output trace.json
```

View output: `chrome://tracing` or https://ui.perfetto.dev/

---

## Hardware Specifications (MI355X Platform)

### RDMA NIC

| Parameter | Value |
|-----------|-------|
| NIC | AMD Pensando ionic (8× per node) |
| Link speed | 400 Gbps (4× NDR) per NIC |
| Max bandwidth | 50 GB/s per NIC |
| Total per node | 3.2 Tbps (400 GB/s) |
| Counter refresh | ~79 Hz (~12.6 ms interval) |
| Counter source | `/sys/class/infiniband/<dev>/ports/1/hw_counters/{rx,tx}_rdma_ucast_bytes` |

### xGMI (Inter-GPU)

| Parameter | Value |
|-----------|-------|
| GPU count | 8× MI355X per node (4 OAMs, 2 GPU per OAM) |
| Topology | Fully connected (all-to-all), no switch |
| Links per GPU | 7 (one to each other GPU) |
| Link speed | 36 Gbps per lane |
| Link width | 16 lanes |
| Link bandwidth | 576 Gbps = 72 GB/s per link (bidirectional) |
| Total per GPU | 504 GB/s (7 links × 72 GB/s) |
| Counter refresh | 500 Hz (2 ms interval) |
| Counter source | `/sys/class/drm/card*/device/gpu_metrics` (binary) |

---

## xGMI Performance Characterization

### Test Environment

- **Node**: smci355-ccs-aus-n08-21 (8× MI355X)
- **ROCm**: 7.0.1 / 7.1.1
- **Test tools**: hipMemcpy (ctypes), MORI shmem P2P, MORI-IO engine
- **Monitor**: `xgmi_monitor.py` (20 ms sampling interval)

### 1. Single-Link P2P Bandwidth (GPU3 → GPU4)

Tested with three independent methods — all converge on the same result:

| Method | Max BW (GB/s) | Efficiency |
|--------|--------------|------------|
| hipMemcpy P2P | 61.17 | 85% |
| MORI shmem_ptr_p2p + hipMemcpyAsync | 61.07 | 85% |
| MORI-IO engine (batch_write, 1 MB msg) | 60.86 | 85% |

**Conclusion**: Single xGMI link application bandwidth = **~61 GB/s** (85% of 72 GB/s theoretical).

### 2. MORI-IO Bandwidth vs Message Size

GPU3 → GPU4, batch=256, iters=128:

| MsgSize | Max BW (GB/s) | Avg BW (GB/s) | Min Lat (us) |
|---------|--------------|---------------|-------------|
| 8 B | 0.06 | 0.06 | 31.71 |
| 1 KB | 7.58 | 7.30 | 34.57 |
| 8 KB | 31.41 | 30.45 | 66.76 |
| 64 KB | 54.80 | 54.43 | 306.13 |
| 256 KB | 59.53 | 59.43 | 1127.24 |
| 1 MB | 60.86 | 60.83 | 4410.51 |

### 3. Bidirectional P2P (GPU3 ↔ GPU4, Simultaneous)

| Mode | Per-Direction BW | xGMI Link Usage |
|------|-----------------|-----------------|
| **Unidirectional** (GPU3→4) | 61 GB/s | fwd=69 GB/s data, rev=33 GB/s ACK |
| **Bidirectional** (GPU3↔4) | ~40 GB/s each | ~48 GB/s each direction (symmetric) |

Total link throughput: ~95-102 GB/s in both modes. Bidirectional splits the link capacity between two data streams + their ACKs.

### 4. 8-GPU Ring P2P

Each GPU writes to next GPU in ring (GPU_i → GPU_{i+1}):

| Metric | Value |
|--------|-------|
| Per-pair bandwidth | 61.07 GB/s |
| Latency per 256 MB | 4396 us |
| All 8 pairs identical | Yes |

### 5. xGMI Protocol Overhead (ACK Analysis)

Unidirectional GPU3 → GPU4 xGMI counter readings:

| Counter | GPU3 | GPU4 |
|---------|------|------|
| write (send) | 68.9 GB/s | 33.0 GB/s (ACK) |
| read (recv) | 33.0 GB/s (ACK) | 68.9 GB/s |

- **Data direction**: 68.9 GB/s (96% link utilization)
- **ACK direction**: 33.0 GB/s (46% link utilization)
- **ACK overhead**: ~47% of forward data bandwidth
- **Total bidirectional link usage**: ~102 GB/s

This ~40-47% ACK overhead is inherent to the Infinity Fabric protocol (per-flit ACK + cache coherency probes) and cannot be configured from software.

### 6. Latency Breakdown (Software vs Hardware)

GPU3 → GPU4, measured via hipMemcpy/hipMemcpyAsync/GPU events:

| Size | hipMemcpy (us) | Async+Sync (us) | AsyncOnly (us) | GPU Event (us) | SW Overhead (us) |
|------|---------------|-----------------|----------------|---------------|-----------------|
| 8 B | 10.86 | 18.39 | 2.75 | 8.33 | 10.05 |
| 64 B | 11.59 | 18.24 | 3.20 | 8.03 | 10.21 |
| 4 KB | 11.48 | 18.01 | 3.20 | 8.00 | 10.00 |
| 64 KB | 12.53 | 18.83 | 3.26 | 8.65 | 10.18 |
| 1 MB | 27.26 | 32.91 | 3.49 | 22.30 | 10.61 |
| 16 MB | 285.61 | 294.64 | 1.65 | 291.32 | 3.32 |

**Layer decomposition (small messages ≤64KB):**

| Layer | Latency | Description |
|-------|---------|-------------|
| CPU submission | ~3 us | HIP runtime → kernel driver → SDMA command queue |
| GPU hardware | ~8 us | SDMA engine startup + xGMI transfer + completion |
| Sync/poll overhead | ~10 us | hipStreamSync polling + kernel→user return |
| **Total** | **~18 us** | Async+Sync end-to-end |
| MORI-IO framework | +14 us | Engine scheduling, session, batch assembly |
| **MORI-IO total** | **~32 us** | As measured in MORI-IO benchmark |

**Key insight**: For small messages, software overhead (13 us) exceeds hardware transfer time (8 us). For large transfers (16 MB), hardware dominates (291 us vs 3 us overhead).

### 7. Counter Refresh Rates

| Interconnect | Refresh Rate | Interval |
|-------------|-------------|----------|
| RDMA NIC (ionic) | ~79 Hz | ~12.6 ms |
| xGMI (gpu_metrics) | 500 Hz | 2.0 ms |

xGMI counters update 6.3× faster than NIC counters, enabling finer-grained monitoring.

---

## xGMI vs NVLink Protocol Comparison

| Aspect | xGMI (MI355X) | NVLink 4.0 (H100) |
|--------|--------------|-------------------|
| Origin | CPU Infinity Fabric | Dedicated GPU interconnect |
| Flow control | Per-flit ACK/NAK | Credit-based |
| Design priority | Low latency (coherency) | High throughput |
| Topology | Direct all-to-all | Via NVSwitch (2 hops) |
| ACK overhead | ~40-47% reverse BW | <5% |
| Unidirectional efficiency | ~85% | ~96% |
| Bidirectional (per-dir) | ~56% | ~96% |
| Hardware latency | Lower (direct, 1 hop) | Higher (NVSwitch, 2 hops) |
| SW configurable ACK | No | No (but UC mode avail in RDMA) |

The fundamental tradeoff: xGMI sacrifices throughput efficiency for lower latency, inherited from its CPU interconnect lineage where cache coherency response time is critical.

---

## Counter Sources

### RDMA NIC

Priority order:
1. `hw_counters/rx_rdma_ucast_bytes` / `tx_rdma_ucast_bytes` — ionic/RoCE
2. `hw_counters/rx_bytes` / `tx_bytes` — traditional InfiniBand
3. `net/statistics/rx_bytes` / `tx_bytes` — fallback

Rate capped at link speed to filter firmware counter batch artifacts.

### xGMI

Binary parsing of `/sys/class/drm/card*/device/gpu_metrics`:
- Format: `format_revision=1, content_revision≥3`
- Read offset: `0x88` (8 links × 8 bytes = `xgmi_read_data_acc[0..7]` in KB)
- Write offset: `0xC8` (8 links × 8 bytes = `xgmi_write_data_acc[0..7]` in KB)
- Link speed: offset `0x60` (Gbps), link width: offset `0x62`
- Read time: ~0.77 ms for all 8 GPUs (vs ~50ms for `rocm-smi`)

---

## Trace Format

Chrome Trace JSON with counter events (`"ph": "C"`):

```json
{
  "name": "GPU0 read",
  "ph": "C",
  "ts": 1500000,
  "pid": 0,
  "args": { "GB/s": 43.5 }
}
```

- `pid` = node index (0, 1, ...)
- Process metadata maps pid → hostname
- Bandwidth unit: GB/s

## Dependencies

- Python 3 standard library only (zero external packages)
- Docker (for perftest.sh / MORI container tests)
- SSH passwordless access between nodes
- ROCm (for xGMI gpu_metrics sysfs access)
