# RDMA NIC Traffic Monitor

Multi-node RDMA traffic monitor with Chrome Trace output for ionic/RoCE NICs.

## Architecture

```
┌─────────────────────────────────────────────┐
│  perftest.sh (orchestrator)                 │
│  ├─ docker containers (ib_write_bw)         │
│  ├─ start.sh (monitor launcher)             │
│  │   ├─ rdma_monitor.py --master (local)    │
│  │   ├─ ssh → rdma_monitor.py --node 0      │
│  │   └─ ssh → rdma_monitor.py --node 1      │
│  └─ cleanup & trace save                    │
└─────────────────────────────────────────────┘
```

- **Master**: TCP server on the launch machine, aggregates data, writes Chrome Trace JSON.
- **Slaves**: One per node, read sysfs RDMA counters, send rates to master via TCP.
- **Time reference**: All timestamps use the master's clock.

## Files

| File | Description |
|------|-------------|
| `rdma_monitor.py` | Core monitor (master/slave modes via `--master` / `--node N`) |
| `start.sh` | Deploys `rdma_monitor.py` to nodes via scp, launches master + slaves |
| `perftest.sh` | Full test orchestrator: containers → ib_write_bw → monitor → trace |

## Quick Start

### Monitor only (attach to existing traffic)

```bash
./start.sh --nodes node0,node1,node2 --interval 0.1 --output trace.json
# Ctrl-C to stop and save
```

### Full perftest (generate + monitor RDMA traffic)

```bash
./perftest.sh \
    --nodes node0,node1 \
    --image rocm/pytorch-private:tag \
    --duration 30 --interval 0.1 \
    --nic-index 0 --output trace.json
```

View output: `chrome://tracing` or `https://ui.perfetto.dev/`

## Counter Source

Priority order for reading RDMA traffic:

1. `hw_counters/rx_rdma_ucast_bytes` / `tx_rdma_ucast_bytes` — ionic/RoCE (captures RDMA bypass traffic)
2. `hw_counters/rx_bytes` / `tx_bytes` — traditional InfiniBand
3. `net/statistics/rx_bytes` / `tx_bytes` — fallback (kernel network stack only)

Rate is capped at link speed (read from `/sys/class/net/<dev>/speed`) to filter firmware counter batch artifacts.

## Hardware (tested)

- **NIC**: AMD Pensando ionic (8× per node)
- **Link**: 400 Gbps (4× NDR), max 50 GB/s per NIC
- **Counter update**: ~79 Hz (~12.6ms interval)
- **Total bandwidth**: 3.2 Tbps (400 GB/s) per node

## Trace Format

Chrome Trace JSON with counter events (`"ph": "C"`):

```json
{
  "name": "ionic_0(benic1p1) rx",
  "ph": "C",
  "ts": 1500000,
  "pid": 0,
  "args": { "GB/s": 43.5 }
}
```

- `pid` = node index (0, 1, ...)
- Process metadata maps pid → hostname

## Dependencies

- Python 3 standard library only (zero external packages)
- Docker (for perftest.sh container management)
- SSH passwordless access between nodes
- `ib_write_bw` (perftest) inside container image

## Test Results

ib_write_bw on ionic_0 between 2 nodes:
- **Average**: ~346 Gbps (~43 GB/s), 87% of 400G link
- **Peak monitored**: 50.0 GB/s (capped at link speed)
