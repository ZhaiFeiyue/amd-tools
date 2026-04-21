# deep-cluster.md — Cluster / networking-specific additions

The 9 base sections (SKILL.md) apply. Below is what's **cluster-specific**.

Covers: networking fabrics, topology, collective communication, storage
systems, congestion control.

## §1 System scope

- Scale: rack / pod / datacenter / multi-region?
- Workload: training (all-reduce dominant) / inference (p2p + some
  all-reduce) / storage (throughput + consistency)?
- Generation: what HW class? (IB HDR/NDR/XDR, Ethernet 100G/200G/400G/800G,
  NVLink gen, PCIe gen)

## §3 Topology & physical architecture

Must include a topology diagram (Mermaid `flowchart` with subgraphs for
racks, or drawio if multi-layer). Pin down:

- Node count, GPUs/node, NICs/node, NIC bandwidth
- Switch tier count, radix per tier, over-subscription ratio per tier
- Intra-rail vs inter-rail bandwidth (for rail-optimized fabrics)
- NUMA / PCIe topology within a host (which GPUs share a NIC / root complex)

## §4 作者证明 — Cluster-specific asks

In addition to the 6 base checks (SKILL.md):

- **Bandwidth budget**: for each collective operation, derive expected
  completion time from message size / link BW / topology hops.
  Verify against paper's reported number.
- **Tail latency model**: if paper claims p99 reduction, reproduce the
  p99 from an arrival-rate × service-rate model (M/D/1 / M/M/1 / M/G/1)
  or from the paper's simulation parameters.
- **Loss / retransmit accounting**: congestion control papers must
  account for head-of-line blocking, ECN reaction, and retransmission
  overhead in their throughput math.
- **Scaling formula**: what's `T_collective(N)` as a function of N
  ranks? (ring: `O(N)`, recursive doubling: `O(log N)`, flat: `O(1)` with
  BW cost)

## §5 Collective communication

- Which collectives: all-reduce / reduce-scatter / all-gather / all-to-all /
  broadcast / p2p?
- Algorithm: ring, tree, recursive doubling, double binary tree, halving-
  doubling, custom?
- Hierarchical (intra-node → inter-node → inter-DC)?
- GPU-direct / zero-copy path: NVLink / PCIe P2P / RDMA direct?
- Does paper use NCCL / RCCL / MPI / custom library? Which commit?

## §6 Congestion & traffic engineering

- Congestion control: DCQCN / HPCC / TIMELY / Swift / custom?
- ECN threshold, rate-reducer / rate-increaser parameters
- Load balancing: ECMP / packet-level spray / flowlet / adaptive routing?
- Deadlock avoidance in lossless fabric (IB SL, RoCE PFC priorities)

## §7 Storage & I/O (if applicable)

- Architecture: parallel FS (Lustre / 3FS / BeeGFS) / object (S3-like) /
  block? Metadata split from data?
- Throughput targets: read vs write, small-file vs large-file, random vs
  sequential
- Consistency model: strong / eventual / close-to-open?
- Checkpoint-specific optimizations (async / layered / in-memory staging)?

## §8 Fault tolerance & reliability

- Failure domains: link / NIC / switch / node / rack / pod / PDU
- Detection latency: how fast does the fabric notice a link down?
- Failover: routing protocol reaction time, job impact
- Training-specific: does paper quantify time-to-detect + time-to-recover
  for a training job after a node failure?

## §9 Cost & efficiency

- $/Gbps, $/node, optics cost
- Power (W/switch, W/port), cooling
- Rack density (GPUs/rack, GPUs/MW)

## §10 Scalability / future-proofing

- At what scale (node count) does the topology / protocol start to
  degrade?
- Next-gen HW (UE, XDR IB, 400G+ RoCE, NVLink-6) — what changes?

## §11 Software → Hardware reverse implication (MANDATORY for cluster)

What NIC / fabric features does the software argue for?
- SHARP / in-network reduction?
- Programmable congestion control (P4)?
- Hardware-accelerated PFC / RoCE ECN marking?
- RDMA extensions (unreliable datagram for ML, atomic collective offload)?
- Per-flow vs per-packet hashing to kill elephant-flow hotspots?
