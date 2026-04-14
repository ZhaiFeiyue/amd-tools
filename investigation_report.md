# SGLang PD Disaggregation Performance Investigation Report

## Executive Summary

This document records the full investigation into SGLang PD (Prefill-Decode) disaggregation performance bottlenecks on AMD MI355X GPU clusters. Starting from a mysterious "ghost time" discrepancy between benchmark client duration and server-side active time, we systematically traced the root cause to the **detokenizer single-process bottleneck** -- specifically, redundant full-text serialization in IPC messages.

Three optimizations were implemented, achieving **+28% output throughput** improvement with only **~15 lines of code changed**.

---

## 1. Problem Statement

### Initial Observation

Running `bench_serving.py` with 10,240 prompts (ISL=1024, OSL=1024, BS=160, DP8, conc=2048):

| Metric | Value |
|--------|-------|
| `bench_serving` wall-clock duration | ~832s |
| Server active time (prefill + decode) | ~550s |
| **Unaccounted "ghost time"** | **~282s (34%)** |

### Key Question

Where does the ~282 seconds go? Client-side? Router? Internal server overhead?

---

## 2. Investigation Steps

### Step 1: Client-Side Timing Instrumentation

**Hypothesis**: Client SSE processing overhead.

**Method**: Added wall-clock timestamps to `bench_serving.py`:
- `task_created_wall`: Async task creation
- `semaphore_acquired_wall`: Concurrency semaphore acquisition
- `http_sent_wall`: HTTP request sent
- `first_chunk_wall`: First SSE chunk received
- `last_chunk_wall`: Last SSE chunk / `[DONE]` received
- `task_done_wall`: Task completion

**Finding**: Client-side processing was negligible. The time between `last_chunk_wall` and `task_done_wall` was <1ms per request.

### Step 2: Router SSE Logging

**Hypothesis**: Router (sgl-model-gateway, Rust) introduces delay between receiving decode stream and forwarding to client.

**Method**: Added `info!` logs in `pd_router.rs` for:
- `SSE_FIRST_CHUNK`: Timestamp of first chunk from decode
- `SSE_DONE`: Timestamp + chunk count when `[DONE]` is received

**Finding**: Router forwarding is near-instantaneous. The delay is upstream of the router -- inside the decode server.

### Step 3: Detokenizer Performance Instrumentation

**Hypothesis**: Single-process Python detokenizer is the bottleneck.

**Method**: Added `[DETOK_PERF]` logging to `detokenizer_manager.py::event_loop()`:
- `recv_wait`: Idle time waiting for ZMQ message
- `recv_deser`: pickle deserialization time
- `process_time`: `_request_dispatcher` processing
- `send_time`: pickle serialization + ZMQ send
- `queue_has_more`: Whether more messages are queued (backpressure indicator)

**Finding** (Critical):

| Metric | Value |
|--------|-------|
| `recv_deser` (deserialization) | **17.0ms** avg |
| `process_time` | 0.4ms avg |
| `send_time` | 0.1ms avg |
| `queue_has_more=True` | **89%** of steps |

The detokenizer spends **97%** of its per-step time on pickle deserialization. With 8 DP ranks each sending a message per decode step, the detokenizer processes 8 messages serially at 17ms each = **136ms per step** -- far exceeding the ~50ms decode step interval.

### Step 4: Root Cause Analysis -- `decoded_texts`

**Deep Dive**: Profiled the pickle payload.

The `BatchTokenIDOutput` dataclass sent from scheduler to detokenizer contains a `decoded_texts: List[str]` field. This field carries the **full accumulated output text** for every request in the batch, in every step.

With BS=160, after 500 tokens generated, each `decoded_texts` entry averages ~500 chars. Total payload: 160 x 500 = 80KB of text **per message, per step**.

However, the detokenizer only reads `decoded_texts[i]` **once** -- when creating a `DecodeStatus` for a new request ID. All subsequent sends are redundant.

**Verification**:
- `decoded_texts` consumed at `detokenizer_manager.py:225` in `DecodeStatus.__init__`
- Not used by tokenizer_manager, HTTP handler, or bench client
- Detokenizer maintains its own `s.decoded_text` state via incremental detokenization

### Step 5: Optimizations Applied

1. **decoded_texts elimination**: Only send full text on first output, empty string thereafter
2. **pickle protocol upgrade**: Use protocol 5 (PEP 574) for ZMQ IPC
3. **KV cache pre-allocation**: Configurable via `SGLANG_DISAGGREGATION_NUM_PRE_ALLOCATE_REQS`

---

## 3. Results

### Throughput Comparison

| Metric | Baseline | Optimized | Δ |
|--------|:--------:|:---------:|:-:|
| Output TPS | 16,289 | **20,865** | **+28.1%** |
| Total TPS | 32,577 | **41,728** | **+28.1%** |
| Duration | 579.6s | **452.5s** | **-21.9%** |
| TPOT Median | 62.1ms | **56.3ms** | -9.3% |
| TPOT P99 | 82.7ms | **60.9ms** | **-26.4%** |

### Detokenizer Metrics

| Metric | Before | After | Δ |
|--------|:------:|:-----:|:-:|
| recv_deser avg | 17.0ms | **6.5ms** | **-62%** |
| process avg | 0.4ms | 0.4ms | - |
| send avg | 0.1ms | 0.1ms | - |
| queue_has_more=True | 89% | 91% | +2pp |

---

## 4. Token Processing Pipeline Analysis

### Decode Forward → Client (full path)

```
GPU decode forward
    ↓
Scheduler: collect output token IDs
    ↓
Scheduler: build BatchTokenIDOutput (includes decoded_texts)
    ↓
Scheduler: pickle.dumps → ZMQ PUSH
    ↓  ← IPC (intra-process)
Detokenizer: ZMQ PULL → pickle.loads  ← BOTTLENECK (17ms → 6.5ms after fix)
    ↓
Detokenizer: incremental detokenize (token ID → text)
    ↓
Detokenizer: build BatchStrOutput
    ↓
Detokenizer: pickle.dumps → ZMQ PUSH
    ↓  ← IPC (intra-process)
TokenizerManager: ZMQ PULL → pickle.loads
    ↓
TokenizerManager: SSE stream → uvicorn HTTP
    ↓  ← Network
Router (Rust): tokio recv → forward to client
    ↓  ← Network
bench_serving client: aiohttp SSE parse
```

### Key Insight

With 8 DP ranks and a single detokenizer, each decode step produces 8 `BatchTokenIDOutput` messages. The detokenizer processes them serially:

```
Step N: [DP0: 17ms][DP1: 17ms]...[DP7: 17ms] = 136ms total
                 ↑ After fix: 6.5ms each = 52ms total
```

At a decode step interval of ~50ms, the baseline detokenizer can't keep up (136ms > 50ms), causing growing backpressure. After the fix (52ms ≈ 50ms), it barely keeps up at DP8.

---

## 5. KV Transfer Analysis

### Mori Backend Architecture

```
Prefill DP Rank 0 ──RDMA──→ Decode DP Rank 0
Prefill DP Rank 1 ──RDMA──→ Decode DP Rank 1
...
Prefill DP Rank 3 ──RDMA──→ Decode DP Rank 7
```

### transfer_lock Bottleneck

`MoriKVManager` uses a global `threading.Lock` (`transfer_lock`) for all KV transfers within a DP rank. Even with `SGLANG_MORI_QP_PER_TRANSFER=4` and `SGLANG_MORI_NUM_WORKERS=4`, transfers serialize:

```python
# MoriKVManager.kv_transfer()
with self.transfer_lock:      # ← Global mutex
    self._do_rdma_transfer()  # Multiple requests wait here
```

**Impact**: Decode running-request ramp-up takes ~21s (visible in logs as gradual increase from 0 to 160 running requests).

### Alternative: Mooncake Backend

Mooncake uses lock-free multi-queue design:
- Per-request transfer state machine (no global lock)
- Async completion callbacks
- Better suited for high-concurrency PD disaggregation

---

## 6. Scaling Analysis

### Current: EP8 (1 node decode)

- 8 DP ranks → 8 messages/step to detokenizer
- After optimization: 8 × 6.5ms = 52ms/step ≈ borderline OK

### EP16 (2 nodes decode)

- 16 DP ranks → 16 messages/step
- Overhead: 16 × 6.5ms = **104ms/step** -- will bottleneck
- **Requires**: Multi-process detokenizer

### EP32/EP64

- Single-process detokenizer is completely infeasible
- Options:
  1. **Multi-process detokenizer**: DP-rank-sharded (N detokenizers, each handling K ranks)
  2. **Scheduler-embedded detokenize**: Skip detokenizer process entirely
  3. **Rust detokenizer**: Eliminate Python overhead entirely

---

## 7. Standalone vs PD Performance

### Test: ISL=1024, OSL=1024, 10 prompts (isolated)

| Config | TTFT Median |
|--------|:-----------:|
| Standalone (no PD) | ~480ms |
| PD Disaggregation | ~420ms |

PD shows slightly **lower** TTFT due to pipeline overlap: prefill compute overlaps with KV transfer initiation.

---

## 8. External Orchestration Comparison

### SGLang Built-in Router vs External Frameworks

| Feature | SGLang Router | llm-d | NVIDIA Dynamo |
|---------|:------------:|:-----:|:------------:|
| Architecture | Rust binary (sgl-model-gateway) | K8s-native Go | Distributed runtime (Rust+Python) |
| Routing | Hash/Round-Robin | Prefix-cache-aware EPP | KV-cache-aware |
| Scaling | Manual | K8s HPA | Custom autoscaler |
| Engine Support | SGLang only | vLLM | vLLM, SGLang, TRT-LLM |
| PD Disaggregation | Built-in | Gateway-level | Runtime-level |
| Production Ready | Yes | Alpha/Beta | Preview |

### Key Insight

SGLang's built-in router is production-ready and sufficient for most deployments. External frameworks add value for:
- **Multi-model serving** (Dynamo)
- **Kubernetes-native autoscaling** (llm-d)
- **Prefix-cache-aware routing** across instances (llm-d EPP, Dynamo)

---

## 9. Configuration Reference

### Best Performance Config

```bash
# Decode server
--max-running-requests 160
--context-length 4096
--cuda-graph-bs 160
--stream-interval 1
--dp-size 8

# Environment
SGLANG_DISAGGREGATION_NUM_PRE_ALLOCATE_REQS=128
SGLANG_MORI_NUM_MAX_DISPATCH_TOKENS_PER_RANK=256

# Prefill server
--max-running-requests 128  # 32 per DP rank
--dp-size 4
--tp-size 4
```

### Patches Applied (in order)

1. `decoded_texts_patch.py` -- Eliminate redundant text in IPC
2. `pickle5_ipc_patch.py` -- Use pickle protocol >= 5
3. `prealloc_patch.py` -- Configurable KV cache pre-allocation
4. `ms_logging_patch.py` -- Millisecond precision timestamps (optional)
5. `detok_perf_patch.py` -- Performance instrumentation (optional, for debugging)

---

## 10. Files Modified

| File | Change |
|------|--------|
| `scheduler_output_processor_mixin.py:998` | Send `decoded_texts` only on first output |
| `scheduler.py` (SenderWrapper) | `send_pyobj` → `pickle.dumps(protocol>=5)` |
| `detokenizer_manager.py` (event_loop) | `recv_pyobj/send_pyobj` → explicit pickle |
| `tokenizer_manager.py` (handle_loop) | `recv_pyobj` → `pickle.loads(recv())` |
| `model_runner_kv_cache_mixin.py` | Configurable `pre_alloc_size` via env var |
| `common.py` (optional) | `.%(msecs)03d` in log format |

**Total**: ~15 lines changed across 5-6 files.

---

## Appendix A: Test Environment

- **Hardware**: AMD MI355X GPUs, 8 GPUs per node
- **Network**: RDMA (Mori backend) between nodes
- **Model**: DeepSeek-R1-0528-MXFP4-th
- **Topology**: 1P1D (1 Prefill node + 1 Decode node)
  - Prefill: DP4 TP4 EP4
  - Decode: DP8 TP8 EP8
- **Benchmark**: `bench_serving.py` with async HTTP/SSE

## Appendix B: Patch Files

All patches are located in `/apps/feiyue/upstream/amd-tools/`:

| File | Purpose |
|------|---------|
| `decoded_texts_patch.py` | Eliminate redundant decoded_texts |
| `pickle5_ipc_patch.py` | Upgrade pickle protocol to >=5 |
| `prealloc_patch.py` | Configurable KV cache pre-allocation |
| `ms_logging_patch.py` | Millisecond precision logging |
| `detok_perf_patch.py` | Detokenizer performance instrumentation |
| `timeline_patch.py` | bench_serving wall-clock timestamps |
| `run_best_config.sh` | One-click reproduction script |
| `optimizations.md` | Optimization summary with code diffs |
