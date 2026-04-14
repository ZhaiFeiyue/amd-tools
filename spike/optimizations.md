# SGLang PD Disaggregation Optimizations

## Overview

Three optimizations that together achieve **+28% Output TPS** (16,289 → 20,865 tok/s) on BS=160, DP8, SI=1, conc=2048.

| Optimization | File | Effect |
|-------------|------|--------|
| decoded_texts redundancy elimination | `scheduler_output_processor_mixin.py` | Reduce pickle payload 62% |
| pickle protocol upgrade | `scheduler.py`, `detokenizer_manager.py`, `tokenizer_manager.py` | Ensure protocol >= 5 |
| KV cache pre-allocation | `model_runner_kv_cache_mixin.py` + env var | Faster decode bootstrap |

---

## 1. decoded_texts Redundancy Elimination

### Problem

`stream_output_generation()` sends `req.decoded_text` (the full accumulated output text) in every `BatchTokenIDOutput` message to the detokenizer. However, the detokenizer only uses this field **once** -- when creating `DecodeStatus` for a new request ID. All subsequent sends are redundant.

With 160 requests each generating ~900 tokens, the redundant text reaches ~500KB per message, which is the dominant cost in pickle serialization/deserialization (17ms per message).

### Fix

**File**: `python/sglang/srt/managers/scheduler_output_processor_mixin.py`

**Line**: 998

```python
# Before
decoded_texts.append(req.decoded_text)
decode_ids, read_offset = req.init_incremental_detokenize()

# After
if not getattr(req, '_decoded_text_sent', False):
    decoded_texts.append(req.decoded_text)
    req._decoded_text_sent = True
else:
    decoded_texts.append("")
decode_ids, read_offset = req.init_incremental_detokenize()
```

### Verification

- Full code search confirms `decoded_texts` is only consumed at `detokenizer_manager.py:225` in `DecodeStatus` initialization
- No downstream consumer (tokenizer_manager, HTTP handler, bench client) uses this field
- Detokenizer maintains its own `s.decoded_text` state after initialization

---

## 2. pickle Protocol Upgrade

### Problem

ZMQ's `send_pyobj()` / `recv_pyobj()` use `pickle.DEFAULT_PROTOCOL` which is **4** on Python 3.10. Protocol 5 (PEP 574) supports out-of-band data for large buffers.

Note: micro-benchmarks show minimal difference between protocol 4 and 5 for plain Python objects (List[int], List[str]). The measured improvement in earlier tests (6.5ms -> 4.0ms) was likely due to machine variance rather than protocol difference. However, using `max(DEFAULT_PROTOCOL, 5)` is a safe forward-compatible change that ensures the best available protocol is always used.

### Fix

Four changes across three files. All use `max(pickle.DEFAULT_PROTOCOL, 5)` to remain forward-compatible with future Python versions that may ship protocol 6+.

**File A**: `python/sglang/srt/managers/scheduler.py`

Add at top of file:
```python
import pickle
```

In `SenderWrapper.send_output()`:
```python
# Before
self.socket.send_pyobj(output)

# After
self.socket.send(pickle.dumps(output, protocol=max(pickle.DEFAULT_PROTOCOL, 5)))
```

**File B**: `python/sglang/srt/managers/detokenizer_manager.py`

Add at top of file:
```python
import pickle
```

In `event_loop()`, recv:
```python
# Before
recv_obj = self.recv_from_scheduler.recv_pyobj()

# After
recv_obj = pickle.loads(self.recv_from_scheduler.recv())
```

In `event_loop()`, send:
```python
# Before
self.send_to_tokenizer.send_pyobj(output)

# After
self.send_to_tokenizer.send(pickle.dumps(output, protocol=max(pickle.DEFAULT_PROTOCOL, 5)))
```

**File C**: `python/sglang/srt/managers/tokenizer_manager.py`

Add at top of file:
```python
import pickle
```

In `handle_loop()`:
```python
# Before
recv_obj = await self.recv_from_detokenizer.recv_pyobj()

# After
recv_obj = pickle.loads(await self.recv_from_detokenizer.recv())
```

### Note

SGLang requires Python >= 3.10. pickle protocol 5 requires Python >= 3.8. The `max(DEFAULT_PROTOCOL, 5)` pattern ensures:
- Python 3.10 (DEFAULT_PROTOCOL=4) -> uses 5
- Future Python (DEFAULT_PROTOCOL=6+) -> uses the higher value

---

## 3. KV Cache Pre-allocation

### Problem

In PD disaggregation, decode server needs to pre-allocate KV cache slots for incoming requests. The default behavior (`pre_alloc_size = 0` when `max_num_reqs > 32`) means no pre-allocation, adding latency to the decode bootstrap process.

### Fix

**File**: `python/sglang/srt/model_executor/model_runner_kv_cache_mixin.py`

```python
# Before
pre_alloc_size = max_num_reqs * 2 if max_num_reqs <= 32 else 0

# After
pre_alloc_size = int(__import__("os").environ.get(
    "SGLANG_DISAGGREGATION_NUM_PRE_ALLOCATE_REQS", "0"
))
pre_alloc_size = (
    max_num_reqs * 2 if max_num_reqs <= 32 else pre_alloc_size
)
```

### Configuration

Add to decode server environment:
```bash
SGLANG_DISAGGREGATION_NUM_PRE_ALLOCATE_REQS=128
```

---

## 4. Millisecond Logging Timestamps (Optional)

### Problem

Default SGLang log timestamps are second-precision (`2026-04-08 11:06:30`), insufficient for analyzing per-step decode timing and KV transfer latency.

### Fix

**File**: `python/sglang/srt/utils/common.py`

Three changes to add `.%(msecs)03d` to log format:

```python
# 1. basicConfig format (line ~1140)
# Before
format=format,
# After
format=format.replace("%(asctime)s", "%(asctime)s.%(msecs)03d") if "%(asctime)s" in format else format,

# 2. uvicorn default formatter (line ~2352)
# Before
"fmt"] = "[%(asctime)s] %(levelprefix)s %(message)s"
# After
"fmt"] = "[%(asctime)s.%(msecs)03d] %(levelprefix)s %(message)s"

# 3. uvicorn access formatter (line ~2356)
# Before
"fmt"] = '[%(asctime)s] %(levelprefix)s ...'
# After
"fmt"] = '[%(asctime)s.%(msecs)03d] %(levelprefix)s ...'
```

Output: `[2026-04-08 11:06:30.937]` (millisecond precision)

---

## Results

### Test Environment

- Machines: smci355-ccs-aus-n08-29 (prefill) + n08-33 (decode)
- Model: DeepSeek-R1-0528-MXFP4-th
- Config: 1P1D, Prefill DP4 TP4 EP4, Decode DP8 TP8 EP8
- BS=160, SI=1 (default), conc=2048, 10240 prompts, ISL=1024, OSL=1024

### Comparison

| Metric | Baseline | Optimized | Change |
|--------|:--------:|:---------:|:------:|
| Output TPS | 16,289 | **20,865** | **+28.1%** |
| Total TPS | 32,577 | **41,728** | **+28.1%** |
| Duration | 579.6s | **452.5s** | **-21.9%** |
| TPOT Median | 62.1ms | **56.3ms** | -9.3% |
| TPOT P99 | 82.7ms | **60.9ms** | **-26.4%** |
| Successful | 10,240 | 10,240 | - |

### Detokenizer Performance (with DETOK_PERF instrumentation)

| Metric | Baseline | After decoded_texts fix |
|--------|:--------:|:----------------------:|
| recv_deser avg | 17.0ms | **6.5ms** (-62%) |
| process avg | 0.4ms | 0.4ms |
| send_ser avg | 0.1ms | 0.1ms |
| queue_has_more=True | 89% | 91% |

---

## Files Changed Summary

| File | Lines Changed | Description |
|------|:------------:|-------------|
| `scheduler_output_processor_mixin.py` | +4 -1 | decoded_texts: send full text only on first output |
| `scheduler.py` | +2 -1 | SenderWrapper: pickle protocol >= 5 |
| `detokenizer_manager.py` | +3 -2 | recv/send: pickle protocol >= 5 |
| `tokenizer_manager.py` | +2 -1 | recv: pickle protocol >= 5 |
| `model_runner_kv_cache_mixin.py` | +4 -1 | Configurable pre_alloc_size via env var |
| `common.py` (optional) | +3 -3 | Millisecond timestamps in logs |

Total: **~15 lines changed** across 5-6 files.

---

## Reproduction

One-click script:

```bash
./run_best_config.sh <PREFILL_NODE> <DECODE_NODE> [NUM_PROMPTS] [CONCURRENCY]

# Example:
./run_best_config.sh smci355-ccs-aus-n08-29 smci355-ccs-aus-n08-33 10240 2048
```

The script handles: container creation, patch application, server launch, router setup, warmup, benchmark, and result collection.

---

## Known Limitations and Future Work

### Detokenizer Bottleneck (Partially Solved)

- Single-process detokenizer handles all 8 DP ranks serially
- `decoded_texts` fix reduced per-message overhead from 17ms to 6.5ms
- At BS=160 DP8, overhead is 8 msgs x 6.5ms = 52ms/step, barely fitting in ~50ms decode step interval
- ITL Median=0 and P99=~1800ms due to burst SSE output pattern

### Scaling Concerns (EP16+)

- EP16 (2 nodes): 16 msgs/step x 6.5ms = 104ms/step -- will bottleneck
- Recommended: multi-process detokenizer (plan saved in `.cursor/plans/`)

### KV Transfer Bottleneck

- Mori uses `transfer_lock` (global mutex) -- all KV transfers serialize within a DP rank
- `SGLANG_MORI_QP_PER_TRANSFER=4` and `SGLANG_MORI_NUM_WORKERS=4` tested but no improvement (lock is the bottleneck, not QP/worker count)
- Mooncake backend uses lock-free multi-queue design -- better for high concurrency
- Decode running-req ramp-up takes ~21s due to serialized KV transfer

### Potential Optimizations Not Yet Implemented

1. Multi-process detokenizer (DP-sharded) -- eliminates detokenizer bottleneck for EP8-64
2. Scheduler-embedded detokenize -- skip detokenizer process entirely
3. Remove `transfer_lock` in Mori -- enable parallel KV transfer per DP rank
4. `msgspec.Struct` for IPC -- requires SGLang dataclass refactor (like vLLM v1)
