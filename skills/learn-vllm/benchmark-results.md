# vLLM Benchmark Results & Performance

## Industry Benchmarks (2025-2026)

### vLLM vs Competitors (NVIDIA Blackwell RTX PRO 6000)

| Engine | Throughput (tok/s) | TTFT (ms) | Concurrency 128 |
|--------|-------------------|-----------|-----------------|
| vLLM (NVFP4) | 8,033 | 10.7 | 100% success |
| SGLang (GPTQ-INT4) | 6,395 | — | 100% success |
| Ollama | 484 | 65 | Fails under load |

### NVIDIA GB300 Performance (DeepSeek, NVFP4)

| Model | Scenario | Performance |
|-------|----------|-------------|
| DeepSeek-V3.2 | Prefill-only | 7,360 tok/GPU/s |
| DeepSeek-R1 (2x GPU) | Prefill | 22,476 tok/GPU/s |

8-20x improvement over Hopper GPUs with Blackwell + NVFP4.

## Local Benchmark: Kimi-K2.5 on AMD MI355X

**Setup**:
- Model: moonshotai/Kimi-K2.5 (1T total, 32B activated, INT4 compressed-tensors)
- Engine: vLLM v0.17.2rc1.dev43 (nightly 20260318)
- Hardware: 8x AMD MI355X (gfx950, 288GB HBM3E each)
- Parallelism: TP=4 (TRITON_MLA attention)
- Config: max_model_len=32768, gpu_memory_utilization=0.85
- Workload: random, input_len=1024, output_len=128

### Throughput & Latency

| Concurrency | Req/s | Output tok/s | Total tok/s | Mean TTFT (ms) | P99 TTFT (ms) | Mean ITL (ms) | P99 ITL (ms) |
|-------------|-------|-------------|-------------|----------------|---------------|---------------|--------------|
| 1 | 0.26 | 32.87 | 295.79 | 566 | 574 | 26.21 | 26.42 |
| 8 | 0.68 | 87.55 | 787.99 | 1326 | 2336 | 81.61 | 90.98 |
| 32 | 0.96 | 122.55 | 1102.96 | 2403 | 6100 | 214.82 | 261.63 |
| 64 | 1.41 | 181.12 | 1630.07 | 3919 | 11819 | 274.11 | 1958.73 |

### Key Observations

- Single-request: 26ms/token ITL, 566ms TTFT
- Peak throughput: ~181 output tok/s at concurrency=64 (1630 total tok/s)
- TTFT degrades significantly at high concurrency (continuous batching impact)
- P99 ITL tail latency spike at concurrency=64 — prefill interference

### Profiling Summary (Prefill, BS=1, TP4)

Top GPU kernels (1.343s total):

| Kernel | CUDA Time | % | Category |
|--------|-----------|---|----------|
| fused_moe_kernel_gptq_awq | 715ms | 53.25% | MoE |
| _fwd_grouped_kernel_stage1 (MLA decode) | 179ms | 13.31% | Attention |
| gatherTopK (routing) | 50ms | 3.74% | MoE routing |
| cross_device_reduce (allreduce) | 47ms | 3.50% | Communication |
| wvSplitK (GEMM) | 44ms | 3.31% | GEMM |
| flash_attn varlen forward (MLA prefill) | 43ms | 3.21% | Attention |
| ck_tile FmhaFwd (CK flash attn) | 37ms | 2.77% | Attention |
| bmm (MLA) | 27ms | 1.98% | Attention |

**Insights**:
- MoE expert GEMM (INT4 GPTQ/AWQ) dominates at 53% — fused MoE kernel for 384 experts
- MLA attention total: ~19% (prefill + decode)
- Communication (allreduce): only 3.5% — TP4 is efficient
- Optimization targets: MoE kernel (53%), MLA attention (19%)

## Benchmark Commands

### Standard Benchmark

```bash
# Inside container or with vllm installed
vllm bench serve \
  --model <model-path> \
  --host 0.0.0.0 --port 8000 \
  --dataset-name random \
  --random-input-len 1024 --random-output-len 128 \
  --num-prompts 20 --max-concurrency 8 \
  --request-rate inf --trust-remote-code
```

### Profiling Workflow

```bash
# 1. Start server with profiler config
vllm serve <model> --port 8000 \
  --profiler-config '{"profiler":"torch","torch_profiler_dir":"/tmp/profile","torch_profiler_with_stack":true,"torch_profiler_with_flops":true,"torch_profiler_use_gzip":true}'

# 2. Warmup (2 rounds to eliminate JIT effects)
vllm bench serve --model <model> --port 8000 \
  --dataset-name random --random-input-len 2048 --random-output-len 8 \
  --num-prompts 8 --max-concurrency 4 --request-rate inf

# 3. Profile run
vllm bench serve --model <model> --port 8000 \
  --dataset-name random --random-input-len 2048 --random-output-len 4 \
  --num-prompts 4 --max-concurrency 1 --request-rate inf --profile

# 4. View traces at https://ui.perfetto.dev/
```

### Local Results Location

Benchmark results and profiling data from the team's Kimi-K2.5 evaluation:
- `vllm_kimi/bench_results/SUMMARY.md` — Full results with Docker commands
- `vllm_kimi/bench_results/serve_conc*.log` — Raw benchmark logs
- `vllm_kimi/profiles/` — Torch profiler trace files
- `vllm_kimi/analyze_traces.py` — Trace analysis tool
