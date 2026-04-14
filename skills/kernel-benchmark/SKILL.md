---
name: kernel-benchmark
description: Compiles, validates, and benchmarks a HIP kernel (.cpp file) against a Python reference (*_ref.py). Auto-detects GPU arch and infers dimension args from the extern C void solve signature. Use when validating or benchmarking a .cpp file.
---

# Kernel Benchmark

对 HIP kernel 执行正确性验证 + 压测。所有命令在项目根目录执行。

## 执行流程

```bash
python3 skills/kernel-benchmark/scripts/benchmark.py <cpp_file> \
    --ref=<ref_file> [--PARAM=VALUE ...] --repeat=20
```

示例：
```bash
python3 skills/kernel-benchmark/scripts/benchmark.py kernel/MatrixTranspose/solution.cpp \
    --ref=kernel/MatrixTranspose/transpose_ref.py --M=10000 --N=1000 --repeat=20
```

## 参数推断

| 参数 | 推断方式 |
|------|---------|
| `<cpp_file>` | 用户提供 |
| `<ref_file>` | 同目录下 `*_ref.py` |
| 维度参数 | 从 `extern "C" void solve(...)` 签名推断 |
| `--repeat` | 默认 20 |

## 输出

```markdown
## Kernel 验证报告
- **结果**: ALL PASS / FAILED
| 指标 | Kernel | Reference |
|------|--------|-----------|
| Average | X.XXXX ms | X.XXXX ms |
| Speedup | X.XXx vs ref | - |
```
