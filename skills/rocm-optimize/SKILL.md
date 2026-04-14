---
name: rocm-optimize
description: Orchestrates a full profiling-driven HIP kernel optimization loop (write, validate, profile, analyze, optimize) on AMD GPUs until performance converges. Use when optimizing .cpp HIP files, improving GPU kernel performance on ROCm, or running a HIP/ROCm optimization workflow.
---

# HIP Kernel Optimize (ROCm)

驱动 **生成 → 验证压测 → 评估 → rocprof 分析 → 再优化** 的完整循环。

> **核心约束**：本技能是编排者，子技能返回后必须立刻继续下一步。

## 执行流程

### 进度追踪

```
Optimization Loop #N:
- [ ] Step 1: 正确性验证 + 性能压测 (kernel-benchmark)
- [ ] Step 2: 评估退出条件
- [ ] Step 3: rocprof Profiling + 瓶颈分析 (rocprof-analyze)
- [ ] Step 4: 实施优化 (hip-code-gen)
```

### 准备阶段

1. 没有 reference → 在 `kernel/<AlgoName>/` 创建 `<algo_name>_ref.py`
2. 没有 `.cpp` → 调用 hip-code-gen 生成初版

Reference 格式：
```python
"""Reference for: solve(const float* A, const float* B, float* C, int N)"""
import torch

def reference(*, A, B, C, N, **kwargs):
    C[:N] = A[:N] + B[:N]
```

### Step 1：正确性验证 + 性能压测

调用 kernel-benchmark 技能。验证失败 → hip-code-gen 修复 → 重新 Step 1。

### Step 2：评估退出条件

连续 2 轮性能提升 < 2% → 输出最终报告。

### Step 3：rocprof Profiling + 瓶颈分析

调用 rocprof-analyze。瓶颈类型：VRAM_MEMORY_BOUND / LDS_PRESSURE_BOUND / LATENCY_BOUND / COMPUTE_BOUND / OCCUPANCY_BOUND。

### Step 4：实施优化

调用 hip-code-gen，基于 P0 优化建议生成新 `solution_opt_<timestamp>.cpp`。

### 最终报告

```markdown
## HIP Kernel 优化报告
### 优化历程
| 轮次 | Kernel 文件 | Average (ms) | Speedup vs Ref | 主要优化项 |
|------|------------|-------------|----------------|-----------|
| 初版 | solution.cpp | X.XX | 0.XXx | 朴素实现 |
| #1   | solution_opt_<ts>.cpp | X.XX | X.XXx | P0: LDS Tiling |
```

## 技能依赖

| 子技能 | 用途 |
|--------|------|
| `hip-code-gen` | 生成 / 修复 / 优化 kernel 代码 |
| `kernel-benchmark` | 正确性验证 + 性能压测 |
| `rocprof-analyze` | rocprof Profiling + 优化建议 |
