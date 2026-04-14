---
name: hip-code-gen
description: Generates optimized HIP kernel code for AMD GPUs based on performance analysis reports. Reads rocprof analysis reports and produces compilable .cpp files with applied optimizations. Does not handle compilation, execution, or profiling.
---

# HIP Code Gen

读取分析报告，生成优化后的 HIP kernel（`.cpp` 文件）。

## 执行流程

1. **需求解析**：算法类型、数据规模、精度、GPU 架构 (gfxXXX)
2. **读取输入**：`*_analysis.md` 报告 + 现有 kernel 代码
3. **确定策略**：按瓶颈类型从 [hip-optimization-strategies.md](hip-optimization-strategies.md) 查找
4. **生成代码**：带时间戳的新文件，不覆盖原文件

## 输出文件头模板

```cpp
/*
 * Optimized HIP Kernel - <算法名称>
 * 优化措施：
 *   [P0] LDS Tiling (TILE_SIZE=16)
 *   [P1] LDS Padding (+1 消除 Bank Conflict)
 * 编译：hipcc -O3 --offload-arch=gfx942 -o kernel solution_opt_<ts>.cpp
 */
```

## 代码质量要求

- 边界检查：处理尾 Tile 越界
- 架构兼容性：MFMA 仅支持 gfx908+
- 函数签名不变：`extern "C" void solve(...)`
- 使用 `#include <hip/hip_runtime.h>`

## 参考

- 优化策略 → [hip-optimization-strategies.md](hip-optimization-strategies.md)
