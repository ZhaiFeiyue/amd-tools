"""
Patch KV cache pre-allocation to use SGLANG_DISAGGREGATION_NUM_PRE_ALLOCATE_REQS.

Makes pre_alloc_size configurable via environment variable instead of hardcoded logic.
Default behavior: only pre-allocate when max_num_reqs <= 32 (upstream default).
With env var set: use the specified pre-allocation count.
"""
import sys

path = "/sgl-workspace/sglang/python/sglang/srt/model_executor/model_runner_kv_cache_mixin.py"

with open(path) as f:
    code = f.read()

old = """        pre_alloc_size = max_num_reqs * 2 if max_num_reqs <= 32 else 0"""

new = """        pre_alloc_size = int(__import__("os").environ.get("SGLANG_DISAGGREGATION_NUM_PRE_ALLOCATE_REQS", "0"))
        pre_alloc_size = (
            max_num_reqs * 2 if max_num_reqs <= 32 else pre_alloc_size
        )"""

if old not in code:
    print("ERROR: pre_alloc_size pattern not found")
    sys.exit(1)

code = code.replace(old, new, 1)
with open(path, "w") as f:
    f.write(code)
print(f"Patched: pre_alloc_size now reads SGLANG_DISAGGREGATION_NUM_PRE_ALLOCATE_REQS env var")
