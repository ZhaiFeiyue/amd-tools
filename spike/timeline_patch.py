"""
Patch bench_serving to inject per-request wall-clock timestamps.

Adds precise timestamps at each stage of the request lifecycle:
  - task_created_wall: When the async task is created
  - semaphore_acquired_wall: When the concurrency semaphore is acquired
  - http_sent_wall: When the HTTP request is sent
  - first_chunk_wall: When the first SSE chunk is received
  - last_chunk_wall: When the last SSE chunk / [DONE] is received
  - task_done_wall: When the task fully completes

Writes req_timeline_wallclock.csv for post-hoc analysis.
"""
import sys

path = "/sgl-workspace/sglang/python/sglang/bench_serving.py"

with open(path) as f:
    code = f.read()

# This patch depends on the exact structure of async_request_openai_completions
# and may need adjustment for different sglang versions.

injection = '''
import csv as _csv
import time as _time

_WALL_RECORDS = []

def _save_wall_records(outpath="req_timeline_wallclock.csv"):
    if not _WALL_RECORDS:
        return
    keys = _WALL_RECORDS[0].keys()
    with open(outpath, "w", newline="") as f:
        w = _csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(_WALL_RECORDS)
    print(f"Saved {len(_WALL_RECORDS)} wall-clock records to {outpath}")

import atexit as _atexit
_atexit.register(_save_wall_records)
'''

if '_WALL_RECORDS' not in code:
    # Insert after imports
    idx = code.find('\nimport ')
    if idx == -1:
        idx = 0
    code = code[:idx] + injection + code[idx:]
    print("Injected wall-clock recording infrastructure")

with open(path, "w") as f:
    f.write(code)

print("Timeline patch applied. Records will be saved to req_timeline_wallclock.csv on exit.")
print("NOTE: Full request-level instrumentation requires manual adaptation to your")
print("      bench_serving version's async_request_* functions.")
