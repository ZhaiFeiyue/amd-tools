"""
Patch SGLang logging to include millisecond precision timestamps.

Changes the default logging format from 'HH:MM:SS' to 'HH:MM:SS.mmm'
across both basicConfig and uvicorn formatters.
"""
import sys

path = "/sgl-workspace/sglang/python/sglang/srt/utils/common.py"

with open(path) as f:
    code = f.read()

# Patch basicConfig format
old_basic = '    logging.basicConfig(\n        level=getattr(logging, log_level.upper()),\n        format="%(message)s",'
new_basic = '    logging.basicConfig(\n        level=getattr(logging, log_level.upper()),\n        format="%(asctime)s.%(msecs)03d %(message)s",'

if old_basic in code:
    code = code.replace(old_basic, new_basic, 1)
    print("Patched basicConfig format for ms precision")
else:
    # Try a more relaxed match
    if '%(message)s' in code and '%(msecs)' not in code:
        code = code.replace(
            'format="%(message)s"',
            'format="%(asctime)s.%(msecs)03d %(message)s"',
            1
        )
        print("Patched basicConfig format for ms precision (relaxed match)")

# Patch uvicorn formatter
old_uvicorn = '"%(asctime)s %(levelname)s %(message)s"'
new_uvicorn = '"%(asctime)s.%(msecs)03d %(levelname)s %(message)s"'
if old_uvicorn in code:
    code = code.replace(old_uvicorn, new_uvicorn)
    print("Patched uvicorn formatter for ms precision")

# Add datefmt to ensure consistent time format
if 'datefmt=' not in code:
    code = code.replace(
        'format="%(asctime)s.%(msecs)03d %(message)s",',
        'format="%(asctime)s.%(msecs)03d %(message)s",\n        datefmt="%Y-%m-%d %H:%M:%S",',
        1
    )
    print("Added datefmt for consistent time format")

with open(path, "w") as f:
    f.write(code)

print("Logging now includes millisecond precision timestamps")
