"""
Patch scheduler to only send decoded_text on first output for each request.
After the first send, send empty string to avoid redundant pickle overhead.
"""
import sys

path = "/sgl-workspace/sglang/python/sglang/srt/managers/scheduler_output_processor_mixin.py"

with open(path) as f:
    code = f.read()

old = """                decoded_texts.append(req.decoded_text)
                decode_ids, read_offset = req.init_incremental_detokenize()"""

new = """                if not getattr(req, '_decoded_text_sent', False):
                    decoded_texts.append(req.decoded_text)
                    req._decoded_text_sent = True
                else:
                    decoded_texts.append("")
                decode_ids, read_offset = req.init_incremental_detokenize()"""

if old not in code:
    print("ERROR: pattern not found")
    sys.exit(1)

code = code.replace(old, new, 1)
with open(path, "w") as f:
    f.write(code)
print("Patched: decoded_texts now sends empty string after first output")
