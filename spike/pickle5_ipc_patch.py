"""
Patch ZMQ IPC to use pickle with at least protocol 5 for faster serialization.

pickle.DEFAULT_PROTOCOL is 4 on Python 3.10, but protocol 5 (PEP 574) is
available since Python 3.8. This patch ensures protocol >= 5 is used while
remaining forward-compatible with future Python versions.

Changes:
  - scheduler.py: SenderWrapper.send_output uses pickle.dumps with min protocol 5
  - detokenizer_manager.py: send uses pickle.dumps with min protocol 5
  - detokenizer_manager.py: recv uses pickle.loads (handles any protocol)
  - tokenizer_manager.py: recv uses pickle.loads (handles any protocol)
"""
import sys

_PICKLE_PROTOCOL_SNIPPET = "max(pickle.DEFAULT_PROTOCOL, 5)"

# 1. Patch scheduler.py
sched_path = "/sgl-workspace/sglang/python/sglang/srt/managers/scheduler.py"
with open(sched_path) as f:
    scode = f.read()

old_send = "        self.socket.send_pyobj(output)"
new_send = f"        self.socket.send(pickle.dumps(output, protocol={_PICKLE_PROTOCOL_SNIPPET}))"

if old_send not in scode:
    print("ERROR: SenderWrapper.send_pyobj pattern not found in scheduler.py")
    sys.exit(1)

if "import pickle" not in scode:
    scode = "import pickle\n" + scode
scode = scode.replace(old_send, new_send, 1)
with open(sched_path, "w") as f:
    f.write(scode)
print(f"Scheduler: send_pyobj -> pickle.dumps(protocol={_PICKLE_PROTOCOL_SNIPPET})")

# 2. Patch detokenizer_manager.py
path = "/sgl-workspace/sglang/python/sglang/srt/managers/detokenizer_manager.py"
with open(path) as f:
    code = f.read()

if "import pickle" not in code:
    code = "import pickle\n" + code

old_recv = "                recv_obj = self.recv_from_scheduler.recv_pyobj()"
new_recv = "                recv_obj = pickle.loads(self.recv_from_scheduler.recv())"
if old_recv in code:
    code = code.replace(old_recv, new_recv, 1)
    print("Detokenizer recv: recv_pyobj -> pickle.loads(recv())")

old_send = "                self.send_to_tokenizer.send_pyobj(output)"
new_send = f"                self.send_to_tokenizer.send(pickle.dumps(output, protocol={_PICKLE_PROTOCOL_SNIPPET}))"
if old_send in code:
    code = code.replace(old_send, new_send, 1)
    print(f"Detokenizer send: send_pyobj -> pickle.dumps(protocol={_PICKLE_PROTOCOL_SNIPPET})")

with open(path, "w") as f:
    f.write(code)

# 3. Patch tokenizer_manager.py
tok_path = "/sgl-workspace/sglang/python/sglang/srt/managers/tokenizer_manager.py"
with open(tok_path) as f:
    tcode = f.read()

old_tok = "                recv_obj = await self.recv_from_detokenizer.recv_pyobj()"
new_tok = "                recv_obj = pickle.loads(await self.recv_from_detokenizer.recv())"
if "import pickle" not in tcode:
    tcode = "import pickle\n" + tcode
if old_tok in tcode:
    tcode = tcode.replace(old_tok, new_tok, 1)
    print("TokenizerManager recv: recv_pyobj -> pickle.loads(recv())")

with open(tok_path, "w") as f:
    f.write(tcode)

print(f"\nAll patches applied! Using pickle protocol={_PICKLE_PROTOCOL_SNIPPET} for all IPC.")
