"""
Patch detokenizer event_loop to log per-step performance metrics.

Adds [DETOK_PERF] logs measuring:
  - recv_wait: Time waiting for data from scheduler (idle time)
  - recv_deser: Deserialization time (pickle.loads)
  - process_time: Request dispatch processing time
  - send_time: Serialization + send time (pickle.dumps + send)
  - queue_has_more: Whether more data was already queued (backpressure indicator)
"""
import sys

path = "/sgl-workspace/sglang/python/sglang/srt/managers/detokenizer_manager.py"

with open(path) as f:
    code = f.read()

# Check if pickle5 patch was already applied (uses pickle.loads / pickle.dumps)
uses_pickle_direct = "pickle.loads(self.recv_from_scheduler.recv())" in code

if uses_pickle_direct:
    old_loop = """            while True:
                recv_obj = pickle.loads(self.recv_from_scheduler.recv())

                output = self._request_dispatcher(recv_obj)

                if output is not None:
                    self.send_to_tokenizer.send(pickle.dumps(output, protocol=max(pickle.DEFAULT_PROTOCOL, 5)))"""
else:
    old_loop = """            while True:
                recv_obj = self.recv_from_scheduler.recv_pyobj()

                output = self._request_dispatcher(recv_obj)

                if output is not None:
                    self.send_to_tokenizer.send_pyobj(output)"""

if old_loop not in code:
    print("WARNING: Could not find exact loop pattern. Trying relaxed match...")
    print("Please verify and apply manually if needed.")
    sys.exit(1)

import_line = "import time as _time\nimport zmq as _zmq\n"
if "import time as _time" not in code:
    code = import_line + code

if uses_pickle_direct:
    new_loop = """            _counter = 0
            _last_send_time = _time.monotonic()
            while True:
                t0 = _time.monotonic()
                _raw = self.recv_from_scheduler.recv()
                t_recv = _time.monotonic()
                recv_obj = pickle.loads(_raw)
                t1 = _time.monotonic()
                recv_wait = t1 - _last_send_time
                recv_deser = t1 - t_recv

                output = self._request_dispatcher(recv_obj)
                t2 = _time.monotonic()
                process_time = t2 - t1

                if output is not None:
                    _out_bytes = pickle.dumps(output, protocol=max(pickle.DEFAULT_PROTOCOL, 5))
                    self.send_to_tokenizer.send(_out_bytes)
                t3 = _time.monotonic()
                send_time = t3 - t2
                _last_send_time = t3

                has_more = bool(self.recv_from_scheduler.get(_zmq.EVENTS) & _zmq.POLLIN)
                _counter += 1
                if _counter % 10 == 0:
                    num_reqs = getattr(recv_obj, 'num_reqs', '?')
                    logger.info(
                        f"[DETOK_PERF] step={_counter} reqs={num_reqs} "
                        f"recv_wait={recv_wait*1000:.1f}ms recv_deser={recv_deser*1000:.1f}ms "
                        f"process={process_time*1000:.1f}ms send={send_time*1000:.1f}ms "
                        f"queue_has_more={has_more}"
                    )"""
else:
    new_loop = """            _counter = 0
            _last_send_time = _time.monotonic()
            while True:
                t0 = _time.monotonic()
                recv_obj = self.recv_from_scheduler.recv_pyobj()
                t1 = _time.monotonic()
                recv_wait = t1 - _last_send_time
                recv_deser = t1 - t0

                output = self._request_dispatcher(recv_obj)
                t2 = _time.monotonic()
                process_time = t2 - t1

                if output is not None:
                    self.send_to_tokenizer.send_pyobj(output)
                t3 = _time.monotonic()
                send_time = t3 - t2
                _last_send_time = t3

                has_more = bool(self.recv_from_scheduler.get(_zmq.EVENTS) & _zmq.POLLIN)
                _counter += 1
                if _counter % 10 == 0:
                    num_reqs = getattr(recv_obj, 'num_reqs', '?')
                    logger.info(
                        f"[DETOK_PERF] step={_counter} reqs={num_reqs} "
                        f"recv_wait={recv_wait*1000:.1f}ms recv_deser={recv_deser*1000:.1f}ms "
                        f"process={process_time*1000:.1f}ms send={send_time*1000:.1f}ms "
                        f"queue_has_more={has_more}"
                    )"""

code = code.replace(old_loop, new_loop, 1)
with open(path, "w") as f:
    f.write(code)
print("Detokenizer event_loop patched with [DETOK_PERF] instrumentation")
