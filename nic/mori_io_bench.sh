#!/bin/bash
set -eo pipefail

NODE="${NODE:-smci355-ccs-aus-n08-21}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
OUTPUT="/tmp/xgmi_mori_io_trace.json"
IMAGE="rocm/pytorch-private:sglang-0.5.9-rocm700-mi35x-mori-tbo-0317v2"

echo "=== MORI-IO xGMI Write Benchmark (GPU3 -> GPU4) ==="

# Clean up
ssh -o StrictHostKeyChecking=no $NODE "pkill -f 'xgmi_monitor.py' 2>/dev/null; fuser -k 29501/tcp 2>/dev/null" || true
sleep 1

scp -q -o StrictHostKeyChecking=no "$SCRIPT_DIR/xgmi_monitor.py" "$NODE:/tmp/xgmi_monitor.py"
LOCAL_IP=$(ssh -o StrictHostKeyChecking=no $NODE "hostname -I | awk '{print \$1}'")

# Start xGMI monitor
echo "[1/3] Starting xgmi_monitor..."
ssh -o StrictHostKeyChecking=no $NODE "python3 /tmp/xgmi_monitor.py --master --init-addr $LOCAL_IP --port 29501 --interval 0.02 --output $OUTPUT" &
MASTER_SSH=$!
sleep 1
ssh -o StrictHostKeyChecking=no $NODE "python3 /tmp/xgmi_monitor.py --node 0 --init-addr $LOCAL_IP --port 29501 --interval 0.02" &
SLAVE_SSH=$!
sleep 2

# Run MORI-IO benchmark
echo "[2/3] Running MORI-IO benchmark..."
ssh -o StrictHostKeyChecking=no $NODE "docker run --rm --privileged --network=host \
  --device=/dev/kfd --device=/dev/dri --shm-size=64g \
  -v /tmp/mori:/workspace/mori \
  $IMAGE \
  bash -c 'pip install prettytable -q && cd /workspace/mori && PYTHONPATH=/workspace/mori:\$PYTHONPATH python3 tests/python/io/benchmark.py --backend xgmi --all --op-type write --enable-batch-transfer --enable-sess --src-gpu 3 --dst-gpu 4'" 2>&1 | tee /tmp/mori_io_output.txt

echo "[2/3] Benchmark complete."
sleep 2

# Stop monitor
echo "[3/3] Stopping monitor..."
ssh -o StrictHostKeyChecking=no $NODE "pkill -INT -f 'xgmi_monitor.py --master' 2>/dev/null" || true
sleep 3

kill $MASTER_SSH 2>/dev/null || true
kill $SLAVE_SSH 2>/dev/null || true
wait 2>/dev/null || true

# Retrieve trace
scp -o StrictHostKeyChecking=no "$NODE:$OUTPUT" "$SCRIPT_DIR/xgmi_mori_io_trace.json"
ls -la "$SCRIPT_DIR/xgmi_mori_io_trace.json"

echo ""
echo "=== Trace Summary ==="
python3 -c "
import json
with open('$SCRIPT_DIR/xgmi_mori_io_trace.json') as f:
    t = json.load(f)
events = t['traceEvents']
counter = [e for e in events if e['ph'] == 'C']
ts_vals = sorted(set(e['ts'] for e in counter))
dur = (ts_vals[-1] - ts_vals[0]) / 1e6 if len(ts_vals) >= 2 else 0
print(f'Events: {len(events)}, Duration: {dur:.1f}s')
for name in sorted(set(e['name'] for e in counter)):
    vals = [e['args']['GB/s'] for e in counter if e['name'] == name]
    peak = max(vals)
    nonzero = [v for v in vals if v > 1]
    avg_nz = sum(nonzero)/len(nonzero) if nonzero else 0
    if peak > 1:
        print(f'  {name}: peak={peak:.1f} GB/s, avg={avg_nz:.1f} GB/s')
"
echo ""
echo "Trace: $SCRIPT_DIR/xgmi_mori_io_trace.json"
