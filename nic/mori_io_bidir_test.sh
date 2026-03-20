#!/bin/bash
set -eo pipefail

NODE="${NODE:-smci355-ccs-aus-n08-21}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
OUTPUT="/tmp/xgmi_mori_io_bidir2_trace.json"
IMAGE="rocm/pytorch-private:sglang-0.5.9-rocm700-mi35x-mori-tbo-0317v2"

echo "=== MORI-IO Simultaneous Bidirectional (GPU3 <-> GPU4) ==="

ssh -o StrictHostKeyChecking=no $NODE "pkill -f 'xgmi_monitor.py' 2>/dev/null; fuser -k 29501/tcp 2>/dev/null" || true
sleep 1

scp -q -o StrictHostKeyChecking=no "$SCRIPT_DIR/xgmi_monitor.py" "$NODE:/tmp/xgmi_monitor.py"
scp -q -o StrictHostKeyChecking=no /tmp/mori_io_bidir_sim.py "$NODE:/tmp/mori_io_bidir_sim.py"
LOCAL_IP=$(ssh -o StrictHostKeyChecking=no $NODE "hostname -I | awk '{print \$1}'")

echo "[1/3] Starting xgmi_monitor..."
ssh -o StrictHostKeyChecking=no $NODE "python3 /tmp/xgmi_monitor.py --master --init-addr $LOCAL_IP --port 29501 --interval 0.02 --output $OUTPUT" &
MASTER_SSH=$!
sleep 1
ssh -o StrictHostKeyChecking=no $NODE "python3 /tmp/xgmi_monitor.py --node 0 --init-addr $LOCAL_IP --port 29501 --interval 0.02" &
SLAVE_SSH=$!
sleep 2

echo "[2/3] Running simultaneous bidirectional write..."
ssh -o StrictHostKeyChecking=no $NODE "docker run --rm --privileged --network=host \
  --device=/dev/kfd --device=/dev/dri --shm-size=64g \
  -v /tmp/mori:/workspace/mori -v /tmp:/host_tmp \
  $IMAGE \
  bash -c 'cd /workspace/mori && PYTHONPATH=/workspace/mori:\$PYTHONPATH python3 /host_tmp/mori_io_bidir_sim.py'" 2>&1

sleep 2

echo "[3/3] Stopping monitor..."
ssh -o StrictHostKeyChecking=no $NODE "pkill -INT -f 'xgmi_monitor.py --master' 2>/dev/null" || true
sleep 3
kill $MASTER_SSH $SLAVE_SSH 2>/dev/null || true
wait 2>/dev/null || true

scp -o StrictHostKeyChecking=no "$NODE:$OUTPUT" "$SCRIPT_DIR/xgmi_mori_io_bidir2_trace.json"

echo ""
echo "=== Trace Summary ==="
python3 -c "
import json
with open('$SCRIPT_DIR/xgmi_mori_io_bidir2_trace.json') as f:
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
echo "Trace: $SCRIPT_DIR/xgmi_mori_io_bidir2_trace.json"
