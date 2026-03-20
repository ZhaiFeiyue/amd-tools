#!/bin/bash
set -eo pipefail

NODE="${NODE:-smci355-ccs-aus-n08-21}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
OUTPUT="/tmp/xgmi_mori_trace.json"
INTERVAL="0.02"
IMAGE="rocm/pytorch-private:sglang-0.5.9-rocm700-mi35x-mori-tbo-0317v2"
DURATION=15

echo "=== MORI shmem P2P + xGMI Profile ==="

# Clean up
ssh -o StrictHostKeyChecking=no $NODE "pkill -f 'xgmi_monitor.py' 2>/dev/null; fuser -k 29501/tcp 2>/dev/null" || true
sleep 1

scp -q -o StrictHostKeyChecking=no "$SCRIPT_DIR/xgmi_monitor.py" "$NODE:/tmp/xgmi_monitor.py"
LOCAL_IP=$(ssh -o StrictHostKeyChecking=no $NODE "hostname -I | awk '{print \$1}'")

# Start xGMI monitor
echo "[1/4] Starting xgmi_monitor..."
ssh -o StrictHostKeyChecking=no $NODE "python3 /tmp/xgmi_monitor.py --master --init-addr $LOCAL_IP --port 29501 --interval $INTERVAL --output $OUTPUT" &
MASTER_SSH=$!
sleep 1
ssh -o StrictHostKeyChecking=no $NODE "python3 /tmp/xgmi_monitor.py --node 0 --init-addr $LOCAL_IP --port 29501 --interval $INTERVAL" &
SLAVE_SSH=$!
sleep 2

# Run MORI P2P benchmark with 2 GPUs
echo "[2/4] Running MORI shmem P2P (2 GPUs, ${DURATION}s)..."
ssh -o StrictHostKeyChecking=no $NODE "docker run --rm --privileged --network=host \
  --device=/dev/kfd --device=/dev/dri --shm-size=64g \
  -v /tmp:/host_tmp $IMAGE \
  bash -c 'torchrun --nproc_per_node=2 /host_tmp/mori_p2p_bench.py --size-mb 256 --duration $DURATION'" 2>&1 | grep -v "^W0320\|OMP_NUM_THREADS\|\*\*\*\*\*"

sleep 2

# Run with 8 GPUs (all-to-all ring)
echo "[3/4] Running MORI shmem P2P ring (8 GPUs, ${DURATION}s)..."
ssh -o StrictHostKeyChecking=no $NODE "docker run --rm --privileged --network=host \
  --device=/dev/kfd --device=/dev/dri --shm-size=64g \
  -v /tmp:/host_tmp $IMAGE \
  bash -c 'torchrun --nproc_per_node=8 /host_tmp/mori_p2p_bench.py --size-mb 256 --duration $DURATION'" 2>&1 | grep -v "^W0320\|OMP_NUM_THREADS\|\*\*\*\*\*"

sleep 2

# Stop monitor
echo "[4/4] Stopping monitor..."
ssh -o StrictHostKeyChecking=no $NODE "pkill -INT -f 'xgmi_monitor.py --master' 2>/dev/null" || true
sleep 3

kill $MASTER_SSH 2>/dev/null || true
kill $SLAVE_SSH 2>/dev/null || true
wait 2>/dev/null || true

# Retrieve trace
scp -o StrictHostKeyChecking=no "$NODE:$OUTPUT" "$SCRIPT_DIR/xgmi_mori_trace.json"
ls -la "$SCRIPT_DIR/xgmi_mori_trace.json"

echo ""
echo "=== Trace Summary ==="
python3 -c "
import json
with open('$SCRIPT_DIR/xgmi_mori_trace.json') as f:
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
    if peak > 5:
        print(f'  {name}: peak={peak:.1f} GB/s, avg={avg_nz:.1f} GB/s')
"
echo ""
echo "Trace: $SCRIPT_DIR/xgmi_mori_trace.json"
echo "View: https://ui.perfetto.dev/"
