#!/bin/bash
set -eo pipefail

NODE="${NODE:-smci355-ccs-aus-n08-21}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
OUTPUT="/tmp/xgmi_p2p_trace.json"
INTERVAL="0.02"
GPU_SRC=0
GPU_DST=1
DURATION=15

echo "=== xGMI P2P Profile Test ==="
echo "  GPU${GPU_SRC} -> GPU${GPU_DST}, duration=${DURATION}s, interval=${INTERVAL}s"

# Clean up
ssh -o StrictHostKeyChecking=no $NODE "pkill -f 'xgmi_monitor.py' 2>/dev/null; fuser -k 29501/tcp 2>/dev/null" || true
sleep 1

# Copy scripts
scp -q -o StrictHostKeyChecking=no "$SCRIPT_DIR/xgmi_monitor.py" "$NODE:/tmp/xgmi_monitor.py"

LOCAL_IP=$(ssh -o StrictHostKeyChecking=no $NODE "hostname -I | awk '{print \$1}'")

# Start master
echo "[1/5] Starting xgmi_monitor master..."
ssh -o StrictHostKeyChecking=no $NODE "python3 /tmp/xgmi_monitor.py --master --init-addr $LOCAL_IP --port 29501 --interval $INTERVAL --output $OUTPUT" &
MASTER_SSH=$!
sleep 1

# Start slave
echo "[2/5] Starting xgmi_monitor slave..."
ssh -o StrictHostKeyChecking=no $NODE "python3 /tmp/xgmi_monitor.py --node 0 --init-addr $LOCAL_IP --port 29501 --interval $INTERVAL" &
SLAVE_SSH=$!
sleep 2

# Run P2P tests sequentially
echo "[3/5] Running P2P unidirectional GPU${GPU_SRC}->GPU${GPU_DST} (${DURATION}s)..."
ssh -o StrictHostKeyChecking=no $NODE "LD_LIBRARY_PATH=/opt/rocm/lib python3 /tmp/p2p_bw.py --src $GPU_SRC --dst $GPU_DST --size-mb 256 --duration $DURATION" 2>&1

sleep 2

echo "[4/5] Running P2P bidirectional GPU${GPU_SRC}<->GPU${GPU_DST} (${DURATION}s)..."
ssh -o StrictHostKeyChecking=no $NODE "LD_LIBRARY_PATH=/opt/rocm/lib python3 /tmp/p2p_bw.py --src $GPU_SRC --dst $GPU_DST --size-mb 256 --duration $DURATION --bidir" 2>&1

sleep 2

# Stop monitor
echo "[5/5] Stopping monitor..."
ssh -o StrictHostKeyChecking=no $NODE "pkill -INT -f 'xgmi_monitor.py --master' 2>/dev/null" || true
sleep 3

kill $MASTER_SSH 2>/dev/null || true
kill $SLAVE_SSH 2>/dev/null || true
wait 2>/dev/null || true

# Retrieve trace
scp -o StrictHostKeyChecking=no "$NODE:$OUTPUT" "$SCRIPT_DIR/xgmi_p2p_trace.json"
ls -la "$SCRIPT_DIR/xgmi_p2p_trace.json"

echo ""
echo "=== Trace Summary ==="
python3 -c "
import json
with open('$SCRIPT_DIR/xgmi_p2p_trace.json') as f:
    t = json.load(f)
events = t['traceEvents']
counter = [e for e in events if e['ph'] == 'C']
names = sorted(set(e['name'] for e in counter))
ts_vals = sorted(set(e['ts'] for e in counter))
dur = (ts_vals[-1] - ts_vals[0]) / 1e6 if len(ts_vals) >= 2 else 0

print(f'Events: {len(events)}, Time points: {len(ts_vals)}, Duration: {dur:.1f}s')
print(f'Series: {len(names)}')
print()

for name in names:
    vals = [e['args']['GB/s'] for e in counter if e['name'] == name]
    peak = max(vals)
    nonzero = [v for v in vals if v > 0.5]
    avg_nz = sum(nonzero)/len(nonzero) if nonzero else 0
    if peak > 1:
        print(f'  {name}: peak={peak:.1f} GB/s, avg={avg_nz:.1f} GB/s')
"

echo ""
echo "Trace: $SCRIPT_DIR/xgmi_p2p_trace.json"
echo "View: https://ui.perfetto.dev/"
