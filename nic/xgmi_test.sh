#!/bin/bash
set -eo pipefail

NODE="${NODE:-smci355-ccs-aus-n08-21}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
OUTPUT="/tmp/xgmi_trace.json"
INTERVAL="0.05"

echo "=== xGMI Traffic Monitor Test ==="

# Clean up
ssh -o StrictHostKeyChecking=no $NODE "pkill -f 'xgmi_monitor.py' 2>/dev/null; fuser -k 29501/tcp 2>/dev/null" || true
sleep 1

# Copy script
scp -q -o StrictHostKeyChecking=no "$SCRIPT_DIR/xgmi_monitor.py" "$NODE:/tmp/xgmi_monitor.py"

# Get node IP
LOCAL_IP=$(ssh -o StrictHostKeyChecking=no $NODE "hostname -I | awk '{print \$1}'")
echo "Node IP: $LOCAL_IP"

# Start master
echo "[1/4] Starting xgmi_monitor master..."
ssh -o StrictHostKeyChecking=no $NODE "python3 /tmp/xgmi_monitor.py --master --init-addr $LOCAL_IP --port 29501 --interval $INTERVAL --output $OUTPUT" &
MASTER_SSH=$!
sleep 1

# Start slave
echo "[2/4] Starting xgmi_monitor slave..."
ssh -o StrictHostKeyChecking=no $NODE "python3 /tmp/xgmi_monitor.py --node 0 --init-addr $LOCAL_IP --port 29501 --interval $INTERVAL" &
SLAVE_SSH=$!
sleep 2

# Run traffic generator
echo "[3/4] Running all_reduce_perf (8 GPUs, 1M-1G, 20 iterations)..."
ssh -o StrictHostKeyChecking=no $NODE "
export PATH=/opt/sre-tools/ompi/bin:\$PATH
export LD_LIBRARY_PATH=/opt/sre-tools/ompi/lib:/opt/rocm-7.0.1/lib:/opt/rocm-7.1.1/lib:\$LD_LIBRARY_PATH
/opt/sre-tools/rccl-tests/bin/all_reduce_perf -b 1M -e 1G -f 2 -g 8 -n 20 -w 5
" 2>&1 | tail -20
echo "[3/4] Traffic generation complete."

# Wait a moment for final data
sleep 2

# Stop monitor
echo "[4/4] Stopping monitor..."
ssh -o StrictHostKeyChecking=no $NODE "pkill -INT -f 'xgmi_monitor.py --master' 2>/dev/null" || true
sleep 3

# Kill SSH sessions
kill $MASTER_SSH 2>/dev/null || true
kill $SLAVE_SSH 2>/dev/null || true
wait 2>/dev/null || true

# Copy trace back
echo "=== Retrieving trace file ==="
scp -o StrictHostKeyChecking=no "$NODE:$OUTPUT" "$SCRIPT_DIR/xgmi_trace.json"
ls -la "$SCRIPT_DIR/xgmi_trace.json"

echo ""
echo "=== Trace summary ==="
python3 -c "
import json
with open('$SCRIPT_DIR/xgmi_trace.json') as f:
    t = json.load(f)
events = t['traceEvents']
meta = [e for e in events if e['ph'] == 'M']
counter = [e for e in events if e['ph'] == 'C']
print(f'Total events: {len(events)}, Meta: {len(meta)}, Counter: {len(counter)}')
names = sorted(set(e['name'] for e in counter))
print(f'Series ({len(names)}): {names}')

# Find peak values
for name in names:
    vals = [e['args']['GB/s'] for e in counter if e['name'] == name]
    peak = max(vals) if vals else 0
    avg_nonzero = sum(v for v in vals if v > 0) / max(1, sum(1 for v in vals if v > 0))
    print(f'  {name}: peak={peak:.1f} GB/s, avg(>0)={avg_nonzero:.1f} GB/s, samples={len(vals)}')

ts_vals = sorted(set(e['ts'] for e in counter))
if len(ts_vals) >= 2:
    duration_s = (ts_vals[-1] - ts_vals[0]) / 1e6
    print(f'Duration: {duration_s:.1f}s, Time points: {len(ts_vals)}')
"

echo ""
echo "Trace file: $SCRIPT_DIR/xgmi_trace.json"
echo "View at: https://ui.perfetto.dev/"
