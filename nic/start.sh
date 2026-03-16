#!/bin/bash
#
# Launch multi-node RDMA traffic monitoring.
#
# Copies rdma_monitor.py to all nodes via scp, starts the master aggregator,
# then starts a slave process on each node via SSH (background).
# Ctrl-C stops the master (saves JSON), then all SSH connections are killed.
#
# Usage:
#   ./start.sh --nodes node0,node1,node2,node3 --interval 0.1 --output trace.json
#   ./start.sh --nodes node0 node1 node2 --interval 0.5 --init-addr 10.0.0.1
#

set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SCRIPT_NAME="rdma_monitor.py"
REMOTE_DIR="/tmp"

NODES=()
INTERVAL="1.0"
OUTPUT="rdma_trace.json"
PORT="29500"
INIT_ADDR=""

usage() {
    cat <<EOF
Usage: $(basename "$0") --nodes NODE[,NODE...] [OPTIONS]

Options:
  --nodes       Comma or space separated list of nodes to monitor (required)
  --interval    Sampling interval in seconds (default: 1.0, supports 0.1)
  --output      Output Chrome Trace JSON file (default: rdma_trace.json)
  --port        TCP port for master-slave communication (default: 29500)
  --init-addr   Master TCP IP (default: auto-detect via hostname -I)
  -h, --help    Show this help
EOF
    exit 0
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --nodes)
            shift
            while [[ $# -gt 0 && ! "$1" =~ ^-- ]]; do
                IFS=',' read -ra _SPLIT <<< "$1"
                NODES+=("${_SPLIT[@]}")
                shift
            done
            ;;
        --interval) INTERVAL="$2"; shift 2 ;;
        --output)   OUTPUT="$2";   shift 2 ;;
        --port)     PORT="$2";     shift 2 ;;
        --init-addr) INIT_ADDR="$2"; shift 2 ;;
        -h|--help)  usage ;;
        *) echo "Unknown option: $1"; usage ;;
    esac
done

if [[ ${#NODES[@]} -eq 0 ]]; then
    echo "Error: --nodes is required"
    usage
fi

if [[ -z "$INIT_ADDR" ]]; then
    INIT_ADDR=$(hostname -I 2>/dev/null | awk '{print $1}')
    if [[ -z "$INIT_ADDR" ]]; then
        echo "Error: Could not auto-detect IP. Specify --init-addr."
        exit 1
    fi
    echo "[start.sh] Auto-detected init-addr: $INIT_ADDR"
fi

echo "========================================"
echo " RDMA Traffic Monitor"
echo "========================================"
echo " Nodes:    ${NODES[*]}"
echo " Count:    ${#NODES[@]}"
echo " Interval: ${INTERVAL}s"
echo " Master:   ${INIT_ADDR}:${PORT}"
echo " Output:   ${OUTPUT}"
echo "========================================"

# ── Step 1: Copy script to all nodes via scp ─────────────────────────────

echo "[start.sh] Copying $SCRIPT_NAME to all ${#NODES[@]} nodes..."
SCP_PIDS=()
for node in "${NODES[@]}"; do
    scp -q -o StrictHostKeyChecking=no -o ConnectTimeout=10 \
        "$SCRIPT_DIR/$SCRIPT_NAME" "$node:$REMOTE_DIR/$SCRIPT_NAME" &
    SCP_PIDS+=($!)
done

FAIL=0
for pid in "${SCP_PIDS[@]}"; do
    if ! wait "$pid"; then
        FAIL=1
    fi
done
if [[ $FAIL -ne 0 ]]; then
    echo "[start.sh] ERROR: scp failed for one or more nodes. Aborting."
    exit 1
fi
echo "[start.sh] Copy complete."

# ── Step 2: Start master first (background), then slaves, then wait ──────

SSH_PIDS=()
MASTER_PID=""

cleanup() {
    echo ""
    echo "[start.sh] Terminating SSH connections..."
    for pid in "${SSH_PIDS[@]}"; do
        kill "$pid" 2>/dev/null
    done
    wait 2>/dev/null
    echo "[start.sh] Cleanup complete."
}
trap cleanup EXIT

# Forward Ctrl-C / SIGTERM to the master process, then re-wait
trap 'kill -INT $MASTER_PID 2>/dev/null' INT TERM

# Start master in background so we can launch slaves after it binds
echo "[start.sh] Starting master..."
python3 "$SCRIPT_DIR/$SCRIPT_NAME" \
    --master \
    --init-addr "$INIT_ADDR" \
    --port "$PORT" \
    --interval "$INTERVAL" \
    --output "$OUTPUT" &
MASTER_PID=$!

# Give master time to bind the TCP port
sleep 1

# ── Step 3: Start slaves on all nodes via SSH ────────────────────────────

echo "[start.sh] Starting slaves on all nodes..."
for i in "${!NODES[@]}"; do
    node="${NODES[$i]}"
    echo "[start.sh]   slave $i -> $node"
    ssh -o StrictHostKeyChecking=no -o ConnectTimeout=10 \
        "$node" \
        "python3 $REMOTE_DIR/$SCRIPT_NAME --node $i --init-addr $INIT_ADDR --port $PORT --interval $INTERVAL" \
        </dev/null >/dev/null 2>&1 &
    SSH_PIDS+=($!)
done

echo "[start.sh] All slaves launched. Ctrl-C to stop and save."
echo ""

# ── Wait for master to finish ────────────────────────────────────────────
# Loop because `wait` returns immediately when interrupted by a signal.
# This ensures we don't exit until the master has actually saved and exited.

while kill -0 $MASTER_PID 2>/dev/null; do
    wait $MASTER_PID 2>/dev/null || true
done

echo ""
echo "[start.sh] Master exited. Trace saved to: $OUTPUT"
