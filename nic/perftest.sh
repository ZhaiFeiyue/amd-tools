#!/bin/bash
#
# Run RDMA perftest (ib_write_bw) between nodes in containers and collect
# traffic data with rdma_monitor.
#
# Flow:
#   1. Start "rdmaperftest" containers on all nodes (docker, --privileged, --network=host)
#   2. Auto-detect RDMA devices and IPs
#   3. Start ib_write_bw server on node0, clients on other nodes (inside containers)
#   4. Start rdma_monitor (start.sh) to collect traffic data
#   5. Ctrl-C or perftest completion -> save trace, clean up
#
# Usage:
#   ./perftest.sh --nodes node0,node1 --image rocm/pytorch-private:tag \
#                 --duration 60 --interval 0.1 --nic-index 0
#

set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CONTAINER_NAME="rdmaperftest"

NODES=()
IMAGE=""
DURATION=60
INTERVAL="0.1"
NIC_IDX=0
OUTPUT="rdma_perftest_trace.json"
PERFTEST_PORT=18515
MONITOR_PORT=29500
INIT_ADDR=""

usage() {
    cat <<EOF
Usage: $(basename "$0") --nodes NODE[,NODE...] --image IMAGE [OPTIONS]

Options:
  --nodes       Comma or space separated list of nodes (first = node0/server) (required)
  --image       Docker image with perftest tools (required)
  --duration    Perftest duration in seconds (default: 60)
  --interval    RDMA monitor sampling interval (default: 0.1)
  --nic-index   Which RDMA NIC index to test, 0-based (default: 0)
  --output      Output trace file (default: rdma_perftest_trace.json)
  --init-addr   Master IP for rdma_monitor (default: auto-detect)
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
        --image)      IMAGE="$2";      shift 2 ;;
        --duration)   DURATION="$2";   shift 2 ;;
        --interval)   INTERVAL="$2";   shift 2 ;;
        --nic-index)  NIC_IDX="$2";    shift 2 ;;
        --output)     OUTPUT="$2";     shift 2 ;;
        --init-addr)  INIT_ADDR="$2";  shift 2 ;;
        -h|--help)    usage ;;
        *)            echo "Unknown option: $1"; usage ;;
    esac
done

if [[ ${#NODES[@]} -eq 0 ]]; then echo "Error: --nodes required"; usage; fi
if [[ -z "$IMAGE" ]]; then echo "Error: --image required"; usage; fi
if [[ ${#NODES[@]} -lt 2 ]]; then echo "Error: need at least 2 nodes"; exit 1; fi

NODE0="${NODES[0]}"

if [[ -z "$INIT_ADDR" ]]; then
    INIT_ADDR=$(hostname -I 2>/dev/null | awk '{print $1}')
    if [[ -z "$INIT_ADDR" ]]; then
        echo "Error: Could not auto-detect IP. Specify --init-addr."
        exit 1
    fi
fi

echo "========================================"
echo " RDMA Perftest"
echo "========================================"
echo " Nodes:     ${NODES[*]}"
echo " Node0:     $NODE0 (server)"
echo " Image:     $IMAGE"
echo " Duration:  ${DURATION}s"
echo " NIC index: $NIC_IDX"
echo " Interval:  ${INTERVAL}s"
echo " Output:    $OUTPUT"
echo "========================================"

# ── Cleanup ──────────────────────────────────────────────────────────────

MONITOR_PID=""
PERFTEST_PIDS=()

cleanup() {
    echo ""
    echo "[perftest] Cleaning up..."

    # Stop rdma_monitor master
    if [[ -n "$MONITOR_PID" ]] && kill -0 "$MONITOR_PID" 2>/dev/null; then
        echo "[perftest] Stopping rdma_monitor..."
        pkill -INT -f "rdma_monitor.py --master" 2>/dev/null || true
        for _i in $(seq 1 15); do
            kill -0 "$MONITOR_PID" 2>/dev/null || break
            sleep 1
        done
        kill "$MONITOR_PID" 2>/dev/null || true
        wait "$MONITOR_PID" 2>/dev/null || true
    fi

    # Kill any remaining perftest SSH sessions
    for pid in "${PERFTEST_PIDS[@]}"; do
        kill "$pid" 2>/dev/null
    done
    wait 2>/dev/null || true

    # Stop and remove containers on all nodes
    echo "[perftest] Removing containers..."
    for node in "${NODES[@]}"; do
        ssh -o StrictHostKeyChecking=no "$node" \
            "docker rm -f $CONTAINER_NAME 2>/dev/null" </dev/null >/dev/null 2>&1 &
    done
    wait 2>/dev/null || true

    echo "[perftest] Cleanup complete."
}
trap 'echo ""; echo "[perftest] Ctrl-C received, stopping..."; exit 0' INT TERM
trap cleanup EXIT

# ── Step 1: Check image availability ─────────────────────────────────────

echo "[perftest] Checking image on all nodes..."
MISSING=()
for node in "${NODES[@]}"; do
    if ! ssh -o StrictHostKeyChecking=no "$node" \
        "docker image inspect $IMAGE >/dev/null 2>&1" </dev/null; then
        MISSING+=("$node")
    fi
done
if [[ ${#MISSING[@]} -gt 0 ]]; then
    echo "[perftest] WARNING: Image not found on: ${MISSING[*]}"
    echo "[perftest] Attempting docker pull on missing nodes..."
    for node in "${MISSING[@]}"; do
        ssh -o StrictHostKeyChecking=no "$node" \
            "docker pull $IMAGE" </dev/null 2>&1 &
    done
    wait
    # Re-check
    for node in "${MISSING[@]}"; do
        if ! ssh -o StrictHostKeyChecking=no "$node" \
            "docker image inspect $IMAGE >/dev/null 2>&1" </dev/null; then
            echo "[perftest] ERROR: Cannot get image on $node. Aborting."
            exit 1
        fi
    done
fi
echo "[perftest] Image available on all nodes."

# ── Step 2: Start containers ─────────────────────────────────────────────

echo "[perftest] Starting containers on all nodes..."
for node in "${NODES[@]}"; do
    ssh -o StrictHostKeyChecking=no "$node" \
        "docker rm -f $CONTAINER_NAME 2>/dev/null; \
         docker run -d --name $CONTAINER_NAME \
            --privileged --network=host \
            --ulimit memlock=-1:-1 \
            -v /dev/infiniband:/dev/infiniband \
            $IMAGE sleep infinity" </dev/null >/dev/null 2>&1 &
done
wait
echo "[perftest] Containers started."

# ── Step 3: Discover RDMA device and server IP ───────────────────────────

echo "[perftest] Discovering RDMA devices on $NODE0..."
IB_DEV=$(ssh -o StrictHostKeyChecking=no "$NODE0" \
    "ls /sys/class/infiniband/ | sort | sed -n '$((NIC_IDX + 1))p'" </dev/null)
NET_DEV=$(ssh -o StrictHostKeyChecking=no "$NODE0" \
    "ls /sys/class/infiniband/$IB_DEV/device/net/ 2>/dev/null | head -1" </dev/null)
SERVER_IP=$(ssh -o StrictHostKeyChecking=no "$NODE0" \
    "ip -4 addr show $NET_DEV 2>/dev/null | grep -oP 'inet \K[0-9.]+'" </dev/null)

if [[ -z "$IB_DEV" || -z "$SERVER_IP" ]]; then
    echo "[perftest] ERROR: Could not discover RDMA device at index $NIC_IDX on $NODE0"
    exit 1
fi
echo "[perftest] Device: $IB_DEV ($NET_DEV), Server IP: $SERVER_IP"

# ── Step 4: Start rdma_monitor (background) ──────────────────────────────

echo "[perftest] Starting RDMA traffic monitor..."
"$SCRIPT_DIR/start.sh" \
    --nodes "${NODES[@]}" \
    --interval "$INTERVAL" \
    --init-addr "$INIT_ADDR" \
    --port "$MONITOR_PORT" \
    --output "$OUTPUT" &
MONITOR_PID=$!

# Wait for monitor to be ready
sleep 5

# ── Step 5: Start ib_write_bw server on node0 ────────────────────────────

echo "[perftest] Starting ib_write_bw server on $NODE0 ($IB_DEV)..."
ssh -o StrictHostKeyChecking=no "$NODE0" \
    "docker exec $CONTAINER_NAME ib_write_bw \
        -d $IB_DEV -p $PERFTEST_PORT \
        --duration $((DURATION + 10)) --report_gbits -F" \
    </dev/null 2>&1 &
PERFTEST_PIDS+=($!)

sleep 3

# ── Step 6: Start ib_write_bw clients on other nodes ─────────────────────

for i in $(seq 1 $((${#NODES[@]} - 1))); do
    node="${NODES[$i]}"
    port=$((PERFTEST_PORT + i))

    # Use the same NIC index on the client side
    CLIENT_IB_DEV=$(ssh -o StrictHostKeyChecking=no "$node" \
        "ls /sys/class/infiniband/ | sort | sed -n '$((NIC_IDX + 1))p'" </dev/null)
    if [[ -z "$CLIENT_IB_DEV" ]]; then
        CLIENT_IB_DEV="$IB_DEV"
    fi

    echo "[perftest] Starting ib_write_bw client on $node ($CLIENT_IB_DEV) -> $SERVER_IP..."
    ssh -o StrictHostKeyChecking=no "$node" \
        "docker exec $CONTAINER_NAME ib_write_bw \
            -d $CLIENT_IB_DEV -p $PERFTEST_PORT \
            --duration $DURATION --report_gbits -F \
            $SERVER_IP" \
        </dev/null 2>&1 &
    PERFTEST_PIDS+=($!)
done

echo ""
echo "========================================"
echo " Perftest running for ${DURATION}s"
echo " Ctrl-C to stop early and save trace"
echo "========================================"
echo ""

# ── Step 7: Wait for perftest to complete ─────────────────────────────────

# Wait for all perftest client processes (they have --duration)
for pid in "${PERFTEST_PIDS[@]}"; do
    wait "$pid" 2>/dev/null || true
done

echo ""
echo "[perftest] Perftest completed. Stopping monitor and saving trace..."

# Stop rdma_monitor: send SIGINT directly to the Python master process
# (going through start.sh's trap chain is unreliable for background processes)
pkill -INT -f "rdma_monitor.py --master" 2>/dev/null || true

# Wait for the monitor stack (start.sh + children) to fully exit
if [[ -n "$MONITOR_PID" ]]; then
    for _i in $(seq 1 30); do
        kill -0 "$MONITOR_PID" 2>/dev/null || break
        sleep 1
    done
    kill "$MONITOR_PID" 2>/dev/null || true
    wait "$MONITOR_PID" 2>/dev/null || true
    MONITOR_PID=""
fi

echo ""
echo "========================================"
echo " Done! Trace saved to: $OUTPUT"
echo " Open in: chrome://tracing or https://ui.perfetto.dev/"
echo "========================================"
