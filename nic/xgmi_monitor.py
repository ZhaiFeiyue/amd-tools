#!/usr/bin/env python3
"""
Multi-node xGMI traffic monitor (TCP master-slave).

Reads GPU metrics directly from /sys/class/drm/card*/device/gpu_metrics
binary sysfs file for maximum performance (~0.7ms for all 8 GPUs, vs
~4ms for amdsmi library, vs ~630ms for rocm-smi CLI).

Binary layout (format_revision=1):
  - Header: uint16 structure_size, uint8 format_revision, uint8 content_revision
  - xgmi_read_data_acc  @ offset 0x88 : 8 x uint64 (accumulated kB per link)
  - xgmi_write_data_acc @ offset 0xC8 : 8 x uint64 (accumulated kB per link)

Two modes:
  --master : TCP server, aggregates data from all slaves, writes Chrome Trace JSON.
  --node N : Slave, reads local GPU metrics, sends to master via TCP.

Usage (via start.sh):
    ./start.sh --nodes node0,node1 --interval 0.1 --script xgmi_monitor.py --output xgmi_trace.json

Manual usage:
    # Master (on launch machine):
    python3 xgmi_monitor.py --master --init-addr 10.0.0.1 --interval 0.1 --output xgmi_trace.json
    # Slave (on each node):
    python3 xgmi_monitor.py --node 0 --init-addr 10.0.0.1 --interval 0.1
"""

import os
import sys
import json
import time
import glob
import struct
import socket
import signal
import argparse
import threading
import queue

DRM_BASE = "/sys/class/drm"
RUNNING = True
DEFAULT_PORT = 29501

# gpu_metrics binary offsets (format_revision=1, content_revision>=3)
HEADER_FMT = "<HBB"
HEADER_SIZE = struct.calcsize(HEADER_FMT)
XGMI_READ_OFFSET = 0x88
XGMI_WRITE_OFFSET = 0xC8
XGMI_LINK_SPEED_OFFSET = 0x60  # u16, Gbps per lane
XGMI_LINK_WIDTH_OFFSET = 0x62  # u16, lanes per link
NUM_XGMI_LINKS = 8
MIN_DATA_LEN = XGMI_WRITE_OFFSET + NUM_XGMI_LINKS * 8


# ── GPU discovery & counter reading ──────────────────────────────────────

def discover_gpus():
    """Find all GPUs with gpu_metrics, ordered by PCI BDF.

    Returns: [(gpu_idx, card_name, bdf, metrics_path), ...]
    """
    gpus = []
    for path in glob.glob(os.path.join(DRM_BASE, "card*/device/gpu_metrics")):
        card = path.split("/")[4]
        uevent_path = os.path.join(os.path.dirname(path), "uevent")
        bdf = ""
        try:
            with open(uevent_path) as f:
                for line in f:
                    if line.startswith("PCI_SLOT_NAME="):
                        bdf = line.strip().split("=", 1)[1]
                        break
        except OSError:
            pass
        gpus.append((card, bdf, path))
    gpus.sort(key=lambda x: x[1])
    return [(i, card, bdf, path) for i, (card, bdf, path) in enumerate(gpus)]


def _read_xgmi_link_speed(data):
    """Extract per-link bandwidth (GB/s) from gpu_metrics binary."""
    if len(data) < XGMI_LINK_WIDTH_OFFSET + 2:
        return 72.0  # default: 576 Gbps
    speed_gbps, width = struct.unpack_from("<HH", data, XGMI_LINK_SPEED_OFFSET)
    if speed_gbps > 0 and width > 0:
        return speed_gbps * width / 8.0  # Gbps * lanes / 8 -> GB/s
    return 72.0


_LINK_BW_CACHE = {}


def read_xgmi_counters(gpu_list):
    """Read xGMI accumulated counters from gpu_metrics binary.

    Returns: {dev_name: (total_read_kB, total_write_kB)}
    """
    global _LINK_BW_CACHE
    result = {}
    for gpu_idx, card, bdf, path in gpu_list:
        try:
            with open(path, "rb") as f:
                data = f.read()
        except OSError:
            continue
        if len(data) < MIN_DATA_LEN:
            continue

        _, fmt_rev, _ = struct.unpack_from(HEADER_FMT, data, 0)
        if fmt_rev != 1:
            continue

        reads = struct.unpack_from(f"<{NUM_XGMI_LINKS}Q", data, XGMI_READ_OFFSET)
        writes = struct.unpack_from(f"<{NUM_XGMI_LINKS}Q", data, XGMI_WRITE_OFFSET)

        dev = f"GPU{gpu_idx}"
        result[dev] = (sum(reads), sum(writes))

        if dev not in _LINK_BW_CACHE:
            _LINK_BW_CACHE[dev] = _read_xgmi_link_speed(data)

    return result


def compute_rates(counters, cache, ts):
    """Compute per-GPU xGMI bandwidth (bytes/s) from accumulated kB deltas."""
    rates = {}
    for dev, (read_kb, write_kb) in counters.items():
        if dev in cache:
            prev_read, prev_write, prev_ts = cache[dev]
            dt = ts - prev_ts
            if dt >= 0.05:
                read_rate = max(0, int((read_kb - prev_read) * 1024 / dt))
                write_rate = max(0, int((write_kb - prev_write) * 1024 / dt))
                link_bw = _LINK_BW_CACHE.get(dev, 72.0)
                max_bytes = int(link_bw * (NUM_XGMI_LINKS - 1) * 1e9)
                read_rate = min(read_rate, max_bytes)
                write_rate = min(write_rate, max_bytes)
                rates[dev] = (read_rate, write_rate)
            else:
                continue
        cache[dev] = (read_kb, write_kb, ts)
    return rates


# ── TCP helpers ──────────────────────────────────────────────────────────

def send_line(sock, msg):
    sock.sendall((msg + "\n").encode())


# ── Slave (--node N) ─────────────────────────────────────────────────────

def run_slave(args):
    global RUNNING
    node_id = args.node

    gpu_list = discover_gpus()
    hostname = socket.gethostname()
    dev_names = [f"GPU{i}" for i, *_ in gpu_list]
    print(f"[slave {node_id}] {hostname}, {len(gpu_list)} GPUs: {dev_names}",
          file=sys.stderr)

    print(f"[slave {node_id}] Connecting to {args.init_addr}:{args.port}...",
          file=sys.stderr)
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    max_retries = 30
    for attempt in range(max_retries):
        try:
            sock.settimeout(10)
            sock.connect((args.init_addr, args.port))
            break
        except (ConnectionRefusedError, OSError) as e:
            if attempt < max_retries - 1 and RUNNING:
                time.sleep(1)
                sock.close()
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            else:
                print(f"[slave {node_id}] Connection failed after {max_retries} "
                      f"attempts: {e}", file=sys.stderr)
                sys.exit(1)
    sock.setblocking(False)

    send_line(sock, f"HELLO:{node_id}:{','.join(dev_names)}:{hostname}")
    print(f"[slave {node_id}] Connected. Monitoring...", file=sys.stderr)

    recv_buf = bytearray()
    cache = {}

    def check_stop():
        try:
            data = sock.recv(65536)
            if not data:
                return True
            recv_buf.extend(data)
        except BlockingIOError:
            pass
        except OSError:
            return True
        while b"\n" in recv_buf:
            line, _, rest = recv_buf.partition(b"\n")
            recv_buf.clear()
            recv_buf.extend(rest)
            if line.decode().strip() == "STOP":
                return True
        return False

    start_time = time.time()
    try:
        while RUNNING:
            ts = time.time()
            counters = read_xgmi_counters(gpu_list)
            rates = compute_rates(counters, cache, ts)

            for dev, (read_rate, write_rate) in sorted(rates.items()):
                send_line(sock, f"DATA:{node_id}:{dev}:{read_rate}:{write_rate}")
            send_line(sock, f"TICK:{node_id}")

            elapsed = ts - start_time
            poll = min(0.01, args.interval / 2)
            sleep_target = start_time + ((elapsed // args.interval) + 1) * args.interval
            while RUNNING:
                if check_stop():
                    RUNNING = False
                    print(f"[slave {node_id}] Received STOP.", file=sys.stderr)
                    break
                remaining = sleep_target - time.time()
                if remaining <= 0:
                    break
                time.sleep(min(remaining, poll))
    except (BrokenPipeError, ConnectionResetError):
        print(f"[slave {node_id}] Master disconnected.", file=sys.stderr)

    try:
        send_line(sock, f"BYE:{node_id}")
    except OSError:
        pass
    sock.close()
    print(f"[slave {node_id}] Exiting.", file=sys.stderr)


# ── Master (--master) ────────────────────────────────────────────────────

def run_master(args):
    global RUNNING
    data_queue = queue.Queue()
    agent_socks = {}
    agent_socks_lock = threading.Lock()

    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind((args.init_addr, args.port))
    srv.listen(64)
    srv.settimeout(0.5)
    print(f"[master] Listening on {args.init_addr}:{args.port}", file=sys.stderr)

    def remote_receiver(conn, addr):
        buf = bytearray()
        conn.setblocking(False)
        remote_nid = None
        try:
            while RUNNING:
                try:
                    data = conn.recv(65536)
                    if not data:
                        break
                    buf.extend(data)
                except BlockingIOError:
                    time.sleep(0.005)
                    continue
                except OSError:
                    break
                while b"\n" in buf:
                    line, _, rest = buf.partition(b"\n")
                    buf.clear()
                    buf.extend(rest)
                    msg = line.decode()
                    if msg.startswith("HELLO:"):
                        parts = msg.split(":", 3)
                        remote_nid = int(parts[1])
                        dev_list = parts[2] if len(parts) > 2 else ""
                        hostname = parts[3] if len(parts) > 3 else f"node{remote_nid}"
                        with agent_socks_lock:
                            agent_socks[remote_nid] = conn
                        data_queue.put(("HELLO", remote_nid, dev_list, hostname))
                        print(f"[master] Slave {remote_nid} ({hostname}) connected, "
                              f"GPUs: {dev_list}", file=sys.stderr)
                    elif msg.startswith("DATA:"):
                        parts = msg.split(":")
                        if len(parts) == 5:
                            _, nid, dev, rx_s, tx_s = parts
                            data_queue.put(("DATA", int(nid), dev,
                                            int(rx_s), int(tx_s)))
                    elif msg.startswith("TICK:"):
                        data_queue.put(("TICK", int(msg.split(":")[1])))
                    elif msg.startswith("BYE:"):
                        data_queue.put(("BYE", int(msg.split(":")[1])))
                        break
        except Exception as e:
            print(f"[master] Receiver error: {e}", file=sys.stderr)
        finally:
            if remote_nid is not None:
                with agent_socks_lock:
                    agent_socks.pop(remote_nid, None)
            conn.close()

    def acceptor():
        while RUNNING:
            try:
                conn, addr = srv.accept()
                threading.Thread(target=remote_receiver, args=(conn, addr),
                                 daemon=True).start()
            except socket.timeout:
                continue
            except OSError:
                break

    threading.Thread(target=acceptor, daemon=True).start()

    trace_events = []
    base_ts = None
    pending_data = {}

    print(f"[master] Waiting for slaves... Ctrl-C to stop.\n", file=sys.stderr)

    while RUNNING:
        ts = time.time()
        if base_ts is None:
            base_ts = ts
        ts_us = int((ts - base_ts) * 1_000_000)

        while True:
            try:
                item = data_queue.get_nowait()
            except queue.Empty:
                break
            if item[0] == "HELLO":
                _, nid, dev_list, hostname = item
                trace_events.append({
                    "name": "process_name", "ph": "M", "pid": nid, "tid": 0,
                    "args": {"name": hostname},
                })
            elif item[0] == "DATA":
                _, nid, dev, read_rate, write_rate = item
                if nid not in pending_data:
                    pending_data[nid] = {}
                pending_data[nid][dev] = (read_rate, write_rate)
            elif item[0] == "BYE":
                _, nid = item
                print(f"[master] Slave {nid} disconnected.", file=sys.stderr)

        summary_parts = []
        for nid, dev_dict in sorted(pending_data.items()):
            for dev, (read_rate, write_rate) in sorted(dev_dict.items()):
                trace_events.append({
                    "name": f"{dev} read", "ph": "C", "ts": ts_us,
                    "pid": nid, "args": {"GB/s": read_rate / 1e9},
                })
                trace_events.append({
                    "name": f"{dev} write", "ph": "C", "ts": ts_us,
                    "pid": nid, "args": {"GB/s": write_rate / 1e9},
                })
                if read_rate > 0 or write_rate > 0:
                    r_gb = read_rate / 1e9
                    w_gb = write_rate / 1e9
                    summary_parts.append(
                        f"node{nid}/{dev} r={r_gb:.1f} w={w_gb:.1f} GB/s")
        pending_data.clear()

        elapsed = ts - (base_ts or ts)
        if summary_parts:
            print(f"[{elapsed:6.1f}s] {' | '.join(summary_parts)}", file=sys.stderr)

        poll = min(0.01, args.interval / 2)
        sleep_target = ts + args.interval
        while RUNNING:
            remaining = sleep_target - time.time()
            if remaining <= 0:
                break
            time.sleep(min(remaining, poll))

    # ── Shutdown ──────────────────────────────────────────────────────────
    print("\n[master] Sending STOP to slaves...", file=sys.stderr)
    with agent_socks_lock:
        for nid, conn in list(agent_socks.items()):
            try:
                send_line(conn, "STOP")
            except OSError:
                pass

    time.sleep(0.5)
    while True:
        try:
            item = data_queue.get_nowait()
            if item[0] == "DATA":
                _, nid, dev, read_rate, write_rate = item
                ts_us = int((time.time() - base_ts) * 1_000_000) if base_ts else 0
                trace_events.append({
                    "name": f"{dev} read", "ph": "C", "ts": ts_us,
                    "pid": nid, "args": {"GB/s": read_rate / 1e9},
                })
                trace_events.append({
                    "name": f"{dev} write", "ph": "C", "ts": ts_us,
                    "pid": nid, "args": {"GB/s": write_rate / 1e9},
                })
        except queue.Empty:
            break

    srv.close()

    trace = {"traceEvents": trace_events}
    with open(args.output, "w") as f:
        json.dump(trace, f)

    print(f"Wrote {len(trace_events)} events to {args.output}", file=sys.stderr)
    print(f"Open in chrome://tracing or https://ui.perfetto.dev/", file=sys.stderr)


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    global RUNNING

    parser = argparse.ArgumentParser(
        description="Multi-node xGMI traffic monitor (TCP master-slave)")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--master", action="store_true",
                       help="Run as master (aggregator + trace writer)")
    group.add_argument("--node", type=int, default=None,
                       help="Run as slave with given node ID")
    parser.add_argument("--init-addr", required=True,
                        help="TCP IP of master. Master listens, slaves connect.")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT,
                        help=f"TCP port (default: {DEFAULT_PORT})")
    parser.add_argument("--interval", type=float, default=1.0,
                        help="Sampling interval in seconds (default: 1)")
    parser.add_argument("--output", default="xgmi_trace.json",
                        help="Output Chrome Trace JSON file (master only)")
    args = parser.parse_args()

    def signal_handler(sig, frame):
        global RUNNING
        if not RUNNING:
            sys.exit(1)
        RUNNING = False
        print("\n[INFO] Ctrl-C received, stopping...", file=sys.stderr)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    if args.master:
        run_master(args)
    else:
        run_slave(args)


if __name__ == "__main__":
    main()
