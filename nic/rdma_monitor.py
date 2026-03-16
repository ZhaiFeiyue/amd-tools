#!/usr/bin/env python3
"""
Multi-node RDMA NIC traffic monitor (TCP master-slave).

Two modes:
  --master : TCP server, aggregates data from all slaves, writes Chrome Trace JSON.
  --node N : Slave, reads local counters, sends to master via TCP.

Designed to be launched by start.sh which handles SSH deployment and lifecycle.

Usage (via start.sh):
    ./start.sh --nodes node0,node1,node2 --interval 0.1 --init-addr 10.0.0.1 --output trace.json

Manual usage:
    # Master (on launch machine):
    python rdma_monitor.py --master --init-addr 10.0.0.1 --interval 0.1 --output trace.json
    # Slave (on each node):
    python rdma_monitor.py --node 0 --init-addr 10.0.0.1 --interval 0.1

All timestamps use master's clock. Output is Chrome Trace JSON viewable in
chrome://tracing or https://ui.perfetto.dev/
"""

import os
import sys
import json
import time
import socket
import signal
import argparse
import threading
import queue

IB_BASE = "/sys/class/infiniband"
NET_BASE = "/sys/class/net"
RUNNING = True
DEFAULT_PORT = 29500


# ── Counter reading (used by slaves) ─────────────────────────────────────

def _read_int(path):
    try:
        with open(path) as f:
            return int(f.read().strip())
    except (OSError, ValueError):
        return None


def discover_devices():
    if not os.path.exists(IB_BASE):
        return []
    names = []
    for dev in sorted(os.listdir(IB_BASE)):
        net_dir = os.path.join(IB_BASE, dev, "device", "net")
        if os.path.exists(net_dir):
            nets = os.listdir(net_dir)
            if nets:
                names.append(f"{dev}({nets[0]})")
                continue
        names.append(dev)
    return names


def read_counters():
    """Read RDMA traffic counters for all local IB devices.

    Priority: hw_counters/rx_rdma_ucast_bytes (ionic/RoCE)
           -> hw_counters/rx_bytes (traditional IB)
           -> net/statistics/rx_bytes (fallback)
    """
    result = {}
    if not os.path.exists(IB_BASE):
        return result
    for dev in sorted(os.listdir(IB_BASE)):
        hw = os.path.join(IB_BASE, dev, "ports", "1", "hw_counters")
        display = dev

        net_dir = os.path.join(IB_BASE, dev, "device", "net")
        if os.path.exists(net_dir):
            nets = os.listdir(net_dir)
            if nets:
                display = f"{dev}({nets[0]})"

        # ionic/RoCE: RDMA-specific counters (captures bypass traffic)
        rx = _read_int(os.path.join(hw, "rx_rdma_ucast_bytes"))
        tx = _read_int(os.path.join(hw, "tx_rdma_ucast_bytes"))
        if rx is not None and tx is not None:
            result[display] = (rx, tx)
            continue

        # Traditional IB hw_counters
        rx = _read_int(os.path.join(hw, "rx_bytes"))
        tx = _read_int(os.path.join(hw, "tx_bytes"))
        if rx is not None and tx is not None:
            result[display] = (rx, tx)
            continue

        # Fallback: kernel network statistics
        if os.path.exists(net_dir):
            nets = os.listdir(net_dir)
            if nets:
                stats = os.path.join(NET_BASE, nets[0], "statistics")
                rx = _read_int(os.path.join(stats, "rx_bytes"))
                tx = _read_int(os.path.join(stats, "tx_bytes"))
                if rx is not None and tx is not None:
                    result[display] = (rx, tx)
    return result


def _get_link_speed_bytes():
    """Return {netdev: max_bytes_per_sec} from sysfs link speed."""
    result = {}
    if not os.path.exists(NET_BASE):
        return result
    for nd in os.listdir(NET_BASE):
        speed_mbps = _read_int(os.path.join(NET_BASE, nd, "speed"))
        if speed_mbps and speed_mbps > 0:
            result[nd] = speed_mbps * 1_000_000 // 8
    return result


_LINK_SPEED_CACHE = {}


def compute_rates(counters, cache, ts):
    global _LINK_SPEED_CACHE
    if not _LINK_SPEED_CACHE:
        _LINK_SPEED_CACHE = _get_link_speed_bytes()

    rates = {}
    for dev, (rx, tx) in counters.items():
        if dev in cache:
            prev_rx, prev_tx, prev_ts = cache[dev]
            dt = ts - prev_ts
            if dt >= 0.05:
                rx_rate = max(0, int((rx - prev_rx) / dt))
                tx_rate = max(0, int((tx - prev_tx) / dt))
                # Cap at link speed to filter firmware counter batch artifacts
                net_dev = dev.split("(")[-1].rstrip(")") if "(" in dev else ""
                cap = _LINK_SPEED_CACHE.get(net_dev, 0)
                if cap > 0:
                    rx_rate = min(rx_rate, cap)
                    tx_rate = min(tx_rate, cap)
                rates[dev] = (rx_rate, tx_rate)
            else:
                continue
        cache[dev] = (rx, tx, ts)
    return rates


# ── TCP helpers ───────────────────────────────────────────────────────────

def send_line(sock, msg):
    sock.sendall((msg + "\n").encode())


# ── Slave (--node N) ─────────────────────────────────────────────────────

def run_slave(args):
    global RUNNING
    node_id = args.node

    devs = discover_devices()
    hostname = socket.gethostname()
    print(f"[slave {node_id}] {hostname}, devices: {devs}", file=sys.stderr)

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

    send_line(sock, f"HELLO:{node_id}:{','.join(devs)}:{hostname}")
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
            counters = read_counters()
            rates = compute_rates(counters, cache, ts)

            for dev, (rx_rate, tx_rate) in sorted(rates.items()):
                send_line(sock, f"DATA:{node_id}:{dev}:{rx_rate}:{tx_rate}")
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
                              f"devices: {dev_list}", file=sys.stderr)
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
                _, nid, dev, rx_rate, tx_rate = item
                if nid not in pending_data:
                    pending_data[nid] = {}
                pending_data[nid][dev] = (rx_rate, tx_rate)
            elif item[0] == "BYE":
                _, nid = item
                print(f"[master] Slave {nid} disconnected.", file=sys.stderr)

        summary_parts = []
        for nid, dev_dict in sorted(pending_data.items()):
            for dev, (rx_rate, tx_rate) in sorted(dev_dict.items()):
                trace_events.append({
                    "name": f"{dev} rx", "ph": "C", "ts": ts_us,
                    "pid": nid, "args": {"GB/s": rx_rate / 1e9},
                })
                trace_events.append({
                    "name": f"{dev} tx", "ph": "C", "ts": ts_us,
                    "pid": nid, "args": {"GB/s": tx_rate / 1e9},
                })
                if rx_rate > 0 or tx_rate > 0:
                    rx_mb = rx_rate / (1024 * 1024)
                    tx_mb = tx_rate / (1024 * 1024)
                    summary_parts.append(
                        f"node{nid}/{dev} rx={rx_mb:.1f}MB/s tx={tx_mb:.1f}MB/s")
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

    # ── Shutdown ─────────────────────────────────────────────────────────
    print("\n[master] Sending STOP to slaves...", file=sys.stderr)
    with agent_socks_lock:
        for nid, conn in list(agent_socks.items()):
            try:
                send_line(conn, "STOP")
            except OSError:
                pass

    # Drain remaining data
    time.sleep(0.5)
    while True:
        try:
            item = data_queue.get_nowait()
            if item[0] == "DATA":
                _, nid, dev, rx_rate, tx_rate = item
                ts_us = int((time.time() - base_ts) * 1_000_000) if base_ts else 0
                trace_events.append({
                    "name": f"{dev} rx", "ph": "C", "ts": ts_us,
                    "pid": nid, "args": {"GB/s": rx_rate / 1e9},
                })
                trace_events.append({
                    "name": f"{dev} tx", "ph": "C", "ts": ts_us,
                    "pid": nid, "args": {"GB/s": tx_rate / 1e9},
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
        description="Multi-node RDMA NIC traffic monitor (TCP master-slave)")
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
    parser.add_argument("--output", default="rdma_trace.json",
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
