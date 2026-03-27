#!/usr/bin/env python3
"""
GPU Cluster Monitor — single-process server that runs on the head node.

Pulls GPU metrics from all nodes via SSH every 60s and serves a web dashboard.
No remote agents needed.
"""

import json
import threading
import subprocess
import re
import time
import os
import resource
from http.server import HTTPServer, BaseHTTPRequestHandler
from socketserver import ThreadingMixIn
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor, as_completed

SERVER_PORT = 8765
COLLECT_INTERVAL = 60  # seconds
MAX_HISTORY = 60       # 1h at 1-min intervals

NODES = [
    "smci355-ccs-aus-n06-21",
    "smci355-ccs-aus-n08-21",
    "smci355-ccs-aus-n08-25",
    "smci355-ccs-aus-n08-29",
    "smci355-ccs-aus-n08-33",
    "smci355-ccs-aus-n09-21",
    "smci355-ccs-aus-n09-25",
    "smci355-ccs-aus-n09-29",
    "smci355-ccs-aus-n09-33",
]

data_lock = threading.Lock()
node_data = defaultdict(lambda: {"latest": None, "history": deque(maxlen=MAX_HISTORY)})
collector_status = {"last_run": None, "last_duration_ms": 0, "errors": {}}
server_start_time = time.time()

DASHBOARD_HTML_PATH = os.path.join(os.path.dirname(__file__), "dashboard.html")

# --------------- SSH GPU collection ---------------

COLLECT_CMD = (
    "rocm-smi --showuse --showmemuse --showtemp --showpower --showmeminfo vram 2>/dev/null;"
    "echo '===HOST_INFO===';"
    "grep 'MemTotal\\|MemAvailable' /proc/meminfo;"
    "cat /proc/loadavg;"
    "nproc;"
    "head -1 /proc/stat;"
    "echo '===DOCKER_GPU===';"
    "rocm-smi --showpidgpus 2>/dev/null;"
    "echo '===DOCKER_PS===';"
    "for cid in $(docker ps -q 2>/dev/null); do"
    "  pid=$(docker inspect --format '{{.State.Pid}}' $cid 2>/dev/null);"
    "  name=$(docker inspect --format '{{.Name}}' $cid 2>/dev/null | sed 's/^\\///');"
    "  img=$(docker inspect --format '{{.Config.Image}}' $cid 2>/dev/null);"
    "  started=$(docker inspect --format '{{.State.StartedAt}}' $cid 2>/dev/null | cut -dT -f1);"
    "  host_user=$(ps -o user= -p $pid 2>/dev/null | tr -d ' ');"
    "  echo \"CTR|${cid:0:12}|$name|$img|$pid|$host_user|$started\";"
    "  docker top $cid -eo pid 2>/dev/null | tail -n+2 | while read cpid; do"
    "    echo \"CTRPID|${cid:0:12}|$cpid\";"
    "  done;"
    "done"
)

def ssh_collect(hostname):
    """SSH into a node, collect GPU + host CPU/memory + docker/GPU mapping."""
    cmd = ["ssh", "-o", "ConnectTimeout=8", "-o", "BatchMode=yes", hostname, COLLECT_CMD]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        raw = result.stdout
    except Exception as e:
        return hostname, None, None, str(e)

    if not raw.strip():
        return hostname, None, None, "empty output"

    sections = re.split(r"===(\w+)===", raw)
    sec = {}
    for i in range(1, len(sections) - 1, 2):
        sec[sections[i]] = sections[i + 1]
    gpu_raw = sections[0] if sections else ""
    host_raw = sec.get("HOST_INFO", "")

    gpus = {}
    patterns = {
        "gpu_util":      r"GPU\[(\d+)\]\s*:\s*GPU use \(%\):\s*(\d+)",
        "temp_junction":  r"GPU\[(\d+)\]\s*:\s*Temperature \(Sensor junction\).*?:\s*([\d.]+)",
        "temp_memory":    r"GPU\[(\d+)\]\s*:\s*Temperature \(Sensor memory\).*?:\s*([\d.]+)",
        "power_w":        r"GPU\[(\d+)\]\s*:\s*Current Socket Graphics Package Power \(W\):\s*([\d.]+)",
        "mem_util_pct":   r"GPU\[(\d+)\]\s*:\s*GPU Memory Allocated \(VRAM%\):\s*(\d+)",
        "mem_rw_pct":     r"GPU\[(\d+)\]\s*:\s*GPU Memory Read/Write Activity \(%\):\s*(\d+)",
        "vram_total_bytes": r"GPU\[(\d+)\]\s*:\s*VRAM Total Memory \(B\):\s*(\d+)",
        "vram_used_bytes":  r"GPU\[(\d+)\]\s*:\s*VRAM Total Used Memory \(B\):\s*(\d+)",
    }
    for key, pattern in patterns.items():
        for m in re.finditer(pattern, gpu_raw):
            gid = int(m.group(1))
            gpus.setdefault(gid, {})
            gpus[gid][key] = float(m.group(2))

    gpu_list = []
    for gid in sorted(gpus):
        g = gpus[gid]
        gpu_list.append({
            "id": gid,
            "gpu_util": g.get("gpu_util", 0),
            "mem_util_pct": g.get("mem_util_pct", 0),
            "mem_rw_pct": g.get("mem_rw_pct", 0),
            "vram_used_bytes": g.get("vram_used_bytes", 0),
            "vram_total_bytes": g.get("vram_total_bytes", 0),
            "temp_junction": g.get("temp_junction", 0),
            "temp_memory": g.get("temp_memory", 0),
            "power_w": g.get("power_w", 0),
        })

    host_info = _parse_host_info(host_raw)
    docker_gpu = _parse_docker_gpu(sec.get("DOCKER_GPU", ""), sec.get("DOCKER_PS", ""))
    host_info["containers"] = docker_gpu
    return hostname, gpu_list, host_info, None


def _parse_host_info(raw):
    info = {}
    try:
        lines = raw.strip().splitlines()
        for line in lines:
            if line.startswith("MemTotal:"):
                info["mem_total_kb"] = int(line.split()[1])
            elif line.startswith("MemAvailable:"):
                info["mem_avail_kb"] = int(line.split()[1])

        for line in lines:
            parts = line.split()
            if len(parts) >= 4 and "." in parts[0] and "/" in parts[3]:
                info["load_1"] = float(parts[0])
                info["load_5"] = float(parts[1])
                info["load_15"] = float(parts[2])
                procs = parts[3].split("/")
                info["procs_running"] = int(procs[0])
                info["procs_total"] = int(procs[1])
                break

        for line in lines:
            stripped = line.strip()
            if stripped.isdigit():
                info["cpu_cores"] = int(stripped)
                break

        for line in lines:
            if line.startswith("cpu "):
                fields = [int(x) for x in line.split()[1:]]
                total = sum(fields)
                idle = fields[3] + (fields[4] if len(fields) > 4 else 0)
                info["cpu_total_ticks"] = total
                info["cpu_idle_ticks"] = idle
                break
    except Exception:
        pass

    mem_total = info.get("mem_total_kb", 0)
    mem_avail = info.get("mem_avail_kb", 0)
    mem_used = mem_total - mem_avail
    info["containers"] = []
    info["mem_total_gb"] = round(mem_total / (1024 * 1024), 1)
    info["mem_used_gb"] = round(mem_used / (1024 * 1024), 1)
    info["mem_used_pct"] = round(mem_used / mem_total * 100, 1) if mem_total else 0

    cores = info.get("cpu_cores", 1)
    info["cpu_util_pct"] = round(info.get("load_1", 0) / cores * 100, 1)

    return info


def _parse_docker_gpu(gpu_pid_raw, docker_ps_raw):
    pid_to_gpus = {}
    current_pid = None
    for line in gpu_pid_raw.splitlines():
        m = re.match(r"PID (\d+) is using \d+ DRM device", line)
        if m:
            current_pid = int(m.group(1))
            pid_to_gpus.setdefault(current_pid, [])
            continue
        stripped = line.strip()
        if stripped.isdigit() and current_pid is not None:
            pid_to_gpus.setdefault(current_pid, []).append(int(stripped))

    containers = {}
    ctr_pids = {}
    for line in docker_ps_raw.splitlines():
        parts = line.strip().split("|")
        if len(parts) >= 7 and parts[0] == "CTR":
            cid = parts[1]
            containers[cid] = {
                "id": cid,
                "name": parts[2],
                "image": parts[3].split("/")[-1][:60],
                "main_pid": int(parts[4]) if parts[4].isdigit() else 0,
                "user": parts[5] or "unknown",
                "started": parts[6],
                "gpus": [],
            }
        elif len(parts) >= 3 and parts[0] == "CTRPID":
            cid = parts[1]
            pid = int(parts[2]) if parts[2].strip().isdigit() else 0
            if pid:
                ctr_pids.setdefault(cid, []).append(pid)

    for cid, pids in ctr_pids.items():
        if cid not in containers:
            continue
        gpu_set = set()
        for pid in pids:
            for g in pid_to_gpus.get(pid, []):
                gpu_set.add(g)
        containers[cid]["gpus"] = sorted(gpu_set)

    return list(containers.values())


def collect_all():
    """Collect from all nodes in parallel via SSH."""
    t0 = time.time()
    ts = time.time()
    ts_str = time.strftime("%Y-%m-%d %H:%M:%S")
    errors = {}

    with ThreadPoolExecutor(max_workers=len(NODES)) as pool:
        futures = {pool.submit(ssh_collect, h): h for h in NODES}
        for fut in as_completed(futures):
            hostname, gpu_list, host_info, err = fut.result()
            if err:
                errors[hostname] = err
                print(f"  [WARN] {hostname}: {err}")
                continue
            payload = {
                "hostname": hostname,
                "timestamp": ts,
                "timestamp_str": ts_str,
                "gpu_count": len(gpu_list),
                "gpus": gpu_list,
                "host": host_info or {},
            }
            with data_lock:
                node_data[hostname]["latest"] = payload
                node_data[hostname]["history"].append({
                    "timestamp": ts,
                    "gpus": gpu_list,
                    "host": host_info or {},
                })

    duration_ms = int((time.time() - t0) * 1000)
    with data_lock:
        collector_status["last_run"] = ts_str
        collector_status["last_duration_ms"] = duration_ms
        collector_status["errors"] = errors

    ok_count = len(NODES) - len(errors)
    print(f"[{ts_str}] Collected {ok_count}/{len(NODES)} nodes in {duration_ms}ms")


def collector_loop():
    """Background thread: collect every COLLECT_INTERVAL seconds."""
    while True:
        try:
            collect_all()
        except Exception as e:
            print(f"[ERROR] collector: {e}")
        time.sleep(COLLECT_INTERVAL)

# --------------- Server self-monitoring ---------------

def get_server_info():
    pid = os.getpid()
    uptime_s = time.time() - server_start_time

    rss_bytes = 0
    cpu_pct = 0.0
    num_threads = 0
    num_fds = 0
    try:
        with open(f"/proc/{pid}/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    rss_bytes = int(line.split()[1]) * 1024
                elif line.startswith("Threads:"):
                    num_threads = int(line.split()[1])
        with open(f"/proc/{pid}/stat") as f:
            parts = f.read().split()
            utime = int(parts[13])
            stime = int(parts[14])
            total_ticks = utime + stime
            hz = os.sysconf(os.sysconf_names["SC_CLK_TCK"])
            cpu_seconds = total_ticks / hz
            cpu_pct = (cpu_seconds / uptime_s) * 100 if uptime_s > 0 else 0
        num_fds = len(os.listdir(f"/proc/{pid}/fd"))
    except Exception:
        pass

    ru = resource.getrusage(resource.RUSAGE_SELF)

    with data_lock:
        history_total = sum(len(nd["history"]) for nd in node_data.values())
        node_count = len(node_data)
        last_run = collector_status.get("last_run")
        last_dur = collector_status.get("last_duration_ms", 0)
        errors = collector_status.get("errors", {})

    return {
        "pid": pid,
        "hostname": os.uname().nodename.split(".")[0],
        "uptime_s": round(uptime_s),
        "uptime_str": _fmt_duration(uptime_s),
        "rss_mb": round(rss_bytes / (1024 * 1024), 1),
        "cpu_pct": round(cpu_pct, 2),
        "threads": num_threads,
        "open_fds": num_fds,
        "max_rss_mb": round(ru.ru_maxrss / 1024, 1),
        "user_time_s": round(ru.ru_utime, 2),
        "sys_time_s": round(ru.ru_stime, 2),
        "voluntary_ctx_switches": ru.ru_nvcsw,
        "involuntary_ctx_switches": ru.ru_nivcsw,
        "node_count": node_count,
        "history_entries": history_total,
        "last_collect": last_run,
        "last_collect_ms": last_dur,
        "collect_errors": errors,
        "timestamp": time.time(),
    }


def _fmt_duration(s):
    d = int(s) // 86400
    h = (int(s) % 86400) // 3600
    m = (int(s) % 3600) // 60
    parts = []
    if d: parts.append(f"{d}d")
    if h: parts.append(f"{h}h")
    parts.append(f"{m}m")
    return " ".join(parts)


# --------------- HTTP server ---------------

class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        pass  # suppress per-request logs

    def _cors(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")

    def _json(self, data, code=200):
        body = json.dumps(data).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self._cors()
        self.end_headers()
        self.wfile.write(body)

    def _html(self, html):
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self._cors()
        self.end_headers()
        self.wfile.write(html.encode())

    def do_OPTIONS(self):
        self.send_response(204)
        self._cors()
        self.end_headers()

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            try:
                with open(DASHBOARD_HTML_PATH) as f:
                    self._html(f.read())
            except FileNotFoundError:
                self._json({"error": "dashboard.html not found"}, 404)

        elif self.path == "/api/data":
            with data_lock:
                result = {}
                for hostname, nd in node_data.items():
                    result[hostname] = {
                        "latest": nd["latest"],
                        "history": list(nd["history"]),
                    }
            self._json(result)

        elif self.path == "/api/status":
            with data_lock:
                self._json(collector_status)

        elif self.path == "/api/server_info":
            self._json(get_server_info())

        else:
            self.send_response(404)
            self.end_headers()


class ThreadingHTTPServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True


def main():
    print(f"GPU Cluster Monitor")
    print(f"  Nodes:     {', '.join(NODES)}")
    print(f"  Interval:  {COLLECT_INTERVAL}s")
    print(f"  Dashboard: http://0.0.0.0:{SERVER_PORT}")
    print()

    t = threading.Thread(target=collector_loop, daemon=True)
    t.start()

    server = ThreadingHTTPServer(("0.0.0.0", SERVER_PORT), Handler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down.")
        server.shutdown()


if __name__ == "__main__":
    main()
