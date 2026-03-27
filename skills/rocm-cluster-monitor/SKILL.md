---
name: rocm-cluster-monitor
description: >-
  Deploy a ROCm cluster monitoring web dashboard that collects GPU utilization,
  memory, temperature, power, host CPU/RAM, and Docker container GPU mapping
  from multiple AMD MI355X nodes via SSH. Use when the user wants to monitor
  ROCm cluster utilization, set up ROCm GPU dashboard, or deploy ROCm monitoring.
---

# ROCm Cluster Monitor

Deploy a web-based ROCm GPU monitoring dashboard on an AMD MI355X cluster. The server
runs on one node, pulls metrics from all GPU nodes via SSH every 60 seconds,
and serves a browser dashboard on port 8765.

## Prerequisites

Before starting, verify:
1. A **server node** that can SSH (passwordless) to all GPU nodes
2. All GPU nodes have `rocm-smi` and `docker` installed
3. Python 3.6+ on the server node (stdlib only, no pip install needed)

## Step 1: Verify Connectivity

Test SSH to every GPU node and confirm `rocm-smi` works:

```bash
for node in <NODE_LIST>; do
  echo -n "$node: "
  ssh -o ConnectTimeout=3 -o BatchMode=yes "$node" \
    "hostname && which rocm-smi && rocm-smi --showproductname 2>/dev/null | grep 'Card Series' | head -1" 2>&1 | tr '\n' ' '
  echo
done
```

If any node fails, fix SSH key auth (`ssh-copy-id`) before proceeding.

## Step 2: Create Files

Create directory `gpu_dashboard/` with 3 files. The source files are at:

- [server.py](server.py) — main server (SSH collection + HTTP dashboard)
- [dashboard.html](dashboard.html) — browser frontend with Chart.js
- [restart.sh](restart.sh) — cron restart script

### Customize server.py

Edit the `NODES` list at the top of `server.py`:

```python
NODES = [
    "gpu-node-01",
    "gpu-node-02",
    # ... add all GPU node hostnames
]
```

Other configurable constants:

| Variable | Default | Description |
|----------|---------|-------------|
| `SERVER_PORT` | 8765 | HTTP listen port |
| `COLLECT_INTERVAL` | 60 | Collection interval in seconds |
| `MAX_HISTORY` | 60 | History entries to keep (60 = 1 hour) |

### Customize restart.sh

Update the `cd` path to match the actual deployment location.

## Step 3: Start the Server

```bash
cd /path/to/gpu_dashboard
nohup python3 -u server.py > server.log 2>&1 &
```

Wait ~15 seconds, then verify:

```bash
tail -3 server.log
# Should show: [timestamp] Collected N/N nodes in XXXms

curl -s http://localhost:8765/api/status
# Should return JSON with last_run and errors: {}
```

## Step 4: Set Up Hourly Restart via Cron

```bash
chmod +x restart.sh
(crontab -l 2>/dev/null | grep -v "gpu_dashboard/restart.sh"; \
 echo "0 * * * * /path/to/gpu_dashboard/restart.sh") | crontab -
```

## Step 5: Access from Outside (SSH Port Forwarding)

If the datacenter only allows port 22:

```bash
ssh -i /path/to/key -N -L 9765:localhost:8765 user@server_ip
```

Then open `http://localhost:9765` in browser.

Notes:
- Windows PowerShell does not support `-f` (background), keep the terminal open
- If local port is taken, change `9765` to any available port
- `-N` means no remote command, forwarding only

## Architecture

```
Server Node (single process)
├── Collector Thread (every 60s)
│   └── ThreadPoolExecutor → SSH to all nodes in parallel
│       └── Single SSH call per node collects:
│           ├── rocm-smi: GPU util, mem, temp, power, VRAM
│           ├── /proc/meminfo + /proc/loadavg: host CPU/RAM
│           ├── rocm-smi --showpidgpus: PID-to-GPU mapping
│           └── docker ps + inspect + top: container info
│
└── ThreadingHTTPServer (:8765)
    ├── GET /           → dashboard.html
    ├── GET /api/data   → all node data + history
    ├── GET /api/status → collector status
    └── GET /api/server_info → server process metrics
```

## Dashboard Features

| Section | Content |
|---------|---------|
| Summary cards | Avg GPU util, avg VRAM, total power, active GPU count |
| History charts | Per-node GPU util and memory trends (Chart.js line charts) |
| Server process | PID, uptime, RSS memory, CPU%, threads, FDs, collect latency |
| Node cards | Per-GPU: utilization bar, memory bar, VRAM used/total, temp, power |
| Host resources | CPU util + load avg, RAM used/total, process count |
| Containers | Name, GPU tags (GPU0-GPU7), image, user, start date |

## Troubleshooting

| Problem | Fix |
|---------|-----|
| "Connection closed by UNKNOWN" | SSH server on target is broken, not a client issue |
| "empty output" for a node | `rocm-smi` not working on that node, check GPU driver |
| Port already in use | `fuser -k 8765/tcp` then restart |
| Dashboard loads but no data | Wait 60s for first collection cycle |
| API response slow | Reduce `MAX_HISTORY` or restart server |
| "Address already in use" on SSH tunnel | Local port taken, use a different port number |

## NVIDIA GPU Adaptation

Replace `rocm-smi` commands with `nvidia-smi` equivalents and update the regex
parsing in `ssh_collect()`. The core architecture stays identical.
