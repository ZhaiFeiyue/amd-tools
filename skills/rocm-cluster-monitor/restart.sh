#!/usr/bin/env bash
# Restart GPU dashboard server — called by cron every hour
pkill -f "python3 -u server.py" 2>/dev/null
sleep 2
cd /apps/feiyue/gpu_dashboard && nohup python3 -u server.py > /apps/feiyue/gpu_dashboard/server.log 2>&1 &
echo "$(date): restarted server, PID=$!" >> /apps/feiyue/gpu_dashboard/restart.log
