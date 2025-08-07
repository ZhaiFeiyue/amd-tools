import os
import sys
import datetime
import time

import tg4perfetto
import uuid
from tg4perfetto import TraceGenerator


BASE = "/sys/class/infiniband"

def get_all_rdma_mic():
    nics = os.listdir(BASE)
    return nics

def get_tx_rx(nic):
    p = f'{BASE}/{nic}/ports/1/hw_counters'
    
    with open(f"{p}/rx_bytes", "r") as f:
        rx = int(f.read().strip())
    with open(f"{p}/tx_bytes", "r") as f:
        tx = int(f.read().strip())
    return tx, rx


def get_bw_MB(e,s, interval):
    return (e - s)/interval

def main():
    try:
        file = sys.argv[1]
    except:
        file = str(uuid.uuid4())
    nics = get_all_rdma_mic()
    cache = {}
    
    running  = True
    interval = 0.5
    tgen = TraceGenerator(file)
    nic_to_count = {}
    for k in nics:
        pid = tgen.create_group(k)
        rx = pid.create_counter_track('rx')
        tx = pid.create_counter_track('tx')
        

        nic_to_count[k] =(pid, rx, tx) 

    while running:
        try:
            for k in nics:
                tx, rx = get_tx_rx(k)
                if k not in cache:
                    t = int(time.time() * 1000)
                    cache[k] = [tx, rx, t]
                    nic_to_count[k][1].count(t, 0)
                    nic_to_count[k][2].count(t, 0)
                    continue
    
                t = int(time.time() * 1000)
                duration = t - cache[k][2]
                cache[k][2] = t
                if tx != cache[k][0]:
                    rate = get_bw_MB(tx, cache[k][0], duration)
                    mb = int(rate / (1024 * 1024) * 1000)
                    print(mb)
                    nic_to_count[k][2].count(t, mb)
                    cache[k][0] = tx
    
                if rx != cache[k][1]:
                    rate = get_bw_MB(rx, cache[k][1], duration)
                    mb = int(rate / (1024 * 1024) * 1000)
                    print(mb)
                    nic_to_count[k][1].count(t, mb)
                    cache[k][1] = rx

            print(datetime.datetime.now())
            time.sleep(interval)
        except KeyboardInterrupt:
            print('stop ------------------')
            running = False

if __name__ == "__main__":
    main()
