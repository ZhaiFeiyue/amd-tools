import os
import sys
import datetime
import time

import signal
import tg4perfetto
import uuid


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
    results = {} 
    cache = {}
    
    running  = True
    interval = 1
    with tg4perfetto.open(file):
        while running:
            try:
                for k in nics:
                    tx, rx = get_tx_rx(k)
                    if k not in results:
                        results[k] = (tg4perfetto.count(f'{k}_tx'), tg4perfetto.count(f'{k}_rx'))
                        cache[k] = [tx, rx]
                        continue

        
                    if tx != cache[k][0]:
                        rate = get_bw_MB(tx, cache[k][0], interval)
                        mb = int(rate / (1024 * 1024))
                        results[k][0].count(mb)
                        cache[k][0] = tx
                    else:
                        results[k][0].count(0)
    
                    if rx != cache[k][1]:
                        rate = get_bw_MB(rx, cache[k][1], interval)
                        mb = int(rate / (1024 * 1024))
                        results[k][0].count(mb)
                        cache[k][1] = rx
                    else:
                        results[k][1].count(0)

                print(datetime.datetime.now())
                time.sleep(interval)
            except KeyboardInterrupt:
                print('stop ------------------')
                running = False

if __name__ == "__main__":
    main()
