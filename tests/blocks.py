import os
import json

with open('MI308_decoder_beam3_trace_10times_0106.json', 'r') as fp:
    mi = json.load(fp)

out = {}

devices = []
for e in mi['traceEvents']:
    if 'cat' in e and e["cat"] != 'cpu_op':
        devices.append(e)

kernel = []
for e in devices:
    if "args" in e:
        args = e['args']
        if 'grid' in args and "blocks" in args:
            if e['name'] not in out

for e in kernel:
