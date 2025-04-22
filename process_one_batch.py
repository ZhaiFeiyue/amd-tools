import os


base = '/sgl-workspace'
files = os.listdir(base)

files = [f for f in files if 'bs' in f and 'il' in f and 'ol' in f]
tt = 'il128.ol1024.tp8dp1ep8'

result = {}
# bs1344.il128.ol1024.tp8dp1ep8.log
for f in files:
    if 'il128.ol1024.tp8dp1ep8' not in f:
        continue

    BS = int((f.split('.')[0]).split('bs')[1]) 
    f = os.path.join(base, f)

    with open(f, 'r') as fp:
        lines = fp.readlines()

    # output throughput: 3942.40 token/s
    # (input + output) throughput: 4435.20 token/s
    out_tps = 0
    total_tps = 0

    for l in lines:
        if 'output throughput:' in l:
            out_tps = l.split('output throughput: ')[-1].split(' ')[0]
        if '(input + output) throughput:' in l:
            total_tps = l.split('(input + output) throughput: ')[-1].split(' ')[0]

    out_tps = round(float(out_tps), 2)
    total_tps = round(float(total_tps), 2)
    result[BS] = (out_tps, total_tps)

print(result)

result = sorted(result.items(), key=lambda i: i[0])

with open('{}.csv'.format(tt), 'w') as fp:
    fp.write('bs,output_TPS, total_TPS\n')
    for r in result:
        fp.write('{},{},{}\n'.format(r[0], r[1][0], r[1][1]))
