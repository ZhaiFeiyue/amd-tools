import os
import numpy as np
import sys

gpus = int(sys.argv[1])
xcds_per_gpu = int(sys.argv[2])

required = 2 ** int(sys.argv[3])

total_gpus = [i for i in range(0, gpus * xcds_per_gpu)]
total_gpus = np.reshape(total_gpus, [gpus, xcds_per_gpu])

select_gpus = []
for i in range(required):
    row_idx = i % gpus
    col_idx = i // gpus
    select_gpus.append(total_gpus[row_idx, col_idx])
r = ','.join([str(i) for i in select_gpus])
print(r)