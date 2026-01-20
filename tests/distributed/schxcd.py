

import os
import sys
import numpy as np

num_xcd = int(sys.argv[1])
start = 0
base = int(sys.argv[2])

xcds = np.array([i for i in range(start, start + 8*8)])

xcds = np.reshape(xcds, [8,8])


select_xcds = xcds[:,base: base+(num_xcd-1)//8+1]
select_xcds = select_xcds.T
select_xcds = np.reshape(select_xcds, [-1])
select_xcds = select_xcds.tolist()
r = ','.join([str(i) for i in select_xcds])
print(r)