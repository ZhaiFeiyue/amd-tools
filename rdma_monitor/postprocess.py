import matplotlib.pyplot as plt
from typing import List,Dict
file = "data.txt"

import os

def list_txt_files(folder_path):
    txt_files = []
    if not os.path.isdir(folder_path):
        raise ValueError(f" {folder_path}")
    
    for entry in os.listdir(folder_path):
        full_path = os.path.join(folder_path, entry)
        if os.path.isfile(full_path) and entry.lower().endswith('.txt'):
            txt_files.append(full_path)
            
    return txt_files

def post_process(data_file):
    prev_bytes = 0
    cur_bytes = 0
    timestamp0 = 0
    bw_timestamp = []
    with open(data_file,'r') as f:
        for line in f:
            timestamp,bytes = line.split(",")
            timestamp,bytes = int(timestamp),int(bytes)
            if cur_bytes == prev_bytes:
                continue
            else:
                gbs = (cur_bytes-prev_bytes) / (timestamp-timestamp0)
                timestamp0 = timestamp
                prev_bytes = cur_bytes
                bw_timestamp.append([timestamp0,gbs])
    min_timestamp = min([val[0] for val in bw_timestamp])
    for val in bw_timestamp:
        val[0] -= min_timestamp
        val[0] /= 1000000 # 转换毫秒

    processd_data_file = data_file + "_processed.txt"
    with open(processd_data_file,'w') as f:
        for val in bw_timestamp:
            if val[0] > 0 and val[1] > 0:
                f.write(str(val[0]) +","+ str(val[1]) + "\n") # timestamp,bw
    
    return processd_data_file

def read_data_file_to_list(data_file):
    s = []
    with open(data_file,'r') as f:
        for line in f:
            timestamp,gbs = line.split(",")
            timestamp,gbs = float(timestamp),float(gbs)
            s.append((timestamp,gbs))
    return s

all_data_file = list_txt_files()

processd_data_file = []

# 处理为结构化字典
all_nics_data:Dict[str,List[float,float]] = dict()
for data_file in all_data_file:
    all_nics_data[data_file] = read_data_file_to_list(post_process(data_file=data_file))

# 导出一张图像看看
fig, axes = plt.subplots(
    nrows=2, 
    ncols=4, 
    figsize=(16, 8),  
    dpi=100,  
    constrained_layout=True
)
axes = axes.flatten()
index = 0
for nic,data in all_nics_data.items():
    timestamp = [val[0] for val in data]
    bw = [val[1] for val in data]
    ax.plot(timestamp,bw, color='tab:blue', linewidth=2)
    ax = axes[index]
    ax.set_title(nic, fontsize=12)
    ax.set_xlabel("timestamp", fontsize=10)
    ax.set_ylabel("bw", fontsize=10)
    # ax.grid(alpha=0.3)
    # ax.set_xlim(0, 10)
    # ax.set_ylim(-8, 8)
    index += 1

fig.suptitle("2×4 Subplots Grid", fontsize=16, y=0.98)

# 6. 保存图像到本地
plt.savefig(
    "subplots_grid.png",  # 文件名
    bbox_inches='tight',  # 去除多余白边
    dpi=200              # 提高输出分辨率
)
