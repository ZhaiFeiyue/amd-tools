import matplotlib.pyplot as plt
import pandas as pd
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman"]

data = pd.read_csv(r"C:\vboxcode\amd-tools\rdma_monitor\data_bnxt_re_bond3_rx_processed.txt_16K-16MB-main.txt")
bw = data.iloc[:,1].to_numpy() 
ts = data.iloc[:,0].to_numpy() / 1000 / 1000 # ms

plt.figure(figsize=(10, 6))

plt.plot(ts, bw, 
         color='black', 
         linewidth=0.8)

plt.grid(True, 
         linestyle='--',  
         linewidth=0.5, 
         alpha=1.0,     
         color='lightgray')

plt.title("Bandwidth Time Series", fontsize=14)
plt.xlabel("Time(ms)", fontsize=12)
plt.ylabel("Bandwidth(GB/s)", fontsize=12)


plt.tight_layout()

plt.savefig('bandwidth_plot.png', dpi=600, bbox_inches='tight')
plt.show()
