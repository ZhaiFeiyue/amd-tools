import sys
import os
import matplotlib.pyplot as plt
import pandas as pd
import logging
logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)

plt.rcParams['font.family'] = ['serif', 'sans-serif', 'monospace']

if len(sys.argv) < 2:
    print("Usage: python visualize.py <input_file_path>")
    sys.exit(1)

input_file = sys.argv[1]

try:
    data = pd.read_csv(input_file)
except FileNotFoundError:
    print(f"Error: The file '{input_file}' was not found.")
    sys.exit(1)
except Exception as e:
    print(f"An error occurred while reading the file: {e}")
    sys.exit(1)

# This splits the file path into a root and an extension, then replaces the extension with '.png'
output_file = os.path.splitext(input_file)[0] + '.png'

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

plt.savefig(output_file, dpi=600, bbox_inches='tight')
plt.show()
