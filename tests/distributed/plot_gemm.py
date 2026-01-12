import pandas as pd
import matplotlib.pyplot as plt

# Load the 4 CSV files into Pandas DataFrames
# Replace these paths with the actual paths to your CSV files
df_spx = pd.read_csv('spx_sweep_7168_16384_bf16_f_f.csv')  # Original SPX performance
df_cpx_splitM = pd.read_csv('cpx_sweep_7168_16384_bf16_f_f.csv')  # CPX split M
df_cpx_splitN = pd.read_csv('cpx_sweep_896_16384_bf16_f_f.csv')  # CPX split N
df_cpx_splitK = pd.read_csv('cpx_sweep_7168_2048_bf16_f_f.csv')  # CPX split K

# Function to plot performance against dimensions (M, N, K)
def plot_gemm_performance(df, label, is_split_M=False, is_cpx=False, plot_type='tflops'):
    # Use the M value for the x-axis
    df['dimension'] = df['M']
    if is_split_M:
        df['dimension'] = df['dimension'] * 8

    if is_cpx:
        df['tflops'] = df['tflops'] * 8

    # Choose whether to plot TFLOPS or latency (us)
    if plot_type == 'latency':
        plt.plot(df['dimension'], df['us'], label=label, marker='o')  # Plot latency (us)
    else:
        plt.plot(df['dimension'], df['tflops'], label=label, marker='o')  # Plot TFLOPS

# Create the plot for TFLOPS
plt.figure(figsize=(10, 6))

# Plot each dataset for TFLOPS
plot_gemm_performance(df_spx, 'SPX Mode', plot_type='tflops')
plot_gemm_performance(df_cpx_splitM, 'CPX Split M', is_cpx=True, is_split_M=True, plot_type='tflops')
plot_gemm_performance(df_cpx_splitN, 'CPX Split N', is_cpx=True, plot_type='tflops')
plot_gemm_performance(df_cpx_splitK, 'CPX Split K', is_cpx=True, plot_type='tflops')

# Add labels and title for TFLOPS
plt.xlabel('M (bs)')
plt.ylabel('TFLOPS')
plt.title('DSV3 Out GEMM (M=bs, N=7168, K=16384) Performance Comparison: SPX vs CPX Split M/N/K (TFLOPS)')
plt.legend()

# Save the plot as an image for TFLOPS
plt.grid(True)
plt.savefig('out_gemm_performance_comparison_tflops.png', dpi=300)

# Close the plot to avoid it from displaying
plt.close()

# Create the plot for Latency
plt.figure(figsize=(10, 6))

# Plot each dataset for Latency (us)
plot_gemm_performance(df_spx, 'SPX Mode', plot_type='latency')
plot_gemm_performance(df_cpx_splitM, 'CPX Split M', is_cpx=True, is_split_M=True, plot_type='latency')
plot_gemm_performance(df_cpx_splitN, 'CPX Split N', is_cpx=True, plot_type='latency')
plot_gemm_performance(df_cpx_splitK, 'CPX Split K', is_cpx=True, plot_type='latency')

# Add labels and title for Latency
plt.xlabel('M (bs)')
plt.ylabel('Latency (us)')
plt.title('DSV3 Out GEMM (M=bs, N=7168, K=16384) Performance Comparison: SPX vs CPX Split M/N/K (Latency)')
plt.legend()

# Save the plot as an image for Latency
plt.grid(True)
plt.savefig('out_gemm_performance_comparison_latency.png', dpi=300)

# Close the plot to avoid it from displaying
plt.close()
