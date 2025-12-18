import pandas as pd
import numpy as np
import re
import os

def extract_density(dataset_name):
    match = re.search(r'd(\d+\.\d+e[+-]\d+)', dataset_name)
    if match:
        return float(match.group(1))
    return None

def calculate_correlation(df, time_col):
    densities = []
    times = []
    all_times = []
    
    for _, row in df.iterrows():
        density = extract_density(row['Dataset'])
        if density is not None and pd.notna(row[time_col]):
            all_times.append(row[time_col])
            if row[time_col] != 0.0:
                densities.append(density)
                times.append(row[time_col])
    
    if len(all_times) > 0 and all(t == 0.0 for t in all_times):
        return 0.0, len(all_times), "All values are 0.0 (no correlation possible)"
    
    if len(densities) < 2:
        return None, len(all_times) if len(all_times) > 0 else 0, "Insufficient non-zero data"
    
    if np.std(times) == 0:
        return None, len(densities), "Constant values (std=0)"
    
    try:
        correlation = np.corrcoef(densities, times)[0, 1]
        if np.isnan(correlation):
            return None, len(densities), "NaN correlation"
        return correlation, len(densities), None
    except:
        return None, len(densities), "Calculation error"

base_dir = os.path.dirname(os.path.abspath(__file__))

files = {
    'Sequential': os.path.join(base_dir, 'experiment_results_sequential.csv'),
    'MPI': os.path.join(base_dir, 'experiment_results_mpi.csv'),
    'CUDA': os.path.join(base_dir, 'experiment_results_cuda.csv')
}

algorithms = ['BF', 'BM', 'KMP', 'RK']

results = []

for method, filepath in files.items():
    if not os.path.exists(filepath):
        print(f"Warning: {filepath} not found")
        continue
    
    df = pd.read_csv(filepath)
    print(f"\n{method}:")
    print("=" * 60)
    
    for algo in algorithms:
        time_col = f'{algo}_Time(s)'
        matches_col = f'{algo}_Matches'
        
        if time_col not in df.columns:
            print(f"  {algo}: Column not found")
            continue
        
        if method == 'CUDA':
            time_col = matches_col
        
        correlation, n_samples, error_msg = calculate_correlation(df, time_col)
        
        if correlation is not None:
            results.append({
                'Method': method,
                'Algorithm': algo,
                'Correlation': correlation,
                'N_Samples': n_samples
            })
            print(f"  {algo:4s}: Correlation = {correlation:8.6f} (n={n_samples})")
        else:
            if error_msg:
                print(f"  {algo:4s}: Could not calculate - {error_msg}")
            else:
                print(f"  {algo:4s}: Could not calculate correlation")

results_df = pd.DataFrame(results)

print("\n" + "=" * 60)
print("Summary Table:")
print("=" * 60)
print(results_df.to_string(index=False))

output_file = os.path.join(base_dir, 'density_time_correlations.csv')
results_df.to_csv(output_file, index=False)
print(f"\nResults saved to: {output_file}")

print("\n" + "=" * 60)
print("Correlation Matrix by Method:")
print("=" * 60)

for method in ['Sequential', 'MPI', 'CUDA']:
    method_df = results_df[results_df['Method'] == method]
    if len(method_df) > 0:
        print(f"\n{method}:")
        for _, row in method_df.iterrows():
            print(f"  {row['Algorithm']:4s}: {row['Correlation']:8.6f}")

print("\n" + "=" * 60)
print("Correlation Matrix by Algorithm:")
print("=" * 60)

for algo in ['BF', 'BM', 'KMP', 'RK']:
    algo_df = results_df[results_df['Algorithm'] == algo]
    if len(algo_df) > 0:
        print(f"\n{algo}:")
        for _, row in algo_df.iterrows():
            print(f"  {row['Method']:10s}: {row['Correlation']:8.6f}")

