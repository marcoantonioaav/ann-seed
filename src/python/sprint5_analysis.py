import os
import glob
import pandas as pd
import matplotlib.pyplot as plt

def generate_plots_and_tables(dataset_name, input_dir, output_dir):
    csv_files = glob.glob(f"{input_dir}/results_{dataset_name}_*.csv")
    if not csv_files:
        print(f"No CSV files found for {dataset_name} in {input_dir}")
        return
        
    all_dfs = []
    for f in csv_files:
        approach = os.path.basename(f).replace(f"results_{dataset_name}_", "").replace(".csv", "")
        df = pd.read_csv(f)
        df['ApproachFamily'] = approach
        all_dfs.append(df)
        
    full_df = pd.concat(all_dfs, ignore_index=True)
    full_df['QPS'] = 1000.0 / full_df['Search Time (ms/q)']
    
    # 1. Plots
    os.makedirs(output_dir, exist_ok=True)
    
    def plot_metric_vs_qps(metric, filename):
        plt.figure(figsize=(10, 6))
        for family in full_df['ApproachFamily'].unique():
            subset = full_df[full_df['ApproachFamily'] == family].sort_values('QPS')
            if not subset.empty:
                plt.plot(subset['QPS'], subset[metric], marker='o', label=family)
        
        plt.xlabel("QPS (Queries per Second)")
        plt.ylabel(metric)
        plt.title(f"{metric} vs QPS ({dataset_name})")
        plt.legend()
        plt.grid(True)
        # Using log scale for QPS usually makes sense for these plots
        plt.xscale('log')
        plt.tight_layout()
        plt.savefig(f"{output_dir}/{filename}")
        plt.close()

    plot_metric_vs_qps('Recall@1', f"plot_{dataset_name}_recall1_vs_qps.png")
    plot_metric_vs_qps('Recall@100', f"plot_{dataset_name}_recall100_vs_qps.png")
    plot_metric_vs_qps('Mean Dist', f"plot_{dataset_name}_meandist_vs_qps.png")
    plot_metric_vs_qps('1-NN Diff', f"plot_{dataset_name}_1nndiff_vs_qps.png")
    
    # 2. Tables
    # Select parameter combination that achieves closest Search Time to 1 ms/q (1000 QPS)
    # The prompt says "closest Search Time to 1 second". We'll assume this means 1.0 ms/q 
    # or 1000 ms/q. Let's find the one closest to 1.0 ms/q.
    best_rows = []
    for family in full_df['ApproachFamily'].unique():
        subset = full_df[full_df['ApproachFamily'] == family].copy()
        subset['Diff'] = abs(subset['Search Time (ms/q)'] - 1.0)
        best_row = subset.loc[subset['Diff'].idxmin()]
        best_rows.append(best_row)
        
    best_df = pd.DataFrame(best_rows)
    
    # Offline metrics table
    offline_cols = ['Parameter Setup', 'Build Time (s)', 'Memory Footprint (MB)', 'Index Size (MB)']
    offline_df = best_df[offline_cols]
    
    # Online metrics table
    online_cols = ['Parameter Setup', 'Search Time (ms/q)', 'QPS', 'Dist Comps', 'Recall@1', 'Recall@100', 'Mean Dist', '1-NN Diff']
    online_df = best_df[online_cols]
    
    print(f"=== {dataset_name} Offline Metrics ===")
    print(offline_df.to_string(index=False))
    print(f"\n=== {dataset_name} Online Metrics ===")
    print(online_df.to_string(index=False))
    print("\n")
    
    with open(f"{output_dir}/{dataset_name}_tables.tex", "w") as f:
        f.write(f"% Offline Metrics for {dataset_name}\n")
        f.write(offline_df.to_latex(index=False))
        f.write("\n\n")
        f.write(f"% Online Metrics for {dataset_name}\n")
        f.write(online_df.to_latex(index=False))

if __name__ == "__main__":
    output_dir = "results"
    generate_plots_and_tables("MSMARCO", output_dir, output_dir)
    generate_plots_and_tables("NQ", output_dir, output_dir)
