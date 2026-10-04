import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def load_dataset_results(dataset_name, input_dir):
    csv_files = glob.glob(f"{input_dir}/results_{dataset_name}_*.csv")
    if not csv_files:
        print(f"No CSV files found for {dataset_name} in {input_dir}")
        return pd.DataFrame()
        
    all_dfs = []
    for f in csv_files:
        approach = os.path.basename(f).replace(f"results_{dataset_name}_", "").replace(".csv", "")
        approach = approach.replace("_", " ")
        approach = approach.replace("t KD-Trees", "KD-trees").replace("t K-Means", "K-Means trees")
        df = pd.read_csv(f)
        df['ApproachFamily'] = approach
        all_dfs.append(df)
        
    full_df = pd.concat(all_dfs, ignore_index=True)
    full_df['QPS'] = 1000.0 / full_df['Search Time (ms/q)']
    full_df = full_df.rename(columns={'Mean Dist': 'Mean Distance', '1-NN Diff': '1-NN Difference'})
    return full_df

def print_tables(full_df, dataset_name, output_dir):
    if full_df.empty: return
    best_rows = []
    for family in full_df['ApproachFamily'].unique():
        subset = full_df[full_df['ApproachFamily'] == family].copy()
        subset['Diff'] = abs(subset['Search Time (ms/q)'] - 1.0)
        best_row = subset.loc[subset['Diff'].idxmin()]
        best_rows.append(best_row)
        
    best_df = pd.DataFrame(best_rows)
    best_df = best_df.sort_values('1-NN Difference', ascending=False)
    best_df['Approach'] = best_df['ApproachFamily']
    
    offline_cols = ['Approach', 'Build Time (s)', 'Memory Footprint (MB)', 'Index Size (MB)']
    offline_df = best_df[offline_cols]
    
    online_cols = ['Approach', 'Search Time (ms/q)', 'Dist Comps', 'Recall@1', 'Recall@100', 'Mean Distance', '1-NN Difference']
    online_df = best_df[online_cols]
    
    direction_dict = {
        'Build Time (s)': 'min',
        'Memory Footprint (MB)': 'min',
        'Index Size (MB)': 'min',
        'Search Time (ms/q)': 'min',
        'Dist Comps': 'min',
        'Recall@1': 'max',
        'Recall@100': 'max',
        'Mean Distance': 'min',
        '1-NN Difference': 'min'
    }
    
    format_dict = {
        'Build Time (s)': '{:.2f}',
        'Memory Footprint (MB)': '{:.2f}',
        'Index Size (MB)': '{:.2f}',
        'Search Time (ms/q)': '{:.4f}',
        'Dist Comps': '{:.2f}',
        'Recall@1': '{:.4f}',
        'Recall@100': '{:.4f}',
        'Mean Distance': '{:.4f}',
        '1-NN Difference': '{:.4f}'
    }

    def to_formatted_latex(df):
        df_fmt = df.copy()
        for col in df.columns:
            if col in direction_dict:
                def fmt(x, c=col):
                    if pd.isna(x): return ""
                    return format_dict[c].format(x) if c in format_dict else str(x)
                df_fmt[col] = df[col].apply(fmt)
                
        new_cols = []
        for col in df.columns:
            if col == 'Approach':
                new_cols.append("\\textbf{Approach}")
                continue
            arrow = "$\\uparrow$" if direction_dict.get(col) == 'max' else "$\\downarrow$"
            words = col.split(' ', 1)
            if len(words) == 2:
                header = f"\\textbf{{\\makecell{{{words[0]} \\\\ {words[1]} {arrow}}}}}"
            else:
                header = f"\\textbf{{{col} {arrow}}}"
            new_cols.append(header)
        df_fmt.columns = new_cols
        col_format = 'l' + 'r' * (len(df.columns) - 1)
        return df_fmt.to_latex(index=False, escape=False, column_format=col_format)

    print(f"=== {dataset_name} Offline Metrics ===")
    print(offline_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"\n=== {dataset_name} Online Metrics ===")
    print(online_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("\n")
    
    with open(f"{output_dir}/{dataset_name}_tables.tex", "w") as f:
        f.write(f"% Offline Metrics for {dataset_name}\n")
        f.write(to_formatted_latex(offline_df))
        f.write("\n\n")
        f.write(f"% Online Metrics for {dataset_name}\n")
        f.write(to_formatted_latex(online_df))

def filter_pareto_frontier(subset, metric_col):
    """
    Extracts Pareto optimal points for trade-off plots to prevent duplicate X-axis vertical line artifacts.
    """
    if subset.empty:
        return subset

    df_copy = subset.copy()
    df_copy['metric_rounded'] = df_copy[metric_col].round(4)
    is_recall = metric_col.startswith('Recall')
    
    if is_recall:
        grouped = df_copy.groupby('metric_rounded', as_index=False).agg({'QPS': 'max'})
        grouped = grouped.merge(df_copy, on=['metric_rounded', 'QPS'], how='inner').drop_duplicates(subset=['metric_rounded'])
        grouped = grouped.sort_values(metric_col, ascending=True)

        pareto_rows = []
        max_qps_seen = float('inf')
        for _, row in grouped.iterrows():
            if row['QPS'] <= max_qps_seen:
                pareto_rows.append(row)
                max_qps_seen = row['QPS']
        return pd.DataFrame(pareto_rows)
    else:
        grouped = df_copy.groupby('metric_rounded', as_index=False).agg({'QPS': 'max'})
        grouped = grouped.merge(df_copy, on=['metric_rounded', 'QPS'], how='inner').drop_duplicates(subset=['metric_rounded'])
        grouped = grouped.sort_values(metric_col, ascending=True)

        pareto_rows = []
        max_qps_seen = float('-inf')
        for _, row in grouped.iterrows():
            if row['QPS'] >= max_qps_seen:
                pareto_rows.append(row)
                max_qps_seen = row['QPS']
        return pd.DataFrame(pareto_rows)

def compute_robust_series_limits(series, quantile_low=0.0, quantile_high=0.90, default_min=None, default_max=None, is_log=False):
    """Computes robust plot limits using IQR / quantiles to filter extreme outliers."""
    s = series.dropna()
    if s.empty:
        return default_min, default_max
    
    if is_log:
        s = s[s > 0]
        if s.empty:
            return default_min, default_max
        log_vals = np.log10(s)
        q_low, q_high = log_vals.quantile(quantile_low), log_vals.quantile(quantile_high)
        iqr = q_high - q_low
        margin = 0.1 * iqr if iqr > 0 else 0.3
        min_val = 10 ** (q_low - margin)
        max_val = 10 ** (q_high + margin)
    else:
        q25 = s.quantile(0.25)
        q75 = s.quantile(0.75)
        iqr = q75 - q25
        
        if iqr > 0:
            upper_bound = min(s.quantile(quantile_high), q75 + 1.5 * iqr)
            lower_bound = max(s.min(), q25 - 1.5 * iqr)
        else:
            lower_bound = s.min()
            upper_bound = s.quantile(quantile_high)
            
        margin = 0.05 * (upper_bound - lower_bound) if (upper_bound - lower_bound) > 0 else (0.05 * abs(upper_bound) if upper_bound != 0 else 0.05)
        min_val = lower_bound - margin
        max_val = upper_bound + margin

    if default_min is not None:
        min_val = max(min_val, default_min)
    if default_max is not None:
        max_val = min(max_val, default_max)

    return min_val, max_val

def adjust_limits_to_visible_points(plotted_vals, lim_min, lim_max, default_min=None, default_max=None, max_allowed_gap_ratio=0.10, is_log=False):
    """
    Checks if there is a gap between visible points and the axis limits (lim_min, lim_max),
    and tightens the limits closer to the visible point distribution to avoid out-of-focus plots.
    Supports linear and log10 scale axes.
    """
    if not plotted_vals:
        return lim_min, lim_max

    if is_log:
        valid_vals = [v for v in plotted_vals if v > 0]
        if not valid_vals:
            return lim_min, lim_max

        visible_vals = [v for v in valid_vals if (lim_min is None or v >= lim_min * 0.999) and (lim_max is None or v <= lim_max * 1.001)]
        if not visible_vals:
            visible_vals = valid_vals

        log_vis = np.log10(visible_vals)
        vis_min_log = min(log_vis)
        vis_max_log = max(log_vis)
        vis_range_log = vis_max_log - vis_min_log

        if vis_range_log <= 0:
            vis_range_log = 0.5

        new_min = lim_min
        new_max = lim_max

        if lim_min is not None and lim_min > 0:
            lim_min_log = np.log10(lim_min)
            left_gap = vis_min_log - lim_min_log
            if left_gap > max_allowed_gap_ratio * vis_range_log:
                adjusted_log = vis_min_log - 0.05 * vis_range_log
                if default_min is not None and default_min > 0:
                    adjusted_log = max(adjusted_log, np.log10(default_min))
                new_min = 10 ** adjusted_log

        if lim_max is not None and lim_max > 0:
            lim_max_log = np.log10(lim_max)
            right_gap = lim_max_log - vis_max_log
            if right_gap > max_allowed_gap_ratio * vis_range_log:
                adjusted_log = vis_max_log + 0.05 * vis_range_log
                if default_max is not None and default_max > 0:
                    adjusted_log = min(adjusted_log, np.log10(default_max))
                new_max = 10 ** adjusted_log

        return new_min, new_max
    else:
        visible_vals = [v for v in plotted_vals if (lim_min is None or v >= lim_min - 1e-9) and (lim_max is None or v <= lim_max + 1e-9)]
        if not visible_vals:
            visible_vals = plotted_vals

        vis_min = min(visible_vals)
        vis_max = max(visible_vals)
        vis_range = vis_max - vis_min

        if vis_range <= 0:
            vis_range = abs(vis_min) if vis_min != 0 else 1.0

        new_min = lim_min
        new_max = lim_max

        if lim_min is not None:
            left_gap = vis_min - lim_min
            if left_gap > max_allowed_gap_ratio * vis_range:
                adjusted = vis_min - 0.05 * vis_range
                if default_min is not None:
                    adjusted = max(adjusted, default_min)
                new_min = adjusted

        if lim_max is not None:
            right_gap = lim_max - vis_max
            if right_gap > max_allowed_gap_ratio * vis_range:
                adjusted = vis_max + 0.05 * vis_range
                if default_max is not None:
                    adjusted = min(adjusted, default_max)
                new_max = adjusted

        return new_min, new_max

def generate_combined_plots(df_msmarco, df_nq, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    plots_dir = os.path.join(output_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    
    target_metrics = ['Recall@1', 'Recall@100', 'Mean Distance', '1-NN Difference']
    
    all_dfs = [df for df in [df_msmarco, df_nq] if not df.empty]
    if not all_dfs:
        return

    all_approaches = set()
    for df in all_dfs:
        all_approaches.update(df['ApproachFamily'].unique())
    # Exclude Medoid approach
    all_approaches = {app for app in all_approaches if app.lower() != 'medoid'}
    sorted_approaches = sorted(all_approaches)

    markers = ['o', 's', '^', 'v', 'D', 'p', '*', 'X']
    linestyles = ['-', '--', '-.', ':']
    colors = plt.cm.tab10.colors
    
    styles = {}
    for i, approach in enumerate(sorted_approaches):
        styles[approach] = {
            'marker': markers[i % len(markers)], 
            'linestyle': linestyles[i % len(linestyles)],
            'color': colors[i % len(colors)]
        }

    scenarios_data = [("MS MARCO", df_msmarco), ("NQ", df_nq)]

    for metric_col in target_metrics:
        qps_series = [df['QPS'] for _, df in scenarios_data if not df.empty]
        ylim_min, ylim_max = compute_robust_series_limits(pd.concat(qps_series), quantile_low=0.01, quantile_high=0.99, is_log=True)

        all_plotted_y_vals = []
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 2.6), sharey=True)
        axes_list = [ax1, ax2]

        for idx, (dataset_name, df) in enumerate(scenarios_data):
            ax = axes_list[idx]
            letter = "(a)" if idx == 0 else "(b)"
            formatted_title = f"{letter} {dataset_name}"

            if df.empty:
                continue

            if metric_col.startswith('Recall'):
                xlim_min, xlim_max = compute_robust_series_limits(df[metric_col], quantile_low=0.0, quantile_high=1.0, default_min=0.0, default_max=1.02)
            else:
                xlim_min, xlim_max = compute_robust_series_limits(df[metric_col], quantile_low=0.0, quantile_high=0.90, default_min=0.0)

            plotted_x_vals = []
            for family in sorted_approaches:
                if family.lower() == 'medoid':
                    continue
                subset = df[df['ApproachFamily'] == family]
                if not subset.empty:
                    pareto_subset = filter_pareto_frontier(subset, metric_col)
                    if not pareto_subset.empty:
                        ax.plot(
                            pareto_subset[metric_col],
                            pareto_subset['QPS'],
                            marker=styles[family]['marker'],
                            linestyle=styles[family]['linestyle'],
                            color=styles[family]['color'],
                            label=family,
                            markersize=4.0,
                            linewidth=1.1
                        )
                        plotted_x_vals.extend(pareto_subset[metric_col].dropna().tolist())
                        all_plotted_y_vals.extend(pareto_subset['QPS'].dropna().tolist())

            if plotted_x_vals:
                def_min = 0.0 if (metric_col.startswith('Recall') or metric_col in ['Mean Distance', '1-NN Difference']) else None
                def_max = 1.02 if metric_col.startswith('Recall') else None
                xlim_min, xlim_max = adjust_limits_to_visible_points(
                    plotted_x_vals, xlim_min, xlim_max, default_min=def_min, default_max=def_max
                )

            ax.set_title(formatted_title, fontsize=8.5, fontweight='bold', pad=3)
            ax.set_yscale('log')
            if xlim_min is not None and xlim_max is not None:
                ax.set_xlim(xlim_min, xlim_max)

            ax.tick_params(axis='both', which='major', labelsize=7.5)
            ax.grid(True, which='both', linestyle='--', alpha=0.4, linewidth=0.5)

            ax.set_xlabel(metric_col, fontsize=8.0, labelpad=2)
            if idx == 0:
                ax.set_ylabel("QPS", fontsize=8.0, labelpad=2)

        if all_plotted_y_vals:
            ylim_min, ylim_max = adjust_limits_to_visible_points(
                all_plotted_y_vals, ylim_min, ylim_max, is_log=True
            )

        for ax in axes_list:
            ax.set_ylim(ylim_min, ylim_max)

            ax.tick_params(axis='both', which='major', labelsize=7.5)
            ax.grid(True, which='both', linestyle='--', alpha=0.4, linewidth=0.5)

            ax.set_xlabel(metric_col, fontsize=8.0, labelpad=2)
            if idx == 0:
                ax.set_ylabel("QPS", fontsize=8.0, labelpad=2)

        unique_labels = {}
        for ax in axes_list:
            h, l = ax.get_legend_handles_labels()
            for handle, label in zip(h, l):
                if label not in unique_labels:
                    unique_labels[label] = handle

        if unique_labels:
            ncol = min(len(unique_labels), 9)
            fig.legend(
                unique_labels.values(),
                unique_labels.keys(),
                loc='upper center',
                bbox_to_anchor=(0.5, 1.15),
                ncol=ncol,
                fontsize=7.5,
                frameon=True,
                handletextpad=0.3,
                columnspacing=0.8
            )

        plt.tight_layout()
        plt.subplots_adjust(top=0.85, wspace=0.18)

        base_filename = f"plot_combined_{metric_col.replace('@', '').replace(' ', '').lower()}_vs_qps"

        plt.savefig(os.path.join(output_dir, f"{base_filename}.png"), dpi=300, bbox_inches='tight')
        plt.savefig(os.path.join(output_dir, f"{base_filename}.pdf"), bbox_inches='tight')

        plt.savefig(os.path.join(plots_dir, f"{base_filename}.png"), dpi=300, bbox_inches='tight')
        plt.savefig(os.path.join(plots_dir, f"{base_filename}.pdf"), bbox_inches='tight')

        plt.close()

if __name__ == "__main__":
    output_dir = "results"
    df_msmarco = load_dataset_results("MSMARCO", output_dir)
    df_nq = load_dataset_results("NQ", output_dir)
    
    print_tables(df_msmarco, "MSMARCO", output_dir)
    print_tables(df_nq, "NQ", output_dir)
    
    generate_combined_plots(df_msmarco, df_nq, output_dir)
