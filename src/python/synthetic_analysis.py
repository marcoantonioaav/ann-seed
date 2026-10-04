import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import math
import numpy as np

def load_scenario_results(input_dir, scenario_name):
    csv_files = glob.glob(f"{input_dir}/results_{scenario_name}_*.csv")
    if not csv_files:
        print(f"No CSV files found for scenario '{scenario_name}' in {input_dir}")
        return pd.DataFrame()
        
    all_dfs = []
    for f in csv_files:
        df = pd.read_csv(f)
        all_dfs.append(df)
        
    full_df = pd.concat(all_dfs, ignore_index=True)
    full_df['QPS'] = 1000.0 / full_df['Search Time (ms/q)']
    full_df = full_df.rename(columns={'Mean Dist': 'Mean Distance', '1-NN Diff': '1-NN Difference'})
    return full_df

def separate_approaches(df):
    """
    Separates trivial single-comparison approaches (Medoid and Random sample=1)
    from multi-comparison approaches (all others, including Random sample > 1).
    """
    if df.empty:
        return pd.DataFrame(), pd.DataFrame()
        
    # Trivial filter: Medoid OR Random (sample=1)
    is_medoid = df['ApproachFamily'] == 'Medoid'
    is_random_r1 = (df['ApproachFamily'] == 'Random') & (df['Approach'].str.contains(r"sample=1\)", regex=True))
    
    trivial_mask = is_medoid | is_random_r1
    
    trivial_df = df[trivial_mask].copy()
    multicomp_df = df[~trivial_mask].copy()
    
    return trivial_df, multicomp_df

def filter_pareto_frontier(subset, metric_col):
    """
    Extracts Pareto optimal points for trade-off plots to prevent duplicate X-axis vertical line artifacts.
    - For Recall metrics (higher is better): keeps maximum QPS for each recall, ensuring strictly increasing recall with non-increasing QPS.
    - For Distance/Difference metrics (lower is better): keeps maximum QPS for each distance level.
    """
    if subset.empty:
        return subset

    # Round metric values to 4 decimal places to group equivalent evaluation outcomes
    df_copy = subset.copy()
    df_copy['metric_rounded'] = df_copy[metric_col].round(4)

    is_recall = metric_col.startswith('Recall')
    
    if is_recall:
        # For equal recall, pick the highest QPS (most efficient configuration)
        grouped = df_copy.groupby('metric_rounded', as_index=False).agg({'QPS': 'max'})
        grouped = grouped.merge(df_copy, on=['metric_rounded', 'QPS'], how='inner').drop_duplicates(subset=['metric_rounded'])
        grouped = grouped.sort_values(metric_col, ascending=True)

        # Monotonic Pareto filter: higher recall should not have higher QPS than lower recall
        pareto_rows = []
        max_qps_seen = float('inf')
        for _, row in grouped.iterrows():
            if row['QPS'] <= max_qps_seen:
                pareto_rows.append(row)
                max_qps_seen = row['QPS']
        return pd.DataFrame(pareto_rows)
    else:
        # For Distance / Difference: lower distance is better
        grouped = df_copy.groupby('metric_rounded', as_index=False).agg({'QPS': 'max'})
        grouped = grouped.merge(df_copy, on=['metric_rounded', 'QPS'], how='inner').drop_duplicates(subset=['metric_rounded'])
        grouped = grouped.sort_values(metric_col, ascending=True)

        # Monotonic Pareto filter: lower distance (better quality) requires lower/equal QPS
        pareto_rows = []
        max_qps_seen = float('-inf')
        for _, row in grouped.iterrows():
            if row['QPS'] >= max_qps_seen:
                pareto_rows.append(row)
                max_qps_seen = row['QPS']
        return pd.DataFrame(pareto_rows)

def to_formatted_latex(df):
    if df.empty:
        return "% Empty Dataframe\n"
        
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

    df_fmt = df.copy()
    for col in df.columns:
        if col in direction_dict:
            def fmt(x, c=col):
                if pd.isna(x): return ""
                return format_dict[c].format(x) if c in format_dict else str(x)
            df_fmt[col] = df[col].apply(fmt)
            
    new_cols = []
    text_col_count = 0
    num_col_count = 0
    for col in df.columns:
        if col not in direction_dict:
            new_cols.append(f"\\textbf{{{col}}}")
            text_col_count += 1
            continue
        num_col_count += 1
        arrow = "$\\uparrow$" if direction_dict.get(col) == 'max' else "$\\downarrow$"
        words = col.split(' ', 1)
        if len(words) == 2:
            header = f"\\textbf{{\\makecell{{{words[0]} \\\\ {words[1]} {arrow}}}}}"
        else:
            header = f"\\textbf{{{col} {arrow}}}"
        new_cols.append(header)
    df_fmt.columns = new_cols
    col_format = 'l' * text_col_count + 'r' * num_col_count
    return df_fmt.to_latex(index=False, escape=False, column_format=col_format)

def print_and_save_tables(trivial_df, multicomp_df, scenario_name, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Print and save Trivial Approaches Tables (No graphs needed)
    if not trivial_df.empty:
        print(f"=== {scenario_name} - Trivial (Single-Comparison) Approaches ===")
        print(trivial_df[['Approach', 'Search Time (ms/q)', 'Dist Comps', 'Recall@1', 'Recall@100', 'Mean Distance', '1-NN Difference']].to_string(index=False))
        print("\n")
        
        with open(f"{output_dir}/table_trivial_{scenario_name}.tex", "w") as f:
            f.write(f"% Trivial (Single-Comparison) Approaches for {scenario_name}\n")
            f.write(to_formatted_latex(trivial_df[['Approach', 'Search Time (ms/q)', 'Dist Comps', 'Recall@1', 'Recall@100', 'Mean Distance', '1-NN Difference']]))
            f.write("\n")

    # 2. Print and save Multi-comparison Approaches Tables
    if not multicomp_df.empty:
        best_rows = []
        for family in multicomp_df['ApproachFamily'].unique():
            subset = multicomp_df[multicomp_df['ApproachFamily'] == family].copy()
            max_rec = subset['Recall@1'].max()
            max_rec_subset = subset[np.isclose(subset['Recall@1'], max_rec, atol=1e-5)]
            best_row = max_rec_subset.loc[max_rec_subset['Search Time (ms/q)'].idxmin()]
            best_rows.append(best_row)
        best_df = pd.DataFrame(best_rows).sort_values('Recall@1', ascending=False)
        
        print(f"=== {scenario_name} - Multi-Comparison Approaches Summary ===")
        print(best_df[['Approach', 'Search Time (ms/q)', 'Dist Comps', 'Recall@1', 'Recall@100', 'Mean Distance', '1-NN Difference']].to_string(index=False))
        print("\n")
        
        with open(f"{output_dir}/table_multicomp_{scenario_name}.tex", "w") as f:
            f.write(f"% Multi-Comparison Approaches for {scenario_name}\n")
            f.write(to_formatted_latex(best_df[['Approach', 'Search Time (ms/q)', 'Dist Comps', 'Recall@1', 'Recall@100', 'Mean Distance', '1-NN Difference']]))
            f.write("\n")

def format_scenario_title(data_type, query_type):
    """Formats scenario into readable string e.g., 'Clustered | Out-Of-Distribution'."""
    data_str = "Clustered" if data_type == "clustered" else "Non-Clustered"
    query_str = "In-Dist" if query_type == "in_dist" else "Out-Dist"
    return f"{data_str} | {query_str}"

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

def generate_combined_metric_plots(all_scenario_dfs, metric_name, output_dir):
    """
    Generates paper-ready row plots:
    - Shared Y-axis (QPS log scale) across all plots in the row
    - INDEPENDENT X-axis limits for each plot tailored to its own scenario point distribution
    - Pareto frontier filtering to eliminate vertical line artifacts and duplicate points
    """
    if not all_scenario_dfs:
        return

    plots_dir = os.path.join(output_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    target_metrics = ['Recall@1', 'Recall@100', 'Mean Distance', '1-NN Difference']
    scenarios_data = []

    desired_order = [("uniform", "in_dist"), ("uniform", "out_dist"), ("clustered", "in_dist"), ("clustered", "out_dist")]

    for data_type, query_type in desired_order:
        key = (metric_name, data_type, query_type)
        if key in all_scenario_dfs:
            df = all_scenario_dfs[key]
            _, multicomp_df = separate_approaches(df)
            if not multicomp_df.empty:
                scenarios_data.append((key, multicomp_df))

    if not scenarios_data:
        return

    all_approaches = set()
    for _, df in scenarios_data:
        all_approaches.update(df['ApproachFamily'].unique())
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

    letters = ['(a)', '(b)', '(c)', '(d)']

    for metric_col in target_metrics:
        qps_series = [df['QPS'] for _, df in scenarios_data]
        ylim_min, ylim_max = compute_robust_series_limits(pd.concat(qps_series), quantile_low=0.01, quantile_high=0.99, is_log=True)

        all_plotted_y_vals = []
        num_scenarios = len(scenarios_data)
        fig, axes = plt.subplots(1, num_scenarios, figsize=(4.0 * num_scenarios, 3.0), sharey=True)
        if num_scenarios == 1:
            axes = [axes]

        for idx, (scenario_tuple, df) in enumerate(scenarios_data):
            ax = axes[idx]
            _, data_type, query_type = scenario_tuple
            letter = letters[idx] if idx < len(letters) else f"({chr(97 + idx)})"
            formatted_title = f"{letter} {format_scenario_title(data_type, query_type)}"

            if metric_col.startswith('Recall'):
                xlim_min, xlim_max = compute_robust_series_limits(df[metric_col], quantile_low=0.0, quantile_high=1.0, default_min=0.0, default_max=1.02)
            else:
                xlim_min, xlim_max = compute_robust_series_limits(df[metric_col], quantile_low=0.0, quantile_high=0.90, default_min=0.0)

            plotted_x_vals = []
            for family in sorted_approaches:
                subset = df[df['ApproachFamily'] == family]
                if not subset.empty:
                    # Apply Pareto frontier filter to eliminate vertical line artifacts and duplicate X points
                    pareto_subset = filter_pareto_frontier(subset, metric_col)
                    if not pareto_subset.empty:
                        ax.plot(
                            pareto_subset[metric_col],
                            pareto_subset['QPS'],
                            marker=styles[family]['marker'],
                            linestyle=styles[family]['linestyle'],
                            color=styles[family]['color'],
                            label=family,
                            markersize=6.0,
                            linewidth=1.5
                        )
                        plotted_x_vals.extend(pareto_subset[metric_col].dropna().tolist())
                        all_plotted_y_vals.extend(pareto_subset['QPS'].dropna().tolist())

            if plotted_x_vals:
                def_min = 0.0 if (metric_col.startswith('Recall') or metric_col in ['Mean Distance', '1-NN Difference']) else None
                def_max = 1.02 if metric_col.startswith('Recall') else None
                xlim_min, xlim_max = adjust_limits_to_visible_points(
                    plotted_x_vals, xlim_min, xlim_max, default_min=def_min, default_max=def_max
                )

            ax.set_title(formatted_title, fontsize=13.0, fontweight='bold', pad=5)
            ax.set_yscale('log')
            if xlim_min is not None and xlim_max is not None:
                ax.set_xlim(xlim_min, xlim_max)

            ax.tick_params(axis='both', which='major', labelsize=11.5)
            ax.tick_params(axis='both', which='minor', labelsize=10.0)
            ax.grid(True, which='both', linestyle='--', alpha=0.4, linewidth=0.5)

            ax.set_xlabel(metric_col, fontsize=13.0, labelpad=4)
            if idx == 0:
                ax.set_ylabel("QPS (log scale)", fontsize=13.0, labelpad=5)

        if all_plotted_y_vals:
            ylim_min, ylim_max = adjust_limits_to_visible_points(
                all_plotted_y_vals, ylim_min, ylim_max, is_log=True
            )

        for ax in axes:
            ax.set_ylim(ylim_min, ylim_max)

        unique_labels = {}
        for ax in axes:
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
                bbox_to_anchor=(0.5, 1.18),
                ncol=ncol,
                fontsize=11.5,
                frameon=True,
                handletextpad=0.5,
                columnspacing=1.1
            )

        plt.subplots_adjust(top=0.81, bottom=0.19, left=0.07, right=0.985, wspace=0.09)
        
        filename = f"plot_row_paper_{metric_name}_{metric_col.replace('@', '').replace(' ', '').lower()}_vs_qps.png"
        plt.savefig(os.path.join(plots_dir, filename), dpi=300, bbox_inches='tight')
        plt.savefig(os.path.join(plots_dir, filename.replace('.png', '.pdf')), bbox_inches='tight')
        plt.close()

def generate_unified_trivial_table(all_scenario_dfs, output_dir):
    """
    Generates a single unified LaTeX table for trivial single-comparison approaches
    (Medoid and Random sample=1) across all 8 synthetic scenarios.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    desired_order = [
        ("cosine", "uniform", "in_dist"),
        ("cosine", "uniform", "out_dist"),
        ("cosine", "clustered", "in_dist"),
        ("cosine", "clustered", "out_dist"),
        ("euclidean", "uniform", "in_dist"),
        ("euclidean", "uniform", "out_dist"),
        ("euclidean", "clustered", "in_dist"),
        ("euclidean", "clustered", "out_dist"),
    ]
    
    trivial_rows = []
    for m, d, q in desired_order:
        scenario_key = f"{m}_{d}_{q}"
        df = all_scenario_dfs.get(scenario_key)
        if df is None or df.empty:
            continue
            
        trivial_df, _ = separate_approaches(df)
        if trivial_df.empty:
            continue
            
        scenario_label = f"{m.capitalize()} | {format_scenario_title(d, q)}"
        sub_df = trivial_df[['Approach', 'Search Time (ms/q)', 'Dist Comps', 'Recall@1', 'Recall@100', 'Mean Distance', '1-NN Difference']].copy()
        sub_df.insert(0, 'Scenario', scenario_label)
        trivial_rows.append(sub_df)
        
    if not trivial_rows:
        return
        
    unified_trivial_df = pd.concat(trivial_rows, ignore_index=True)
    
    print("=== Unified Trivial (Single-Comparison) Approaches Table ===")
    print(unified_trivial_df.to_string(index=False))
    print("\n")
    
    tex_path = os.path.join(output_dir, "table_trivial_unified.tex")
    with open(tex_path, "w") as f:
        f.write("% Unified Trivial (Single-Comparison) Approaches across Synthetic Scenarios\n")
        f.write(to_formatted_latex(unified_trivial_df))
        f.write("\n")

def generate_offline_tables(all_scenario_dfs, output_dir):
    """
    Generates an aggregated LaTeX table for offline index construction metrics
    (Build Time, Memory Footprint, Index Size) per approach across synthetic datasets.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    base_datasets = [
        ("cosine", "uniform"),
        ("cosine", "clustered"),
        ("euclidean", "uniform"),
        ("euclidean", "clustered"),
    ]
    
    offline_rows = []
    for m, d in base_datasets:
        scenario_key = f"{m}_{d}_in_dist"
        df = all_scenario_dfs.get(scenario_key)
        if df is None or df.empty:
            continue
            
        dataset_label = f"{m.capitalize()} {'Clustered' if d == 'clustered' else 'Uniform'}"
        
        unique_families = df['ApproachFamily'].unique()
        for family in unique_families:
            family_subset = df[df['ApproachFamily'] == family]
            first_row = family_subset.iloc[0]
            
            offline_rows.append({
                'Dataset': dataset_label,
                'Approach': family,
                'Build Time (s)': first_row['Build Time (s)'],
                'Memory Footprint (MB)': first_row['Memory Footprint (MB)'],
                'Index Size (MB)': first_row['Index Size (MB)']
            })
            
    if not offline_rows:
        return
        
    offline_df = pd.DataFrame(offline_rows)
    
    print("=== Aggregated Synthetic Offline Metrics Table ===")
    print(offline_df.to_string(index=False))
    print("\n")
    
    tex_path = os.path.join(output_dir, "table_offline_synthetic.tex")
    with open(tex_path, "w") as f:
        f.write("% Offline Metrics for Synthetic Experiments\n")
        f.write(to_formatted_latex(offline_df))
        f.write("\n")

def main():
    parser = argparse.ArgumentParser(description="Analyze synthetic experiment results.")
    parser.add_argument("--input-dir", type=str, default="results_synthetic", help="Input directory containing CSV results")
    parser.add_argument("--output-dir", type=str, default="results_synthetic/analysis", help="Output directory for tables and plots")
    args = parser.parse_args()

    metrics = ["cosine", "euclidean"]
    data_types = ["uniform", "clustered"]
    query_types = ["in_dist", "out_dist"]

    all_scenario_dfs = {}

    for m in metrics:
        metric_scenarios = {}
        for d in data_types:
            for q in query_types:
                scenario_name = f"{m}_{d}_{q}"
                df = load_scenario_results(args.input_dir, scenario_name)
                
                if df.empty:
                    continue
                    
                trivial_df, multicomp_df = separate_approaches(df)
                print_and_save_tables(trivial_df, multicomp_df, scenario_name, args.output_dir)
                metric_scenarios[(m, d, q)] = df
                all_scenario_dfs[scenario_name] = df

        if metric_scenarios:
            generate_combined_metric_plots(metric_scenarios, m, args.output_dir)

    if all_scenario_dfs:
        generate_unified_trivial_table(all_scenario_dfs, args.output_dir)
        generate_offline_tables(all_scenario_dfs, args.output_dir)

if __name__ == "__main__":
    main()
