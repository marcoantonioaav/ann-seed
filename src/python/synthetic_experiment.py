import numpy as np
import h5py
import pandas as pd
from tqdm import tqdm
import os
import gc
import json
import argparse
import sys
import warnings

from metrics import PerformanceTracker, calculate_recall_at_k, calculate_mean_distance, calculate_1nn_distance_diff

try:
    import initialization_cpp
except ImportError:
    print("Warning: initialization_cpp not found.")
    initialization_cpp = None

ALL_METRICS = ["cosine", "euclidean"]
ALL_DATA_TYPES = ["uniform", "clustered"]
ALL_QUERY_TYPES = ["in_dist", "out_dist"]
ALL_APPROACHES = ["VP-tree", "Stacked NSW", "LSH", "t KD-Trees", "t K-Means", "Random", "Medoid", "HVS", "LSB-Tree"]

def run_experiment_on_scenario(metric, data_type, query_type, target_approach, data_dir, output_dir):
    scenario_name = f"{metric}_{data_type}_{query_type}"
    h5_file = os.path.join(data_dir, f"synthetic_{scenario_name}.h5")
    
    if not os.path.exists(h5_file):
        print(f"Warning: Dataset file {h5_file} does not exist. Skipping scenario {scenario_name}. (Run generate_synthetic_data.py first)")
        return
        
    print(f"\n=== Running Synthetic Experiment: Scenario={scenario_name}, Approach={target_approach} ===")
    
    with h5py.File(h5_file, 'r') as f:
        dataset = f['dataset'][:]
        queries = f['queries'][:]
        ground_truth = f['ground_truth'][:].tolist()

    k_search = 100
    load_size = len(dataset)
    
    # Approach definitions matching initialization_cpp constructors
    all_approaches = {
        "VP-tree": {
            "constructor": lambda: initialization_cpp.VPTreeInit(1000, 1.0, 1.0, metric),
            "query_params": [
                ("VP-tree (max_leaves=100)", {"max_leaves_to_visit": "100"}),
                ("VP-tree (max_leaves=250)", {"max_leaves_to_visit": "250"}),
                ("VP-tree (max_leaves=500)", {"max_leaves_to_visit": "500"}),
                ("VP-tree (max_leaves=1000)", {"max_leaves_to_visit": "1000"}),
                ("VP-tree (max_leaves=2000)", {"max_leaves_to_visit": "2000"})
            ]
        },
        "Stacked NSW": {
            "constructor": lambda: initialization_cpp.StackedNSWInit(16, 200, 10, metric),
            "query_params": [
                ("Stacked NSW (ef=10)", {"ef": "10"}),
                ("Stacked NSW (ef=50)", {"ef": "50"}),
                ("Stacked NSW (ef=100)", {"ef": "100"}),
                ("Stacked NSW (ef=200)", {"ef": "200"}),
                ("Stacked NSW (ef=500)", {"ef": "500"})
            ]
        },
        "LSH": {
            "constructor": lambda: initialization_cpp.LSHInit(10, 16, 10, metric),
            "query_params": [
                ("LSH (probes=10)", {"num_probes": "10"}),
                ("LSH (probes=20)", {"num_probes": "20"}),
                ("LSH (probes=50)", {"num_probes": "50"}),
                ("LSH (probes=100)", {"num_probes": "100"}),
                ("LSH (probes=200)", {"num_probes": "200"})
            ]
        },
        "t KD-Trees": {
            "constructor": lambda: initialization_cpp.FlannKDTreeInit(4, 100, metric),
            "query_params": [
                ("t KD-Trees (checks=100)", {"checks": "100"}),
                ("t KD-Trees (checks=500)", {"checks": "500"}),
                ("t KD-Trees (checks=1000)", {"checks": "1000"}),
                ("t KD-Trees (checks=2000)", {"checks": "2000"}),
                ("t KD-Trees (checks=5000)", {"checks": "5000"})
            ]
        },
        "t K-Means": {
            "constructor": lambda: initialization_cpp.FlannKMeansInit(1, 16, 2, 100, metric),
            "query_params": [
                ("t K-Means (checks=100)", {"checks": "100"}),
                ("t K-Means (checks=500)", {"checks": "500"}),
                ("t K-Means (checks=1000)", {"checks": "1000"}),
                ("t K-Means (checks=2000)", {"checks": "2000"}),
                ("t K-Means (checks=5000)", {"checks": "5000"})
            ]
        },
        "Random": {
            "constructor": lambda: initialization_cpp.RandomPointsInit(42, metric),
            "query_params": [
                ("Random Points (sample=1)", {"sample_size": "1"}),
                ("Random Points (sample=100)", {"sample_size": "100"}),
                ("Random Points (sample=1000)", {"sample_size": "1000"}),
                ("Random Points (sample=2000)", {"sample_size": "2000"}),
                ("Random Points (sample=5000)", {"sample_size": "5000"}),
                ("Random Points (sample=10000)", {"sample_size": "10000"})
            ]
        },
        "Medoid": {
            "constructor": lambda: initialization_cpp.MedoidInit(metric),
            "query_params": [
                ("Medoid", {})
            ]
        },
        "HVS": {
            "constructor": lambda: initialization_cpp.HVSInit(1, 0.5, 100, metric),
            "query_params": [
                ("HVS (ef_search=500)", {"ef_search": "500"}),
                ("HVS (ef_search=1000)", {"ef_search": "1000"}),
                ("HVS (ef_search=2000)", {"ef_search": "2000"}),
                ("HVS (ef_search=5000)", {"ef_search": "5000"}),
                ("HVS (ef_search=10000)", {"ef_search": "10000"}),
                ("HVS (ef_search=20000)", {"ef_search": "20000"}),
                ("HVS (ef_search=50000)", {"ef_search": "50000"})
            ]
        },
        "LSB-Tree": {
            "constructor": lambda: initialization_cpp.LSBTreeInit(10, 10, 1.0, metric),
            "query_params": [
                ("LSB-Tree (candidates=100)", {"max_candidates": "100"}),
                ("LSB-Tree (candidates=500)", {"max_candidates": "500"}),
                ("LSB-Tree (candidates=1000)", {"max_candidates": "1000"}),
                ("LSB-Tree (candidates=2000)", {"max_candidates": "2000"}),
                ("LSB-Tree (candidates=5000)", {"max_candidates": "5000"}),
                ("LSB-Tree (candidates=10000)", {"max_candidates": "10000"}),
                ("LSB-Tree (candidates=50000)", {"max_candidates": "50000"})
            ]
        }
    }
    
    if target_approach not in all_approaches:
        print(f"Error: Unknown approach '{target_approach}'. Options are: {list(all_approaches.keys())}")
        return
        
    os.makedirs(output_dir, exist_ok=True)
    tracker = PerformanceTracker()
    
    data = all_approaches[target_approach]
    print(f"Building index for {target_approach} on {scenario_name}...")
    approach = data["constructor"]()
    
    tracker.start()
    chunk_size = 50000
    for start_idx in range(0, load_size, chunk_size):
        end_idx = min(start_idx + chunk_size, load_size)
        chunk = dataset[start_idx:end_idx].astype(np.float32)
        approach.add_items(chunk.tolist())
    
    approach.build_index()
    build_time = tracker.stop()
    print(f"Build time: {build_time:.4f} seconds")
    
    mem_footprint = approach.get_memory_usage() / (1024 * 1024)
    index_size = approach.get_index_size() / (1024 * 1024)
    print(f"Index Size: {index_size:.2f} MB, Memory Footprint: {mem_footprint:.2f} MB")
    
    results_list = []
    
    for param_name, query_kwargs in data["query_params"]:
        print(f"Evaluating {param_name}...")
        approach.set_query_time_params(query_kwargs)
        approach.reset_distance_computations()
        
        tracker.start()
        search_results_indices = []
        search_results_distances = []
        
        for q_idx, q in enumerate(queries):
            results = approach.search(q.tolist(), k_search)
            search_results_indices.append([r.index for r in results])
            search_results_distances.append([r.distance for r in results])
            
        search_time = tracker.stop()
        total_dist_comps = approach.get_distance_computations()
        avg_dist_comps = total_dist_comps / len(queries)
        
        avg_recall = {1: 0.0, 100: 0.0}
        avg_mean_dist = 0.0
        avg_1nn_diff = 0.0
        
        for i in range(len(queries)):
            rec = calculate_recall_at_k(search_results_indices[i], ground_truth[i], k_values=[1, 100])
            avg_recall[1] += rec.get(1, 0.0)
            avg_recall[100] += rec.get(100, 0.0)
            
            avg_mean_dist += calculate_mean_distance(search_results_distances[i])
            found_1nn_dist = search_results_distances[i][0] if len(search_results_distances[i]) > 0 else 0.0
            avg_1nn_diff += calculate_1nn_distance_diff(found_1nn_dist, ground_truth[i][0], queries[i], dataset, metric=metric)
            
        avg_recall[1] /= len(queries)
        avg_recall[100] /= len(queries)
        avg_mean_dist /= len(queries)
        avg_1nn_diff /= len(queries)
        
        results_list.append({
            "Approach": param_name,
            "ApproachFamily": target_approach,
            "Build Time (s)": build_time,
            "Index Size (MB)": index_size,
            "Memory Footprint (MB)": mem_footprint,
            "Search Time (ms/q)": (search_time / len(queries)) * 1000.0,
            "Dist Comps": avg_dist_comps,
            "Recall@1": avg_recall[1],
            "Recall@100": avg_recall[100],
            "Mean Dist": avg_mean_dist,
            "1-NN Diff": avg_1nn_diff
        })
        
    del approach
    gc.collect()
    
    res_df = pd.DataFrame(results_list)
    safe_approach_name = target_approach.replace(" ", "_").replace("-", "_")
    output_csv = os.path.join(output_dir, f"results_{scenario_name}_{safe_approach_name}.csv")
    res_df.to_csv(output_csv, index=False)
    print(f"Saved results to {output_csv}")

def main():
    parser = argparse.ArgumentParser(description="Run synthetic vector experiment.")
    parser.add_argument("--metric", type=str, choices=["cosine", "euclidean"], default=None, help="Metric type. Defaults to running all.")
    parser.add_argument("--data", type=str, choices=["uniform", "clustered"], default=None, help="Data type. Defaults to running all.")
    parser.add_argument("--query-type", type=str, choices=["in_dist", "out_dist"], default=None, help="Query type. Defaults to running all.")
    parser.add_argument("--approach", type=str, default=None, help="Target approach name (e.g. 'VP-tree', 'Medoid', etc.). Defaults to running all.")
    parser.add_argument("--data-dir", type=str, default="data_synthetic")
    parser.add_argument("--output-dir", type=str, default="results_synthetic")
    parser.add_argument("--dry-run", action="store_true", help="Use dry-run data directory")
    args = parser.parse_args()

    metrics = [args.metric] if args.metric else ALL_METRICS
    data_types = [args.data] if args.data else ALL_DATA_TYPES
    query_types = [args.query_type] if args.query_type else ALL_QUERY_TYPES
    approaches = [args.approach] if args.approach else ALL_APPROACHES

    is_multi_run = len(metrics) > 1 or len(data_types) > 1 or len(query_types) > 1 or len(approaches) > 1
    if is_multi_run:
        print("\n" + "=" * 80)
        print(" WARNING: Running multiple scenarios/approaches in a single process.")
        print(" Linux memory allocators may not track RSS or deallocate C++ index memory correctly.")
        print(" For accurate memory footprint measurements, specify single CLI arguments.")
        print("=" * 80 + "\n")

    data_dir = "data_synthetic_dryrun" if args.dry_run else args.data_dir
    output_dir = "results_synthetic_dryrun" if args.dry_run else args.output_dir

    for m in metrics:
        for d in data_types:
            for q in query_types:
                for a in approaches:
                    run_experiment_on_scenario(
                        metric=m,
                        data_type=d,
                        query_type=q,
                        target_approach=a,
                        data_dir=data_dir,
                        output_dir=output_dir
                    )

if __name__ == "__main__":
    main()
