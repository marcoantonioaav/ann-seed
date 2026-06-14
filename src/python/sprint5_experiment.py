import numpy as np
import h5py
import pandas as pd
from tqdm import tqdm
import os
import gc

from metrics import PerformanceTracker, calculate_recall_at_k, calculate_mean_distance, calculate_1nn_distance_diff

try:
    import initialization_cpp
except ImportError:
    print("Warning: initialization_cpp not found.")
    initialization_cpp = None

import json
import os
import argparse

def compute_ground_truth_cosine(dataset_name, corpus_ds, queries, k, output_dir):
    gt_file = os.path.join(output_dir, f"{dataset_name}_ground_truth_k{k}_q{len(queries)}.json")
    if os.path.exists(gt_file):
        print(f"Loading cached ground truth from {gt_file}...")
        with open(gt_file, 'r') as f:
            return json.load(f)
            
    num_queries = len(queries)
    top_k_sims = np.full((num_queries, k), -np.inf, dtype=np.float32)
    top_k_indices = np.zeros((num_queries, k), dtype=np.int64)
    
    chunk_size = 50000
    total_size = corpus_ds.shape[0]
    
    # Single pass over the corpus
    for start_idx in tqdm(range(0, total_size, chunk_size), desc="Computing GT (1 pass)"):
        end_idx = min(start_idx + chunk_size, total_size)
        chunk = corpus_ds[start_idx:end_idx]
        
        # sim shape: (chunk_size, num_queries)
        sim = np.dot(chunk, queries.T)
        
        for q_idx in range(num_queries):
            q_sim = sim[:, q_idx]
            
            # Get top k from this chunk using argpartition
            k_chunk = min(k, len(q_sim))
            if len(q_sim) > k:
                local_top_k = np.argpartition(-q_sim, k_chunk - 1)[:k_chunk]
            else:
                local_top_k = np.arange(len(q_sim))
                
            local_sims = q_sim[local_top_k]
            global_indices = start_idx + local_top_k
            
            # Combine with current top-k
            combined_sims = np.concatenate([top_k_sims[q_idx], local_sims])
            combined_indices = np.concatenate([top_k_indices[q_idx], global_indices])
            
            # Extract top k overall
            best = np.argsort(-combined_sims)[:k]
            top_k_sims[q_idx] = combined_sims[best]
            top_k_indices[q_idx] = combined_indices[best]
            
    ground_truth = top_k_indices.tolist()
    
    # Cache it
    os.makedirs(output_dir, exist_ok=True)
    with open(gt_file, 'w') as f:
        json.dump(ground_truth, f)
        
    return ground_truth

def run_dataset_experiment(dataset_name, corpus_path, queries_path, output_dir, max_queries=None, target_approach=None):
    print(f"=== Running Experiment on {dataset_name} ===")
    corpus_f = h5py.File(corpus_path, 'r')
    f = corpus_f
    total_size = f['embeddings'].shape[0]
    load_size = total_size
    print(f"Dataset total size is {total_size}. Loading all points...")
    with h5py.File(queries_path, 'r') as f_q:
        q_total_size = f_q['embeddings'].shape[0]
        q_load_size = q_total_size if max_queries is None else min(q_total_size, max_queries)
        print(f"Loading {q_load_size} queries...")
        q_dataset = f_q['embeddings'][:q_load_size]
        
    k_search = 100
        
    # Read queries once since they are small enough
    queries = q_dataset
    ground_truth = compute_ground_truth_cosine(dataset_name, corpus_f['embeddings'], queries, k_search, output_dir)
    
    approaches = {
        "VP-tree": {
            "constructor": lambda: initialization_cpp.VPTreeInit(1000, 1.0, 1.0, "cosine"),
            "query_params": [
                ("VP-tree (max_leaves=100)", {"max_leaves_to_visit": "100"}),
                ("VP-tree (max_leaves=500)", {"max_leaves_to_visit": "500"}),
                ("VP-tree (max_leaves=1000)", {"max_leaves_to_visit": "1000"}),
                ("VP-tree (max_leaves=2000)", {"max_leaves_to_visit": "2000"}),
                ("VP-tree (max_leaves=5000)", {"max_leaves_to_visit": "5000"})
            ]
        },
        "Stacked NSW": {
            "constructor": lambda: initialization_cpp.StackedNSWInit(16, 200, 10, "cosine"),
            "query_params": [
                ("Stacked NSW (ef=10)", {"ef": "10"}),
                ("Stacked NSW (ef=50)", {"ef": "50"}),
                ("Stacked NSW (ef=100)", {"ef": "100"}),
                ("Stacked NSW (ef=200)", {"ef": "200"}),
                ("Stacked NSW (ef=500)", {"ef": "500"})
            ]
        },
        "LSH": {
            "constructor": lambda: initialization_cpp.LSHInit(10, 16, 10, "cosine"),
            "query_params": [
                ("LSH (probes=10)", {"num_probes": "10"}),
                ("LSH (probes=50)", {"num_probes": "50"}),
                ("LSH (probes=100)", {"num_probes": "100"}),
                ("LSH (probes=200)", {"num_probes": "200"})
            ]
        },
        "t KD-Trees": {
            "constructor": lambda: initialization_cpp.FlannKDTreeInit(4, 100, "cosine"),
            "query_params": [
                ("t KD-Trees (checks=100)", {"checks": "100"}),
                ("t KD-Trees (checks=500)", {"checks": "500"}),
                ("t KD-Trees (checks=1000)", {"checks": "1000"}),
                ("t KD-Trees (checks=2000)", {"checks": "2000"}),
                ("t KD-Trees (checks=5000)", {"checks": "5000"})
            ]
        },
        "t K-Means": {
            "constructor": lambda: initialization_cpp.FlannKMeansInit(1, 16, 2, 100, "cosine"),
            "query_params": [
                ("t K-Means (checks=100)", {"checks": "100"}),
                ("t K-Means (checks=500)", {"checks": "500"}),
                ("t K-Means (checks=1000)", {"checks": "1000"}),
                ("t K-Means (checks=2000)", {"checks": "2000"}),
                ("t K-Means (checks=5000)", {"checks": "5000"}),
            ]
        },
        "Random": {
            "constructor": lambda: initialization_cpp.RandomPointsInit(42, "cosine"),
            "query_params": [
                ("Random Points (sample=100)", {"sample_size": "100"}),
                ("Random Points (sample=1000)", {"sample_size": "1000"}),
                ("Random Points (sample=10000)", {"sample_size": "10000"}),
                ("Random Points (sample=100000)", {"sample_size": "100000"})
            ]
        },
        "Medoid": {
            "constructor": lambda: initialization_cpp.MedoidInit("cosine"),
            "query_params": [
                ("Medoid", {})
            ]
        }
    }
    
    os.makedirs(output_dir, exist_ok=True)
    tracker = PerformanceTracker()
    
    for approach_family, data in approaches.items():
        if target_approach and approach_family != target_approach:
            continue
            
        results_list = []
        
        print(f"\nBuilding index for {approach_family} on {dataset_name}...")
        approach = data["constructor"]()
        
        tracker.start()
        chunk_size = 50000
        for start_idx in tqdm(range(0, load_size, chunk_size), desc=f"Loading chunks for {approach_family}"):
            end_idx = min(start_idx + chunk_size, load_size)
            chunk = corpus_f['embeddings'][start_idx:end_idx].astype(np.float32)
            approach.add_items(chunk.tolist())
            
        approach.build_index()
        build_time = tracker.stop()
        
        mem_footprint = approach.get_memory_usage() / (1024 * 1024)
        index_size = approach.get_index_size() / (1024 * 1024)
        
        for name, params in data["query_params"]:
            print(f"Testing {name}...")
            approach.set_query_time_params(params)
            approach.reset_distance_computations()
            
            tracker.start()
            search_results_indices = []
            search_results_distances = []
            
            for i in tqdm(range(len(queries)), desc=f"Querying {name}"):
                results = approach.search(queries[i].tolist(), k_search)
                search_results_indices.append([r.index for r in results])
                
                # C++ returns metric-specific distance, typically cosine distance or L2
                # Since we use Euclidean/Cosine, we can convert C++ distance to Euclidean if needed
                # However, for metric comparisons, the distance returned by the index is exactly what we want to average!
                search_results_distances.append([r.distance for r in results])
                
            search_time = tracker.stop()
            total_dist_comps = approach.get_distance_computations()
            avg_dist_comps = total_dist_comps / len(queries)
            
            avg_recall = {1: 0.0, 10: 0.0, 25: 0.0, 100: 0.0}
            avg_mean_dist = 0.0
            avg_1nn_diff = 0.0
            
            for i in range(len(queries)):
                rec = calculate_recall_at_k(search_results_indices[i], ground_truth[i], k_values=[1, 10, 25, 100])
                for k in avg_recall:
                    avg_recall[k] += rec.get(k, 0.0)
                avg_mean_dist += calculate_mean_distance(search_results_distances[i])
                
                # For 1NN diff, if the search returns nothing, we assign 0
                found_1nn_dist = search_results_distances[i][0] if len(search_results_distances[i]) > 0 else 0.0
                
                avg_1nn_diff += calculate_1nn_distance_diff(found_1nn_dist, ground_truth[i][0], queries[i], corpus_f['embeddings'], metric="cosine")
                
            for k in avg_recall:
                avg_recall[k] /= len(queries)
            avg_mean_dist /= len(queries)
            avg_1nn_diff /= len(queries)
            
            results_list.append({
                "Parameter Setup": name,
                "Build Time (s)": build_time,
                "Memory Footprint (MB)": mem_footprint,
                "Index Size (MB)": index_size,
                "Search Time (ms/q)": (search_time / len(queries)) * 1000,
                "Dist Comps": avg_dist_comps,
                "Recall@1": avg_recall[1],
                "Recall@10": avg_recall[10],
                "Recall@25": avg_recall[25],
                "Recall@100": avg_recall[100],
                "Mean Dist": avg_mean_dist,
                "1-NN Diff": avg_1nn_diff
            })
            
        del approach
        gc.collect()
            
        df = pd.DataFrame(results_list)
        df.to_csv(f"{output_dir}/results_{dataset_name}_{approach_family.replace(' ', '_')}.csv", index=False)
        
    corpus_f.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Dense Retrieval Experiment")
    parser.add_argument("--approach", type=str, default=None, help="Specific approach family to run (e.g., 'Stacked NSW', 'VP-tree')")
    parser.add_argument("--dataset", type=str, default=None, help="Specific dataset to run (e.g., 'MSMARCO' or 'NQ')")
    args = parser.parse_args()

    msmarco_corpus = "/home/marco/Text-Dense-Retrieval-Evaluation/embeddings/msmarco/Octen-Embedding-0.6B_corpus_pca256.h5"
    msmarco_queries = "/home/marco/Text-Dense-Retrieval-Evaluation/embeddings/msmarco/Octen-Embedding-0.6B_queries_pca256.h5"
    
    nq_corpus = "/home/marco/Text-Dense-Retrieval-Evaluation/embeddings/nq/Octen-Embedding-0.6B_corpus_pca256.h5"
    nq_queries = "/home/marco/Text-Dense-Retrieval-Evaluation/embeddings/nq/Octen-Embedding-0.6B_queries_pca256.h5"
    
    output_dir = "results"
    
    if (args.dataset is None or args.dataset.upper() == "MSMARCO"):
        if os.path.exists(msmarco_corpus):
            run_dataset_experiment("MSMARCO", msmarco_corpus, msmarco_queries, output_dir, target_approach=args.approach)
        else:
            print(f"Skipping MSMARCO, {msmarco_corpus} not found.")
        
    if (args.dataset is None or args.dataset.upper() == "NQ"):
        if os.path.exists(nq_corpus):
            run_dataset_experiment("NQ", nq_corpus, nq_queries, output_dir, target_approach=args.approach)
        else:
            print(f"Skipping NQ, {nq_corpus} not found.")
