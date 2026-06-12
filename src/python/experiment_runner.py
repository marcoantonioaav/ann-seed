import numpy as np
from tabulate import tabulate
from dataset_generator import DatasetGenerator
from metrics import PerformanceTracker, calculate_recall_at_k, calculate_mean_distance, calculate_1nn_distance_diff

# Attempt to import the compiled C++ extension. 
try:
    import initialization_cpp
except ImportError:
    print("Warning: initialization_cpp not found. Please build the C++ module using CMake.")
    initialization_cpp = None

def compute_ground_truth(dataset, queries, k):
    """Computes exact nearest neighbors for evaluation purposes."""
    ground_truth = []
    for q in queries:
        distances = np.linalg.norm(dataset - q, axis=1)
        nearest_indices = np.argsort(distances)[:k]
        ground_truth.append(nearest_indices.tolist())
    return ground_truth

def run_experiment():
    print("=== Sprint 0 Experiment Runner ===")
    
    # 1. Dataset Generation
    generator = DatasetGenerator(seed=42)
    num_samples = 10000
    num_dimensions = 128
    num_queries = 100
    print(f"Generating dataset with {num_samples} points, {num_dimensions} dimensions...")
    dataset, queries = generator.generate(num_samples, num_dimensions, num_queries)
    dataset_list = dataset.tolist() 
    
    # 2. Ground Truth calculation
    k_max = 25
    print("Computing ground truth...")
    ground_truth = compute_ground_truth(dataset, queries, k=k_max)
    
    if initialization_cpp is None:
        print("Cannot run approaches without initialization_cpp.")
        return

    tracker = PerformanceTracker()
    
    k_search = 100
    # Define the approaches to test
    approaches = [
        ("Random Points", initialization_cpp.RandomPointsInit(42)),
        ("Medoid", initialization_cpp.MedoidInit()),
        ("t KD-Trees (t=1)", initialization_cpp.FlannKDTreeInit(1, k_search)),
        ("t KD-Trees (t=5)", initialization_cpp.FlannKDTreeInit(5, 5*k_search)),
        ("t K-Means trees (t=1)", initialization_cpp.FlannKMeansInit(1, 32, 11, k_search)),
        ("t K-Means trees (t=5)", initialization_cpp.FlannKMeansInit(5, 32, 11, k_search)),
        ("VP-tree", initialization_cpp.VPTreeInit(8, 1.0, 1.0)),
        ("Stacked NSW graphs", initialization_cpp.StackedNSWInit(16, 400, k_search)),
        ("LSH", initialization_cpp.LSHInit(4, 20, 4))
    ]
    
    offline_results = []
    online_results = []
    
    for name, approach in approaches:
        print(f"Testing {name}...")
        
        # Build phase (offline)
        tracker.start()
        approach.build(dataset_list)
        build_time = tracker.stop()
        mem_footprint = approach.get_memory_usage() / (1024 * 1024)
        index_size = approach.get_index_size() / (1024 * 1024)
        
        # Search phase (online)
        approach.reset_distance_computations()
        tracker.start()
        search_results_indices = []
        for i in range(num_queries):
            query_vec = queries[i].tolist()
            # C++ returns a list of SearchResult objects
            results = approach.search(query_vec, k_search)
            search_results_indices.append([r.index for r in results])
        search_time = tracker.stop()
        
        # Get average distance computations from C++
        total_dist_comps = approach.get_distance_computations()
        avg_dist_comps = total_dist_comps / num_queries
        
        # Metrics calculation (averaged over all queries)
        avg_recall = {1: 0.0, 10: 0.0, 25: 0.0}
        avg_mean_dist = 0.0
        avg_1nn_diff = 0.0
        
        for i in range(num_queries):
            rec = calculate_recall_at_k(search_results_indices[i], ground_truth[i], k_values=[1, 10, 25])
            for k in avg_recall:
                avg_recall[k] += rec.get(k, 0.0)
            avg_mean_dist += calculate_mean_distance(search_results_indices[i], queries[i], dataset)
            avg_1nn_diff += calculate_1nn_distance_diff(search_results_indices[i][0], ground_truth[i][0], queries[i], dataset)
            
        # Averaging
        for k in avg_recall:
            avg_recall[k] /= num_queries
        avg_mean_dist /= num_queries
        avg_1nn_diff /= num_queries
        
        # Populate Offline Results
        offline_results.append({
            "Approach": name,
            "Build Time (s)": f"{build_time:.4f}",
            "Memory Footprint (MB)": f"{mem_footprint:.4f}",
            "Index Size (MB)": f"{index_size:.4f}"
        })

        # Populate Online Results
        online_results.append({
            "Approach": name,
            "Search Time (ms/q)": f"{(search_time / num_queries) * 1000:.4f}",
            "Dist Comps": f"{avg_dist_comps:.4f}",
            "Recall@1": f"{avg_recall[1]:.4f}",
            "Recall@10": f"{avg_recall[10]:.4f}",
            "Recall@25": f"{avg_recall[25]:.4f}",
            "Mean Dist": f"{avg_mean_dist:.4f}",
            "1-NN Diff": f"{avg_1nn_diff:.4f}"
        })

    # 3. Print Results Tables
    # Note: Using orgtbl for better alignment/formatting support in the terminal.
    # We align everything to the right except the first column.
    # disable_numparse=True ensures our manual formatting (4 decimals) is preserved.
    print("\n### Offline Performance (Indexing)\n")
    print(tabulate(offline_results, headers="keys", tablefmt="orgtbl", stralign="right", disable_numparse=True))

    print("\n### Online Performance (Querying)\n")
    print(tabulate(online_results, headers="keys", tablefmt="orgtbl", stralign="right", disable_numparse=True))

if __name__ == "__main__":
    run_experiment()
