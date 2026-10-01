import numpy as np
import h5py
import os
import argparse
from tqdm import tqdm

def generate_uniform(num_samples, dim, metric, rng):
    """Generates uniform synthetic vectors."""
    vecs = rng.standard_normal((num_samples, dim)).astype(np.float32)
    if metric == "cosine":
        # Normalize to unit length
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms[norms == 0] = 1e-10
        vecs = vecs / norms
    return vecs

def generate_clustered(num_samples, dim, num_clusters, metric, rng, cluster_centers=None, cluster_std=0.1):
    """Generates clustered synthetic vectors using Gaussian mixture."""
    if cluster_centers is None:
        cluster_centers = rng.standard_normal((num_clusters, dim)).astype(np.float32)
        if metric == "cosine":
            c_norms = np.linalg.norm(cluster_centers, axis=1, keepdims=True)
            c_norms[c_norms == 0] = 1e-10
            cluster_centers = cluster_centers / c_norms

    # Assign samples evenly to clusters
    assignments = rng.choice(num_clusters, size=num_samples)
    vecs = np.zeros((num_samples, dim), dtype=np.float32)
    
    for c in range(num_clusters):
        mask = (assignments == c)
        count = np.sum(mask)
        if count == 0:
            continue
        noise = rng.normal(0, cluster_std, size=(count, dim)).astype(np.float32)
        vecs[mask] = cluster_centers[c] + noise

    if metric == "cosine":
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms[norms == 0] = 1e-10
        vecs = vecs / norms

    return vecs, cluster_centers

def compute_ground_truth(dataset, queries, k, metric):
    """Computes exact top-k ground truth indices for queries against dataset."""
    num_queries = len(queries)
    total_size = len(dataset)
    chunk_size = 50000

    if metric == "cosine":
        # Cosine similarity (larger is closer)
        top_k_vals = np.full((num_queries, k), -np.inf, dtype=np.float32)
        top_k_indices = np.zeros((num_queries, k), dtype=np.int64)

        for start_idx in tqdm(range(0, total_size, chunk_size), desc=f"Computing GT ({metric})"):
            end_idx = min(start_idx + chunk_size, total_size)
            chunk = dataset[start_idx:end_idx]
            sim = np.dot(chunk, queries.T) # shape: (chunk_len, num_queries)

            for q_idx in range(num_queries):
                q_sim = sim[:, q_idx]
                k_chunk = min(k, len(q_sim))
                if len(q_sim) > k:
                    local_top_k = np.argpartition(-q_sim, k_chunk - 1)[:k_chunk]
                else:
                    local_top_k = np.arange(len(q_sim))
                local_sims = q_sim[local_top_k]
                global_indices = start_idx + local_top_k

                combined_sims = np.concatenate([top_k_vals[q_idx], local_sims])
                combined_indices = np.concatenate([top_k_indices[q_idx], global_indices])

                best = np.argsort(-combined_sims)[:k]
                top_k_vals[q_idx] = combined_sims[best]
                top_k_indices[q_idx] = combined_indices[best]

    else: # euclidean
        # L2 distance (smaller is closer)
        top_k_vals = np.full((num_queries, k), np.inf, dtype=np.float32)
        top_k_indices = np.zeros((num_queries, k), dtype=np.int64)

        for start_idx in tqdm(range(0, total_size, chunk_size), desc=f"Computing GT ({metric})"):
            end_idx = min(start_idx + chunk_size, total_size)
            chunk = dataset[start_idx:end_idx] # (chunk_len, dim)
            
            # Squared L2 distance: ||a - b||^2 = ||a||^2 + ||b||^2 - 2 a.b
            chunk_sq = np.sum(chunk**2, axis=1, keepdims=True) # (chunk_len, 1)
            queries_sq = np.sum(queries**2, axis=1, keepdims=True).T # (1, num_queries)
            dist_sq = chunk_sq + queries_sq - 2 * np.dot(chunk, queries.T)
            dist_sq = np.maximum(dist_sq, 0.0)
            dists = np.sqrt(dist_sq)

            for q_idx in range(num_queries):
                q_dists = dists[:, q_idx]
                k_chunk = min(k, len(q_dists))
                if len(q_dists) > k:
                    local_top_k = np.argpartition(q_dists, k_chunk - 1)[:k_chunk]
                else:
                    local_top_k = np.arange(len(q_dists))
                local_dists = q_dists[local_top_k]
                global_indices = start_idx + local_top_k

                combined_dists = np.concatenate([top_k_vals[q_idx], local_dists])
                combined_indices = np.concatenate([top_k_indices[q_idx], global_indices])

                best = np.argsort(combined_dists)[:k]
                top_k_vals[q_idx] = combined_dists[best]
                top_k_indices[q_idx] = combined_indices[best]

    return top_k_indices.tolist()

def generate_dataset_scenario(metric, data_type, query_type, seed, output_dir, num_vectors, num_queries, dim, num_clusters=20):
    scenario_name = f"{metric}_{data_type}_{query_type}"
    output_file = os.path.join(output_dir, f"synthetic_{scenario_name}.h5")
    
    if os.path.exists(output_file):
        print(f"Dataset scenario '{scenario_name}' already exists at {output_file}. Skipping.")
        return

    print(f"=== Generating Dataset Scenario: {scenario_name} (seed={seed}) ===")
    rng = np.random.default_rng(seed)

    if data_type == "uniform":
        dataset = generate_uniform(num_vectors, dim, metric, rng)
        if query_type == "in_dist":
            queries = generate_uniform(num_queries, dim, metric, rng)
        else: # out_dist: shifted uniform (mean shift)
            shift = rng.standard_normal((1, dim)).astype(np.float32) * 0.5
            queries = generate_uniform(num_queries, dim, metric, rng) + shift
            if metric == "cosine":
                q_norms = np.linalg.norm(queries, axis=1, keepdims=True)
                queries = queries / q_norms
    else: # clustered
        dataset, centers = generate_clustered(num_vectors, dim, num_clusters, metric, rng)
        if query_type == "in_dist":
            queries, _ = generate_clustered(num_queries, dim, num_clusters, metric, rng, cluster_centers=centers)
        else: # out_dist: queries generated from different cluster centers
            ood_centers = rng.standard_normal((num_clusters, dim)).astype(np.float32)
            if metric == "cosine":
                c_norms = np.linalg.norm(ood_centers, axis=1, keepdims=True)
                ood_centers = ood_centers / c_norms
            queries, _ = generate_clustered(num_queries, dim, num_clusters, metric, rng, cluster_centers=ood_centers)

    k_gt = 100
    ground_truth = compute_ground_truth(dataset, queries, k_gt, metric)

    os.makedirs(output_dir, exist_ok=True)
    with h5py.File(output_file, 'w') as f:
        f.create_dataset('dataset', data=dataset, compression='gzip')
        f.create_dataset('queries', data=queries, compression='gzip')
        f.create_dataset('ground_truth', data=np.array(ground_truth, dtype=np.int64), compression='gzip')

    print(f"Successfully saved {scenario_name} to {output_file}")

def main():
    parser = argparse.ArgumentParser(description="Generate synthetic datasets and ground truth.")
    parser.add_argument("--output-dir", type=str, default="data_synthetic", help="Output directory for h5 files")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--num-vectors", type=int, default=1000000, help="Number of dataset vectors")
    parser.add_argument("--num-queries", type=int, default=1000, help="Number of query vectors")
    parser.add_argument("--dim", type=int, default=256, help="Vector dimension")
    parser.add_argument("--dry-run", action="store_true", help="Run with small sizes (10k vectors, 100 queries) for fast testing")
    args = parser.parse_args()

    num_vectors = 10000 if args.dry_run else args.num_vectors
    num_queries = 100 if args.dry_run else args.num_queries
    output_dir = "data_synthetic_dryrun" if args.dry_run else args.output_dir

    metrics = ["cosine", "euclidean"]
    data_types = ["uniform", "clustered"]
    query_types = ["in_dist", "out_dist"]

    for m in metrics:
        for d in data_types:
            for q in query_types:
                generate_dataset_scenario(
                    metric=m,
                    data_type=d,
                    query_type=q,
                    seed=args.seed,
                    output_dir=output_dir,
                    num_vectors=num_vectors,
                    num_queries=num_queries,
                    dim=args.dim
                )

if __name__ == "__main__":
    main()

