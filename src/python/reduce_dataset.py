import numpy as np
import h5py
from sklearn.decomposition import PCA
from tqdm import tqdm
import os
import argparse

def reduce_dimensionality(corpus_path, queries_path, output_corpus_path, output_queries_path, target_dims=256, fit_samples=200000):
    print(f"Loading {fit_samples} samples to fit PCA...")
    with h5py.File(corpus_path, 'r') as f:
        total_size = f['embeddings'].shape[0]
        actual_fit_samples = min(total_size, fit_samples)
        
        # We can just take the first N samples for fitting, assuming they are representative enough
        # Random sampling would require reading the entire 33GB file.
        fit_data = f['embeddings'][:actual_fit_samples]
        
    print(f"Fitting PCA from {fit_data.shape[1]} to {target_dims} dimensions...")
    pca = PCA(n_components=target_dims)
    pca.fit(fit_data)
    
    # We no longer need the fit data in memory
    del fit_data
    
    # Process Corpus in chunks
    print(f"Transforming corpus and saving to {output_corpus_path}...")
    chunk_size = 50000
    with h5py.File(corpus_path, 'r') as f_in, h5py.File(output_corpus_path, 'w') as f_out:
        total_size = f_in['embeddings'].shape[0]
        # Create a compressed dataset for the output
        dset_out = f_out.create_dataset('embeddings', shape=(total_size, target_dims), dtype=np.float32)
        
        for start_idx in tqdm(range(0, total_size, chunk_size), desc="Transforming Corpus"):
            end_idx = min(start_idx + chunk_size, total_size)
            chunk = f_in['embeddings'][start_idx:end_idx]
            
            # Transform
            chunk_pca = pca.transform(chunk)
            
            # Re-normalize to unit length (important for cosine similarity)
            norms = np.linalg.norm(chunk_pca, axis=1, keepdims=True)
            chunk_pca = chunk_pca / (norms + 1e-10)
            
            dset_out[start_idx:end_idx] = chunk_pca
            
    # Process Queries
    print(f"Transforming queries and saving to {output_queries_path}...")
    with h5py.File(queries_path, 'r') as f_in, h5py.File(output_queries_path, 'w') as f_out:
        q_data = f_in['embeddings'][:]
        q_pca = pca.transform(q_data)
        
        norms = np.linalg.norm(q_pca, axis=1, keepdims=True)
        q_pca = q_pca / (norms + 1e-10)
        
        f_out.create_dataset('embeddings', data=q_pca)
        
    print("Done!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Reduce dimensionality of HDF5 datasets using PCA.")
    parser.add_argument("--corpus", required=True, help="Path to input corpus HDF5")
    parser.add_argument("--queries", required=True, help="Path to input queries HDF5")
    parser.add_argument("--out-corpus", required=True, help="Path to output corpus HDF5")
    parser.add_argument("--out-queries", required=True, help="Path to output queries HDF5")
    parser.add_argument("--dims", type=int, default=256, help="Target dimensions (default: 256)")
    
    args = parser.parse_args()
    reduce_dimensionality(args.corpus, args.queries, args.out_corpus, args.out_queries, args.dims)
