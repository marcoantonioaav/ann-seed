import h5py
import numpy as np

def check_normalization(file_path, dataset_name, num_samples=1000):
    print(f"Checking normalization for {file_path} [{dataset_name}]")
    try:
        with h5py.File(file_path, 'r') as f:
            if dataset_name not in f:
                print(f"  Dataset '{dataset_name}' not found. Available keys: {list(f.keys())}")
                return
            
            data = f[dataset_name]
            total_samples = data.shape[0]
            samples_to_check = min(num_samples, total_samples)
            
            # Load a random subset to check magnitude
            indices = np.random.choice(total_samples, samples_to_check, replace=False)
            indices.sort()
            
            subset = data[indices]
            magnitudes = np.linalg.norm(subset, axis=1)
            
            avg_mag = np.mean(magnitudes)
            std_mag = np.std(magnitudes)
            
            print(f"  Mean magnitude: {avg_mag:.6f}")
            print(f"  Std magnitude:  {std_mag:.6f}")
            
            if np.isclose(avg_mag, 1.0, atol=1e-3) and std_mag < 1e-3:
                print("  => Vectors appear to be normalized.")
            else:
                print("  => Vectors are NOT normalized.")
    except Exception as e:
        print(f"  Error: {e}")

if __name__ == "__main__":
    check_normalization("/home/marco/Text-Dense-Retrieval-Evaluation/embeddings/msmarco/Octen-Embedding-0.6B_corpus.h5", "embeddings")
    check_normalization("/home/marco/Text-Dense-Retrieval-Evaluation/embeddings/msmarco/Octen-Embedding-0.6B_queries.h5", "embeddings")
    check_normalization("/home/marco/Text-Dense-Retrieval-Evaluation/embeddings/nq/Octen-Embedding-0.6B_corpus.h5", "embeddings")
    check_normalization("/home/marco/Text-Dense-Retrieval-Evaluation/embeddings/nq/Octen-Embedding-0.6B_queries.h5", "embeddings")
