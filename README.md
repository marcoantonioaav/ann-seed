# ANN Seed

Comparison of seed selection strategies for graph-based ANN search.

## Installation Requirements

Ensure you have `cmake` and a relatively modern C++ compiler (for building the NMSLIB/FALCONN backends). 

First, install the required Python libraries:
```bash
pip install -r requirements.txt
```

Second, compile the C++ `initialization_cpp` bindings:
```bash
mkdir build && cd build
cmake ..
cmake --build .
```
> Note: The compiled Python module (`.so` object) will automatically be placed into `src/python/` alongside the execution scripts.

## Running Instructions

### 1. Verification (Fast Testing)
To verify everything is built correctly without downloading massive datasets, test the architecture against a generated random dataset:
```bash
python3 src/python/fast_experiment.py
```

### 2. Full Evaluation Sweep
Run the complete experiments using the main CLI flags to define the datasets and individual approaches:
```bash
python3 src/python/full_experiment.py --dataset MSMARCO --approach "Stacked NSW"
```

### 3. Generate Report Visuals
After generating the `results/*.csv` data via the full experiment sweeps, run the analysis tool to cleanly summarize everything into high-quality side-by-side Matplotlib charts and `.tex` tables:
```bash
python3 src/python/analysis.py
```
