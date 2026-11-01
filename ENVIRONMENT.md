# Execution Environment

The reported experiments used one model per NVIDIA GPU with 48 GB memory,
bfloat16 weights, SDPA attention, and greedy decoding. The frozen run
manifests record model architectures, weight-file checks, generation limits,
MI dimensions, and artifact hashes.

The packaging and integrity checks were validated on:

- Python 3.10
- PyTorch 2.12.0+cu130
- Transformers 5.8.1
- NumPy 1.26.4
- pandas 2.3.3
- SciPy 1.15.3
- scikit-learn 1.7.2
- joblib 1.5.3
- NVIDIA GeForce RTX 4090 D, 48 GB

The original repository dependency snapshot is retained separately as
`requirements_original.txt`. Exact package versions for every original model
generation process were not captured in the run manifests; this limitation is
reflected as `partial` in the reproducibility checklist.
