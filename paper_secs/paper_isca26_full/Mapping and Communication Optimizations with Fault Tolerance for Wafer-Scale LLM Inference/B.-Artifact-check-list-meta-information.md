# *B. Artifact check-list (meta-information)*

- Program: Python 3.9 (BusyBarn framework)
- Run-time environment: Linux or macOS; Conda with Python 3.9; dependencies: numpy, networkx, simanneal, matplotlib, tqdm, PyYAML
- Hardware: Multi-core CPU; 32 GB RAM minimum
- Metrics: We evaluated communication bandwidth (GB/s), speedup over baselines, and convergence traces (objective value vs. SA steps)
- Output: 12 PDF figures and a speedup summary text file.
- How much disk space required (approximately)?: ∼5 GB for generated scripts and results
- How much time is needed to prepare workflow (approximately)?: ∼20 minutes (conda environment setup + dependency installation)
- How much time is needed to complete experiments (approximately)?: The simulation time varies among different experiments. Communication and mapping sensitivity experiments finish individual tasks in seconds to minutes, while end-to-end model evaluations (especially large MoE models) can take up to 1 hour per task. A quick validation subset (run quick test.sh) completes in under 3 hours. The full evaluation with all 12 figures may take several days on a single machine. Parallel simulation is strongly recommended.
- Publicly available?: Yes
- Workflow automation framework used?: Makefile + Bash scripts; optional SLURM integration

