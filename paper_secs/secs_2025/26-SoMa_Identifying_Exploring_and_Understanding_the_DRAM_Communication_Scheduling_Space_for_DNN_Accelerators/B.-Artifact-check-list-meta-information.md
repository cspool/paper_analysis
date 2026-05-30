# *B. Artifact check-list (meta-information)*

- Algorithm: Simulated Annealing
- Program: C++, Shell, Python (only for data collection)
- Compilation: by Makefile
- Hardware: Recommend a server with 96+ cores and at least 1GB RAM per core.
- Metrics: Cost function E×D is employed in all experiments.
- Experiments: reproduce Fig. 7 and Fig. 6.
- How much disk space required (approximately)?: 1GB
- How much time is needed to prepare workflow(approximately)?: Several minutes at most.
- How much time is needed to complete experiments (approximately)?: For all 432 experiments (96 for Fig. 6 and 332 for Fig. 7), it takes about 2 days on a 192 core Intel Xeon Platinum 8260. Most experiments (95%) are completed within 3.5 hours, while the remaining ones, mainly experiments with batch=64, require the full 2 days to finish.
- Publicly available?: Yes
- Code licenses (if publicly available)?: AGPL-3.0 License
- Archived (provide DOI)?: 10.5281/zenodo.14599935

