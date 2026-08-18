# APPENDIX: ARTIFACT EVALUATION

## *A. Abstract*

This artifact provides the Zenodo reproducibility package for the TACO framework and the experimental workflow used in the paper. It contains the benchmark datasets, figuregeneration scripts, and a self-contained copy of the NWQEC codebase with TACO functionality integrated. The artifact is available at [https://doi.org/10.5281/zenodo.19449157,](https://doi.org/10.5281/zenodo.19449157) and NWQEC is open-sourced on GitHub at [https://github.com/](https://github.com/pnnl/nwqec) [pnnl/nwqec.](https://github.com/pnnl/nwqec) The package reproduces key results on Clifford reduction and gate parallelism via automated command-line workflows that generate intermediate CSV files and final PDF figures. All experiments run on a standard CPU machine without specialized hardware and complete within tens of minutes. Detailed build and execution instructions are provided in the artifact README.

## *B. Artifact check-list (meta-information)*

- Algorithm: Clifford+T and Pauli-based computation (PBC) FTQC transpilation; Clifford-reduction analysis; gate-parallelism analysis
- Program: NWQEC (nwqec-cli), Bash workflows, Python plotting/analysis scripts
- Compilation: CMake + C++17
- Transformations: N/A
- Binary: nwqec-cli
- Data set: QASM benchmark circuits (paper benchmark set + prior-work benchmark sets)
- Run-time environment: Linux or macOS shell environment with Python 3
- Hardware: CPU-only
- Execution: Scripted command-line workflows via top-level Bash scripts
- Metrics: Clifford ratio, Clifford reduction ratio, gate parallelism, and operation-weight distributions
- Output: CSV files in results/ and PDF figures in figures/
- Experiments: Fig. [5,](#page-3-1) Fig. [14,](#page-6-3) Fig. [17,](#page-8-2) Fig. [20,](#page-10-3) Fig. [21,](#page-11-1) Fig. [22](#page-11-2)
- How much disk space required (approximately)?: < 100 MB
- How much time is needed to prepare workflow (approximately)?: 5–10 minutes (build + environment setup)
- How much time is needed to complete experiments (approximately)?: 10–20 minutes (Fig. 20 is the longest)
- Publicly available?: yes
- Code licenses (if publicly available)?: MIT
- Data licenses (if publicly available)?: MIT
- Workflow automation framework used?: Bash scripts
- Archived (provide DOI)?: 10.5281/zenodo.19449157

