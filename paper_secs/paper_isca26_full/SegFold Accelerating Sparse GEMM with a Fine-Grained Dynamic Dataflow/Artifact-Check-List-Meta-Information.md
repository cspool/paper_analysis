# *Artifact Check-List (Meta-Information)*

- Algorithm: Segment dataflow for SpGEMM with fine-grained dynamic scheduling and work remapping.
- Program: csegfold cycle-accurate C++ simulator.
- Compilation: CMake ≥ 3.15, GCC 10+ (C++20 required). Ramulator2 is pulled automatically during the build via CMake FetchContent.
- Binary: csegfold/build/csegfold, compiled from source.
- Data set: ∼20 SuiteSparse matrices (∼50 MB, auto-downloaded), covering the 15-matrix baseline suite plus the additional matrices used in ablation studies; synthetic matrices are created on-the-fly.
- Hardware: Commodity server ≥4 CPU cores and ≥64 GB RAM (16+ cores and 256 GB RAM recommended for full parallelism).
- Run-time environment: Linux (tested on Ubuntu 22.04+), Python ≥ 3.8. A Docker image is also available.
- Metrics: Simulated cycle counts and speedup relative to prior accelerators.
- Output: Per-experiment CSV files and PDF/PNG plots corresponding to Figures 8–12.
- Experiments: 209 individual simulation runs, completing in ∼2 hours on a 16-core machine.
- How much disk space required? ∼2 GB (source, benchmarks, and generated outputs).
- How much time is needed to prepare the workflow? ∼5 minutes for building the simulator and downloading matrices.
- How much time is needed to complete experiments? ∼2 hours at 16 cores; runtime scales roughly linearly with core count.
- Publicly available? Yes.
- Code licenses? MIT License.
- Archived? Yes (GitHub + Zenodo) DOI: 10.5281/zenodo.19453259.

