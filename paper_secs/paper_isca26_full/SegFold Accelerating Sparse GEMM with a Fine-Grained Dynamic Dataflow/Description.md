# *Description*

*How to Access:* The artifact is hosted on GitHub: <https://github.com/PolyArch/SegFold-AE>

and also Zenodo:

<https://doi.org/10.5281/zenodo.19453259>

*Hardware Dependencies:*

- CPU: 4 cores minimum, 16+ cores recommended.
- RAM: 64 GB minimum, 256 GB recommended. Memoryintensive experiments (breakdown and mapping ablation) may consume up to 50 GB per process.
- Disk: 2–5 GB.

*Software Dependencies:*

- OS: Ubuntu 22.04 or later (other Linux distributions are expected to work).
- Toolchain: GCC 10+ (C++20), CMake ≥ 3.15.
- Python ≥ 3.8 with numpy, scipy, matplotlib, pandas, and pyyaml.
- Docker (optional, for a self-contained environment).

*Data Sets:* The experiments use 20 sparse matrices from the SuiteSparse Matrix Collection [\(https://sparse.tamu.edu/\)](https://sparse.tamu.edu/), fetched automatically by a provided download script. Synthetic matrices for sensitivity studies are generated at runtime.

