# *E. Installation*

- 1) Download and extract the artifact from Zenodo.
- 2) Build the Docker images:
  - ./scripts/build\_docker.sh run ./scripts/build\_docker.sh covcollect ./scripts/build\_docker.sh naxriscv
- 3) Pre-built simulator binaries are included in simulators/. To optionally rebuild from source (several hours):
  - ./scripts/build\_chipyard.sh
  - ./scripts/build\_toooba.sh
  - ./scripts/build\_xs.sh

#### *F. Experiment Workflow*

All experiments are orchestrated by per-figure Bash scripts in artifact\_reproduction/. Each script runs inside the appropriate Docker container and produces a PDF in figures/.

1) Collect shared benchmark data (required for Figures 10–13):

./artifact\_reproduction/collect\_data.sh

## 2) Generate individual figures:

```
./artifact_reproduction/figure_8.sh
./artifact_reproduction/figure_9.sh
./artifact_reproduction/figure_10.sh
./artifact_reproduction/figure_11.sh
./artifact_reproduction/figure_12.sh
./artifact_reproduction/figure_13.sh
./artifact_reproduction/figure_14.sh
```

Figures 8, 9, and 14 collect their own data and can be run independently of step 1. Figure 14 supports a --quick flag for faster evaluation. We provide a script to run everything in one command: ./artifact\_reproduction/run\_all.sh

#### *G. Evaluation and Expected Results*

Each script produces a PDF figure in figures/ that should match the corresponding figure in the paper:

- Figure 8 (PPO Rule Usage Probabilities): Distribution of rule usage should show similar relative proportions.
- Figure 9 (Memory Operations Distance Distribution): Histogram shape should match the paper.
- Figure 10 (Verification Throughput Overhead): Overhead ratios should be within ±10% of reported values.
- Figure 11 (Instruction Throughput): Throughput values may vary by ±15% depending on the host machine, but relative ordering across designs should be preserved.
- Figure 12 (Simulation Time): Absolute times are machine-dependent; trends across instruction sizes should match.
- Figure 13 (ISS and Simulation Time Breakdown): Relative proportions between ISS and simulation time should match.
- Figure 14 (Coverage Comparison): Coverage curves should show the same relative ranking (HartBreaker with verification > HartBreaker without > RISCV-DV).

Absolute performance numbers are expected to vary across machines; the key claims are about relative comparisons and trends.

