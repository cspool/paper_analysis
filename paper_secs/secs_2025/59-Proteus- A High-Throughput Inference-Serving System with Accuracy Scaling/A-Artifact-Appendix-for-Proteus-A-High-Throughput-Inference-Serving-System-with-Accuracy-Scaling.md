# A Artifact Appendix for Proteus: A High-Throughput Inference-Serving System with Accuracy Scaling

#### A.1 Abstract

This artifact describes the complete workflow to setup the simulation experiments for Proteus. We describe how to obtain the code, and then describe two methods to install the simulator. We explain how to run the experiments and the expected results from simulation. Finally, we also publicize all the workload traces used in our paper.

### A.2 Artifact check-list (meta-information)

- Algorithm: Combinatorial optimization using mixed integer linear programming, adaptive batching.
- Hardware: Docker container for linux\_amd64 platform provided. Source code can be used on any hardware.
- Metrics: Throughput, inference accuracy, latency SLO violations
- Output: Log files are output by the simulator which are then used by the plotting scripts to generate graphs for results.
- Experiments: End-to-end evaluation of Proteus and baselines as well as an evaluation of the responsiveness of Proteus against baselines.
- How much disk space required (approximately)?: Docker container requires approximately 1.5GB of disk space.
- How much time is needed to prepare workflow (approximately)?: 15 minutes.
- How much time is needed to complete experiments (approximately)?: 1-2 hours depending on hardware platform.
- Publicly available?: Yes. See below for access details.

