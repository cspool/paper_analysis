# A.2 Artifact check-list (meta-information)

- Program: Python = 3.7 (analytical backend), Python = 2.7 (GARNET backend), Astra-SIM
- Run-time environment: Ubuntu = 22.04
- Experiments: We include the scripts for running simulations and real machine tests.
- Metrics: We evaluate the performance speedups, communication bandwidth, bandwidth utilization, performance scalability, comparison with Google's routing, end-to-end time breakdown, non-uniform All-to-All, performance under multi failures, and real-machine communication time in our evaluation.
- Output: The outputs of the artifact are figures in PDF format that reproduce the main results of our paper.
- How much disk space required (approximately)?: The disk space should be around 2 TB.
- How much time is needed to prepare workflow (approximately)?: Around 40 minutes.
- How much time is needed to complete experiments (approximately)?: All the simulation experiments with analytical backend takes 9 hours approximately. The simulation time with GARNET backend scales with communication size. The shortest time spent on some small communication size is a few minutes, while the longest time spent on others can be up to 6 days. All experiments might take up to two weeks. Parallel simulations are recommended. Realmachine experiments take 20 minutes approximately.
- Publicly available?: Yes.

