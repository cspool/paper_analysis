# A.4 Evaluation workflow

This section outlines the experiments to be performed to validate that the artifact is functional and can reproduce the key results (figures) of our paper.

#### A.4.1 Major Claims.

- (C1): The artifact is functional and can successfully generate the main figures presented in the paper using the provided data and scripts. This is demonstrated by experiments (E1) through (E9), which correspond to Figures 2, 3, 5, 6, 7, 8, 9, 10, 11 and 12 in the paper.
- A.4.2 Experiments. The following experiments will generate the figures from the paper. The execution time may vary depending on the hardware configuration. The times given were measured on an Apple M3 Pro processor.

Experiment (E1): Generate Figure 2 [1 human-minute + 3 compute-seconds]: This experiment generates the timeseries power consumption plot. [Execution] Run the following command from the root of the repository:

python3 src/Sec-03\_temporally-shared/Fig-02\_TS-power.py

[Results] A PDF file named TS-power.pdf will be created in the figures/ directory, corresponding to Figure 2 in the paper.

Experiment (E2): Generate Figure 3 [1 human-minute + 7 compute-seconds]: This experiment generates the performance vs. power plot for temporal sharing. [Execution] python3

src/Sec-03\_temporally-shared/Fig-03\_TS-perf-power.py

[Results] Generates TS-perf-power.pdf in the figures/ directory.

Experiment (E3): Generate Figure 5 [1 human-minute + 5 compute-seconds]: Generates the power consumption plot for MIG instances. [Execution]

python3 src/Sec-04\_spatially-shared/Fig-05\_MIG-GI-power-all.py

[Results] Generates MIG-GI-power-all.pdf in the figures/ directory.

Experiment (E4): Generate Figure 6 [1 human-minute + 3 compute-seconds]: Generates the MIG driver overhead plot. [Execution]

python3 src/Sec-04\_spatially-shared/Fig-06\_MIG-driver.py

[Results] Generates MIG-driver.pdf in the figures/ directory.

Experiment (E5): Generate Figure 7 [1 human-minute + 10 compute-seconds]: Generates the power and performance plot for 1-slice allocations on H100. [Execution] python3 src/Sec-04\_spatially-shared/Fig-07\_MIG-GI-bench-zoom.py

[Results] Generates MIG-GI-bench-zoom.pdf in the figures/ directory.

Experiment (E6): Generate Figure 8 [1 human-minute + 10 compute-seconds]: Generates the benchmark performance on MIG instances. [Execution]

python3 src/Sec-04\_spatially-shared/Fig-08\_MIG-GI-bench-all.py

<span id="page-16-0"></span>[Results] Generates MIG-GI-bench-all.pdf in the figures/ directory.

Experiment (E7): Generate Figure 9 [1 human-minute + 5 compute-seconds]: Generates the Pearson correlation heatmap for pass-through GPUs. [Execution] python3 src/Sec-05\_pass-through/Fig-09\_PT-pearson-corr-ipmi.py [Results] Generates PT-pearson-corr-ipmi.pdf in the figures/ directory.

Experiment (E8): Generate Figure 10 [1 human-minute + 6 compute-minutes]: Generates the power density plot for 4 A100 GPUs. [Execution]

python3

src/Sec-05\_pass-through/Fig-10\_PT-density-4A100-4states-corrected.py [Results] Add to the figures/ directory the following plot: PT-density-4A100-4states-corrected.pdf

Experiment (E9): Generate Figure 11 [1 human-minute + 1.5 compute-minutes]: Generates the power density plot for 8 A100 GPUs. [Execution]

python3

src/Sec-05\_pass-through/Fig-11\_PT-density-8A100-2states-corrected.py

[Results] Add to the figures/ directory the following plot: PT-density-8A100-2states-corrected.pdf

Experiment (E10): Generate Figure 12 [1 human-minute + 4 compute-seconds]: Generates the plot showing GPU temperature, power, and utilization. [Execution] python3 src/Sec-05\_pass-through/Fig-12\_WC-GPU-temp-pwr-util.py [Results] Generates WC-GPU-temp-pwr-util.pdf in the figures/ directory.

#### A.5 General Notes

For full transparency, the experiments/ directory contains the scripts and configurations used to conduct the original experiments. These scripts handle workload deployment (Blender, YOLO, etc.), power data collection (DCGM, IPMI), and GPU sharing mode configuration. While reproducing these experiments is not the goal of this evaluation due to hardware constraints, we provide them for informational purposes.

Received 15 May 2025; revised 26 September 2025