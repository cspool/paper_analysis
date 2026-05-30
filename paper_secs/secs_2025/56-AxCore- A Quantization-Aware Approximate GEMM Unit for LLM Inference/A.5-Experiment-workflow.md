# A.5 Experiment workflow

The artifact evaluation is split into three main parts, each designed to reproduce a specific set of results from the paper.

- 1. Functional verification of AxCore hardware:
- (1) Please follow the detailed instructions provided in the file Hardware/AxCore/README.md.
- 2. Evaluation of LLM accuracy (reproduces Table 2 and Table 3):
  - (1) Create the Environment: Set up the Conda environment by following the instructions at [https://github.com/CLab-](https://github.com/CLab-HKUST-GZ/micro58-axcore/tree/main/Software/AxCore)[HKUST-GZ/micro58-axcore/tree/main/Software/AxCore.](https://github.com/CLab-HKUST-GZ/micro58-axcore/tree/main/Software/AxCore)
  - (2) Execute evaluation: Run the corresponding shell script for each table. The script will automatically download the required models and datasets from the Hugging Face Hub (if not cached) and then perform the AxCore evaluation.

## 3. Performance of the AxCore simulator (reproduces Figure 17):

- (1) Create the Environment: Set up the Conda environment as instructed at [https://github.com/CLab-HKUST-GZ/micro58](https://github.com/CLab-HKUST-GZ/micro58-axcore/tree/main/Software/axcore_simulator) [axcore/tree/main/Software/axcore\\_simulator.](https://github.com/CLab-HKUST-GZ/micro58-axcore/tree/main/Software/axcore_simulator)
- (2) Run Simulator and plot results: Execute the provided script to run all simulations. The final plot will be generated as results/fig\_17.pdf.
- 4. Gemm operations percentage (reproduces Figure 2) (optional):
  - (1) Calculate Workload Distribution: Run the profiling script to analyze the computational workload.
  - (2) Generate Visualization: Create the visualization chart. This generates figure2.pdf in the current directory.

