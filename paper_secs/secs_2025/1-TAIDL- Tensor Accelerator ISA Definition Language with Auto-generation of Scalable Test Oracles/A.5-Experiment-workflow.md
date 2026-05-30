# A.5 Experiment workflow

The experimental workflow has been encapsulated within bash scripts. These scripts set up appropriate Docker containers, run the simulation experiments, and generate the final plots.

To kick-the-tires, run ./scripts/kick-tires.sh from the base directory. This will generate plots from pre-saved data without running any experiments.

To perform a subset of the evaluation, run ./scripts/lite.sh from the base directory. This will generate plots by benchmarking TAIDL-TOs for Gemmini and oneDNN kernels. This will also perform correctness testing of TAIDL-TO using pre-generated inputs and golden outputs. This takes only 4-5 minutes.

To perform the complete evaluation, run ./scripts/full.sh from the base directory. This will generate plots by benchmarking TAIDL-TOs and the baselines – Gemmini Spike and Intel SDE. This will also generate inputs and golden outputs using Gemmini Spike and Intel SDE for correctness testing of TAIDL-TO. This may take around 30-45 minutes and will not run on ARM machines.

For more details, refer to the README.md in the artifact.

