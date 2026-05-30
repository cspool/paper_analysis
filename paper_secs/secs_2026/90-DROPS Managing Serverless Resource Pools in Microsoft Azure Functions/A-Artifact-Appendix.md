# A Artifact Appendix

#### A.1 Abstract

Our artifact is an implementation of the DROPS algorithm, together with the simulator required to run the experiments described in the paper. Speci!cally, it contains the following items:

- The source code of the serverless platform simulator, written in C#, which includes:
  - Our reference implementation of DROPS.
  - Our implementation of alternative resource optimization methods.
  - Our reference implementation of the aggressive container creation optimization.
- The tools to train the forecasting models.
- The tools to generate the !gures in this paper.

## A.2 Description & Requirements

A.2.1 How to access. The artifact is publicly available in the following GitHub repository:

https://github.com/UWASL/DROPS.git

A snapshot of this repository is archived at:

https://zenodo.org/records/17051480

## A.2.2 Hardware dependencies. To run our experiments, you need the following:

- A machine running Linux. Our code is tested on Ubuntu 22.04. However, it should work on other operating systems, including Windows and MacOS.
- For model training, at least 128 GB of RAM is required to train machine learning models. However, training models is optional: we provide the output trace of the trained models that is needed to reproduce the results in the paper without retraining models.

A.2.3 Software dependencies. Our code has been tested on Ubuntu 22.04. The simulator requires the .NET 9.0 Runtime. The training process requires the AutoGluon library. The plotting scripts require Python 3, along with the pandas, numpy, and matplotlib libraries.

A.2.4 Benchmarks. Reproducing the results in the paper requires two traces (i.e., container-allocation trace and lifecycle trace). Follow the steps under the set-up section (A.3) to download the traces.

## A.3 Set-up

To download DROPS' source code, run the following: git clone https://github.com/UWASL/DROPS.git Then, install all dependencies:

./setup-scripts/install-dep.sh

./training/training-dep.sh

Then, build DROPS:

./setup-scripts/build.sh

Then, run the following command to fetch the input traces.

./setup-scripts/fetch-traces.sh The environment is now ready to run all experiments.

## A.4 Evaluation work#ow

Please see the README.md !le in the artifact repository for a detailed description of the process. The rest of this section discusses our major claims and the experiments that support them.

## A.4.1 Major Claims. Our papermakes the followingmajor claims.

- C1. DROPS meets the target SLO while signi!cantly reducing the cost compared to other approaches. Experiments 1 and 2 and Figure 7 and 12 support this claim.
- C2.DROPS enables precise pool sizing for di"erent SLO levels, ensuring that the failure rate is below the target SLO. Experiment 2 and Figure 12 support this claim.
- C3. Prediction-based approaches fail to meet the target SLO, and their performance is signi!cantly a"ected by the prediction interval. Larger prediction intervals lead to a higher failure rate for the constant and Poisson approaches and alower failure rate for the concentratedload approach. Experiments 1 and 3 and Figures 8 and 7 support this claim.
- C4. The reactive approach fails to meet the target SLO, andits performanceis signi!cantly a"ected by the scaleup and scale-down factors. Increasing the scale-up factor reduces the failure rate andincreases the cost. On the other hand, increasing the scale-down factor increases the failure rate and reduces the cost. Experiment 4 and Figure 11 support this claim.
- C5. The aggressive container creation optimization reduces the failure rate without incurring additional cost. Experiment 5 and Figure 14a support this claim.

A.4.2 Experiment E1: Performance Comparison [0.5 compute-hours]. This experiment compares the failure rate, cost, and ful!llment latency of di"erent optimization methods. The results are used to generate Figure 7 and support claims C1, C3, and C4.

[Preparation] None.

[Execution] Run the following commands:

cd /<repo\_root>/experiments/

./scripts/fig6.sh

[Results] Two !gures similar to Figure 7a and Figure 7b are generated under/<repo\_root>/experiments/directory. Raw results for failure rate, cost breakdown, and latency are generatedincsvformat under/<repo\_root>/experiments/fig6 directory.

A.4.3 Experiment E2: DROPS Sensitivity Analysis [0.5 compute-hours]. This experiment shows that DROPS enables precise pool sizing for di"erent SLOs. The results are used to generate Figure 12 and support claim C2.

[Preparation] None.

## [Execution] Run the following commands:

cd /<repo\_root>/experiments/

./scripts/fig11.sh

[Results]Two !gures similar to Figure 12a and Figure 12b are generated under/<repo\_root>/experiments/directory. Raw results for failure rate, cost breakdown, and latency are generated in csv format under /<repo\_root>/ experiments/fig11 directory.

