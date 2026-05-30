# A.4.1 Major Claims.

- Claim (C1): Samoyeds achieves an average speedup of 1.99× over baselines, as shown in Figure 12 and 13. This is proven by experiment (E1).
- Claim (C2): Samoyeds outperforms the baseline by 1.45× on average, as shown in Figure 14. This is validated by experiment (E2).
- Claim (C3): Samoyeds 1.42× improves overall model performance by 1.42× (Figure 15) and delivers superior throughput across different batch sizes (Figure 16). This is confirmed by experiment (E3).
- Claim (C4): Different optimizations of Samoyeds provide speedup according to our breakdown analysis (Figure 17). This can be reproduced with experiment (E4).
- Claim (C5): The optimization of Samoyeds does not affect the model accuracy, as shown in Table 4 and 5. This is verified by experiment (E5).
- Claim (C6): Samoyeds exhibits superior portability compared to baselines, as shown in Figure 18. This is proven by experiment (E6).

- A.4.2 Experiments. The hardware requirements for each experiment are as follows:
  - E1, E2, E3, and E6: These experiments can be conducted on a single GPU, such as the NVIDIA GeForce RTX 4070 Super used in our paper.
  - E4: This experiment involves post-training of models, which may require high-end GPUs such as the A100- 80G used in our paper.
  - E5: This experiment analyzes performance portability and requires multiple GPUs with different architectures (e.g., RTX 3090, RTX 4070 Super, RTX 4090, and A100, as used in our paper).
  - Experiment (E1): To reproduce the kernel level results (Figure 12, 13), execute:
  - 1 ./artifacts/kernel/synthetic\_scripts.sh
  - 2 ./artifacts/kernel/kernel\_model\_config\_scripts.sh

Figure 12 and 13 can be plotted with following files:

- 1 ./artifacts/kernel/figure12\_plot.ipynb
- 2 ./artifacts/kernel/figure13\_plot.ipynb
- Experiment (E2): To reproduce the MoE module level results (Figure 14), execute:
- 1 ./artifacts/MoE/figure14\_scripts.sh

Figure 14 can be plotted with following files:

- 1 ./artifacts/MoE/figure14\_plot.ipynb
- Experiment (E3): To reproduce the end-to-end level results (Figure 15, 16), execute:
- 1 ./artifacts/model/figure15\_scripts.sh
- 2 ./artifacts/model/figure16\_scripts.sh

Figure 15 and 16 can be plotted with following files:

- 1 ./artifacts/model/figure15\_plot.ipynb
- 2 ./artifacts/model/figure16\_plot.ipynb
- Experiment (E4): To reproduce the breakdown analysis results (Figure 17), execute:
- 1 ./artifacts/MoE/figure17\_scripts.sh

Figure 17 can be plotted with following files:

- 1 ./artifacts/MoE/figure17\_plot.ipynb
- Experiment (E5): We provide several scripts to reproduce the results of model accuracy (Table 4, 5). The following scripts require execution on high-memory GPUs or multi-GPU configurations. Specifically: (1) The script for collecting data in Table 4 is configured to utilize a cluster of 4 GPUs; (2) The scripts for collecting data in Table 5 must be run on an NVIDIA A100 80GB GPU to avoid Out-Of-Memory (OOM) errors. Lower-capacity GPUs may not have sufficient memory to handle these operations.
- 1 cd sparseml
- 2 # Table 4
- 3 bach benchmark/scripts/samoyeds\_gradual\_pair.sh
- 4 # Table 5
- 5 bash benchmark/scripts/samoyeds\_qwen2\_80G.sh
- 6 bash benchmark/scripts/samoyeds\_tiny\_llama\_80G.sh

- <span id="page-17-0"></span>The results are stored in the ./benchmark/output\_dir/ directory.
- Experiment (E6): To reproduce the performance portability results of Samoyeds (Figure 18), the following script need to run on multiple GPUs, including NVIDIA GeForce RTX 3070, NVIDIA GeForce RTX 4070 Super, NVIDIA GeForce RTX 4090, and NVIDIA A100.
- 1 ./artifacts/kernel/synthetic\_scripts.sh
  - Figure 18 can be reproduced by collecting results on different GPUs into ./artifacts/results/kernel/ folder. Figure 18 then can be plotted with following files:
- 1 ./artifacts/MoE/figure18\_plot.ipynb