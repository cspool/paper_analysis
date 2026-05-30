# Artifact Analysis (incl. Outputs)

1.1: Figure 9 can be found in the directory exps-1.0/imgs/ by executing python exps-1.0/fig9.py

1.2: The data in Table 3 can be obtained by executing python exps-1.0/table3.py.

1.3: Figure 10 can be found in the directory exps-1.0/imgs/ by executing python exps-1.0/fig10.py.

1.4: Figure 11 can be found in the directory exps-1.0/imgs/ by executing python exps-1.0/fig11.py.

# Artifact Analysis (incl. Outputs)

Fig.9: The experiment evaluates the overall performance of HyTiS in accelerating matrix multiplication, using cuBLAS, Split-K, Stream-K, Inductor-Triton, and two ablated variants, HyTiS(L1) and HyTiS(STL). All methods are benchmarked on representative

workloads, with performance measured by normalized speedups. On the H100 GPU, HyTiS achieves an average speedup of 1.12× over cuBLAS, with a maximum of 1.95×. It consistently outperforms Split-K and Stream-K, and delivers an average speedup of 1.22× over Inductor-Triton. Compared to its ablations, HyTiS outperforms HyTiS(L1) by 1.06× in 41% of the cases, and HyTiS(STL) by up to 1.32×, with an average of 1.04× in 15% of the cases. These results confirm the effectiveness of HyTiS across diverse configurations.

Table.3: To evaluate the contributions of HyTiS's core components, hybrid tile scheduling and adaptive tile layout tuning, a breakdown analysis is conducted using three key metrics: speedup (relative to cuBLAS runtime), SM balance (measuring warp-level load uniformity across streaming multiprocessors), and DRAM read volume (indicating memory traffic from DRAM to L2 cache). Each metric is normalized to the cuBLAS baseline and categorized into three performance regions: low ([0, 0.98), indicating degradation), mid ([0.98, 1.02], indicating parity), and high ((1.02, ∞), indicating improvement). The full HyTiS system is expected to yield higher proportions in the high region across all metrics compared to its ablated variants, HyTiS(L1) and HyTiS(STL). On the H100 GPU, HyTiS achieves an average speedup of 1.12×, with 81% of cases in the high region. For SM balance, it reaches an average improvement of 3.2×, with only 32% of cases in the low region. In terms of DRAM read, HyTiS reduces low-region cases to 20% and increases high-region cases to 28%, demonstrating improved memory efficiency.

Fig.10: To evaluate the effectiveness of HyTiS in mitigating wave quantization artifacts, the experiment measures GEMM execution time while varying the matrix size M, with N and K held constant. Two settings are considered: Figure 10(a) uses N and K values of 1024 and 4096, respectively, while Figure 10(b) uses 2048 and 8192. The evaluated methods include cuBLAS, Inductor-Triton, and HyTiS. Results are divided into two regions: the quantization-prominent region (highlighted in orange), where latency fluctuations typically occur due to inefficient wave alignment, and the non-quantizationprominent region, where execution time remains relatively stable. HyTiS is expected to reduce performance variability and improve average execution time, particularly in the quantization-prominent region where performance cliffs are commonly observed.

Fig.11: To analyze the impact of HyTiS's tuning hyperparameters, the experiment measures how varying the values of 1 and 2 affects performance and search space size as the input matrix size (and thus the number of virtual tiles) increases. The figure reports the optimal values of 1 and 2 selected across different workloads, along with the resulting tuning space size. As the number of virtual tiles increases, indicating larger problem sizes, the value of 1 converges toward 1, suggesting that a smaller set of high-throughput kernels suffices for the first-level scheduling. In contrast, <sup>2</sup> fluctuates modestly within the range of 1.0 to 1.3, showing that the second-level tuning space remains relatively stable. These results confirm that HyTiS adapts well to varying input scales, and that its hyperparameters offer intuitive control over the trade-off between tuning cost and search flexibility.

<span id="page-14-1"></span><sup>1</sup>https://doi.org/10.5281/zenodo.15244191