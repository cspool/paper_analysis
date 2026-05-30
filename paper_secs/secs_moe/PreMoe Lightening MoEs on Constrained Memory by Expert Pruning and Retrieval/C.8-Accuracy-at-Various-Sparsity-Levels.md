# **C.8 Accuracy at Various Sparsity Levels**

**Specialist vs. Generalist Degradation Under Extreme Sparsity.** Comparing Tables [C.8](#page-18-0) and [C.9,](#page-18-1) we observe that domain-specific specialists degrade more gracefully at extreme sparsity than multi-domain generalists. At 75% sparsity, the specialist average (62.83%) significantly outperforms the generalist (49.46%). The gap is particularly pronounced on GPQA (45.45% vs 38.38%) and especially LCB (49.63% vs 20.22%). This indicates that when expert budgets are severely constrained, dedicating the limited experts to a single domain yields better task-specific performance than attempting multi-domain coverage. This finding suggests a trade-off between model versatility and efficiency: at moderate sparsity (50%), the generalist retains broad capabilities with minimal loss, but at extreme sparsity (75%), domain-focused specialists become the more practical choice.

### **C.9 Complete Deployment Efficiency Metrics**

<span id="page-18-2"></span>Table C.10: Complete deployment efficiency metrics for DeepSeek-R1 at various sparsity levels. Size refers to the number of model parameters in billions (B). All measurements are taken on servers with Ascend 910B2-64GB NPUs.

| Sparsity | # Experts                              | # NPUs   | Latency (ms/tok) | Size (B)      | Throughput (tok/s) |
|----------|----------------------------------------|----------|------------------|---------------|--------------------|
| 0%       | 256                                    | 64       | 115.35           | 670.92        | 52.01              |
| 50%      | 128                                    | 32       | 93.72            | 343.96        | 64.02              |
| 75%      | 64                                     | 16       | 73.20            | 180.49        | 81.97              |
|          | Relative Improvement (vs. 0% sparsity) |          |                  |               |                    |
| 50%      | –                                      | 2× fewer | 1.23× faster     | 1.95× smaller | 1.23× higher       |
| 75%      | –                                      | 4× fewer | 1.58× faster     | 3.72× smaller | 1.58× higher       |

These metrics demonstrate that PreMoE enables significant infrastructure savings while maintaining high accuracy. At 50% sparsity, we halve the deployment cost (NPUs and parameters) while improving inference speed by 23%. At 75% sparsity, we achieve 4× NPU reduction and 58% throughput improvement, though with accuracy trade-offs on some benchmarks as shown in the main text.

