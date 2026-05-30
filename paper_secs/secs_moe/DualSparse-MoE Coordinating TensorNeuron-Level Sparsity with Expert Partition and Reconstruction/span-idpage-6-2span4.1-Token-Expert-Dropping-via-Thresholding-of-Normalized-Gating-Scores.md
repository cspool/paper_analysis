# <span id="page-6-2"></span>4.1 Token-Expert Dropping via Thresholding of Normalized Gating Scores

Since each token-expert computation is weighted by its corresponding gating score, a lower score indicates a lesser contribution to the final result. In the extreme case where the gating score is zero, the computation has no effect.

To improve efficiency, we propose an operation termed "1T-Drop", which selectively drops token-expert computations whose normalized gating scores fall below a specified threshold ( $T^1_{drop}$ ). Specifically, for each input token at each MoE layer, we normalize the gating scores of the Top-K activated experts and only retain experts whose normalized gating scores exceeding the threshold. The output of each token-expert computation remains weighted by its original gating score. It is worth noting that for some MoE models [29, 56] already normalize the gating scores of activated experts, this additional normalization step is unnecessary.

Interestingly, our empirical results show that applying a low threshold (approximately 0.05) for dropping computations can even improve accuracy, as illustrated in Figure 7. Across all benchmarks, the highest accuracy is achieved when some token-expert computations are dropped, suggesting that computations with very low gating scores may negatively impact overall performance. However, as the threshold increases further, thereby dropping more computations, the accuracy decreases across benchmarks. Note that the accuracy sensitivity to the drop operation also varies by task. For instance, GSM8K [11], a benchmark evaluating mathematical reasoning ability, exhibits the most pronounced accuracy decline as the drop rate increases.

#### <span id="page-6-0"></span>4.2 Dual-Threshold Token-Expert Dropping with Expert Partition and Reconstruction

Directly dropping token-expert computations based on a single threshold of normalized gating scores (1T-Drop) can lead to accuracy degradation, particularly at higher thresholds. Motivated by our observed dual sparsity in MoE models (Figure 1), we propose a dual-threshold token-expert dropping strategy, referred to as "2T-Drop". It coordinates both tensor-level and neuron-level sparsity to alleviate accuracy degradation while preserving computational savings and efficiency gains. As illustrated in Figure 8, 2T-Drop consists of the following three key operations:

- (a) Expert Partition. We employ the expert partition (partial transformation) method to enhance tensor-level sparsity, enabling finer-grained and thus more flexible combinations of token-expert computations dropping at the tensor level.
- (b) Expert Reconstruction. To exploit neuron-level sparsity within each expert, we perform neuron importance profiling on calibration samples. Neurons are then reorganized to reconstruct a major sub-expert comprising neurons of higher importance and a minor sub-expert comprising those of lower importance. Note that in our implementation, expert partitioning and reconstruction are executed as a unified process: all neurons in an original expert are first profiled and then reorganized into two separate sub-experts, one major and one minor. We employ this static approach to leverage neuron-level sparsity and avoid the challenges of dynamically identifying neuron activations for runtime dropping.

Furthermore, we experiment with various neuron importance profiling methods within SwiGLU experts: (1) accumulated gate value

$$Importance = \sum Swish(\mathbf{x} \cdot \mathbf{W}_1^{neuron}), \tag{14}$$

| Model                      | E-Activ./Total | $T_{drop}^1$ | Drop Rate | ARC-C | BoolQ | GSM8K | HellaSwag | MMLU  | OBQA  | PIQA  | RTE   | WinoGrande | AVG.(↑)      |
|----------------------------|----------------|--------------|-----------|-------|-------|-------|-----------|-------|-------|-------|-------|------------|--------------|
|                            | 2/8            | -            | 0         | 59.47 | 85.14 | 58.07 | 84.05     | 67.13 | 47.00 | 83.79 | 70.40 | 76.56      | 70.18        |
| Mixtral-8×7B               | $4/16 \ (P=2)$ | -            | 0         | 59.56 | 85.32 | 58.30 | 84.02     | 67.05 | 47.20 | 83.41 | 70.76 | 76.01      | 70.18        |
|                            | $8/32 \ (P=4)$ | -            | 0         | 59.47 | 85.26 | 58.07 | 83.99     | 67.22 | 46.80 | 83.46 | 70.76 | 76.72      | <u>70.19</u> |
| Fine-Tuned<br>Mixtral-8×7B | 2/8            | -            | 0         | 60.58 | 87.06 | 60.73 | 82.99     | 64.92 | 46.20 | 83.62 | 71.84 | 76.87      | 70.53        |
|                            | $4/16 \ (P=2)$ | -            | 0         | 59.56 | 87.06 | 62.85 | 82.96     | 65.65 | 47.00 | 83.3  | 72.92 | 76.48      | 70.86        |
| Mixuai-ox/D                | $8/32 \ (P=4)$ | -            | 0         | 60.67 | 87.55 | 62.85 | 83.06     | 65.10 | 47.60 | 83.46 | 72.92 | 76.87      | 71.12        |
| Fine-Tuned                 | 2/8            | 0.30         | 20.3%     | 59.39 | 87.06 | 61.84 | 82.72     | 64.26 | 46.40 | 82.86 | 71.48 | 76.64      | 70.29        |
| Mixtral-8×7B               | $4/16 \ (P=2)$ | 0.15         | 21.0%     | 59.64 | 87.00 | 63.46 | 82.58     | 64.75 | 46.40 | 83.13 | 73.65 | 76.24      | 70.76        |
| Threshold Drop             | 8/32 (P = 4)   | 0.08         | 23.9%     | 59.73 | 87.31 | 62.85 | 82.75     | 64.76 | 47.00 | 83.03 | 74.01 | 76.48      | 70.88        |

<span id="page-7-0"></span>**Table 1.** Comparison of downstream accuracy between the original Mixtral-8×7B model and its expert-partitioned variant.

(2) accumulated absolute gate value

$$Importance = \sum \left| Swish(\mathbf{x} \cdot \mathbf{W}_{1}^{neuron}) \right|, \tag{15}$$

(3) accumulated gate-up value

$$Importance = \sum (Swish(\mathbf{x} \cdot \mathbf{W}_1^{neuron}) \odot (\mathbf{x} \cdot \mathbf{W}_3^{neuron})), (16)$$

(4) accumulated absolute gate-up value

$$Importance = \sum \left| Swish(\mathbf{x} \cdot \mathbf{W}_1^{neuron}) \odot (\mathbf{x} \cdot \mathbf{W}_3^{neuron}) \right|. (17)$$

Here,  $\mathbf{W}_1^{\text{neuron}}$  and  $\mathbf{W}_3^{\text{neuron}}$  denote each neuron's  $\mathbf{W}_1$  and  $\mathbf{W}_3$  weights, following the formulation of the SwiGLU expert in Equation (4). Empirically, we observe that different models exhibit varying affinities for different profiling methods, highlighting the need to empirically determine the optimal configuration for each specific model.

In addition, we have considered partitioning experts into more granular based on neuron importance, which may yield higher accuracy. However, since this approach could reduce computational intensity and lead to low GPU utilization, we choose to partition and reconstruct each original expert into only two sub-experts.

(c) Dual-Threshold Drop. Building on the reconstructed minor and major experts, we propose the dual-threshold drop (2T-Drop) method. This approach applies token-expert computation dropping using a higher threshold-minor ( $T_{minor}^2$ ) for minor sub-experts and a lower threshold-major ( $T_{major}^2$ ) for major sub-experts. Specifically, original experts with gating scores above  $T_{minor}^2$  are fully engaged in computation, while those with gating scores below  $T_{major}^2$  are entirely dropped—similar to the 1T-Drop method. Uniquely, experts with gating scores between  $T_{minor}^2$  and  $T_{major}^2$  compute only the major half of their neurons. Based on our empirical experiments, we select dual thresholds of  $T_{major}^2 = T_{drop}^1 - 0.01$  and  $T_{minor}^2 = T_{drop}^1 + 0.01$ , which preserve a similar drop rate while achieving higher inference accuracy.

Given that our approach affects the computation granularity of token-expert grouped-GEMM and introduces additional control operations in the gating function, we optimize the corresponding Triton kernel to enhance efficiency.

#### 4.3 Load-Aware Thresholding in Expert Parallelism

Load imbalance among distributed devices is a major factor limiting efficiency in MoE model inference with expert parallelism. Since the overall MoE computation is blocked by the device with the heaviest computational load, simply dropping computations uniformly across devices can unnecessarily degrade accuracy on devices with lighter workloads.

To address this, we propose a load-aware thresholding mechanism that dynamically adjusts token-expert dropping based on the load of each device. This approach enables the system to adaptively balance computation across devices while maintaining high accuracy.

As shown in Figure 8(d), we employ a step-down thresholding strategy: devices with higher workloads apply higher thresholds, dropping more token-expert computations, while devices with lighter workloads use lower thresholds. To minimize control overhead in distributed environments, we calculate the ratio of the actual load to the ideal balanced load for each device. If this ratio exceeds 1, the threshold is set to a predefined maximum value; if it is below 1, the threshold is proportionally reduced according to the deviation from 1. This method ensures that all devices drop computation as little as possible, while maintaining their workload at or below that of the originally most-loaded device. By incorporating load-aware thresholding in expert parallelism, our approach achieves higher inference accuracy while maintaining the same level of acceleration.

#### 5 Evaluation

#### 5.1 Experimental Setup

To evaluate the efficacy of our proposed methods, we conduct experiments on a server equipped with 8 Nvidia H20 GPUs. Specifically, we utilize EleutherAI's LM-Evaluation-Harness [16] to assess model quality, reporting either accuracy or normalized accuracy for each benchmark, as applicable. Our evaluation tasks include zero-shot evaluations on the ARC-C [10], BoolQ [9], HellaSwag [57], MMLU [19], OBQA [34], PIQA [5], RTE [55], and WinoGrande [45] benchmarks, as well as 5-shot evaluation on GSM8K [11]. We utilize the Tulu-3-sft-mixture dataset [25] for our fine-tuning experiments.

<span id="page-8-1"></span>![](_page_8_Figure_1.jpeg)

![](_page_8_Figure_2.jpeg)

Figure 9. Comparison of communication bandwidth across different input sizes using ETP and S-ETP. In real-world tests (a), "E2T4" denotes a configuration with EP=2 and TP=4, while "E4T2" denotes a configuration with EP=4 and TP=2. In simulation (b), NVL72 [\[38\]](#page-12-27) is configured with EP=9 and TP=8, whereas CloudMatrix384 (CM384) [\[62\]](#page-13-17) is configured with EP=48 and TP=8.

Furthermore, we implement our proposed DualSparse-MoE inference system and evaluate its acceleration effectiveness upon the SGLang framework [\[59\]](#page-13-18), which supports efficient distributed inference for prevailing MoE models such as Mixtral [\[21\]](#page-12-2), OLMoE [\[35\]](#page-12-4), and DeepSeek [\[28\]](#page-12-3).

Additionally, we perform small-scale real-world tests using the PyTorch Distributed framework with the NCCL backend, as well as large-scale simulations using the ASTRA-SIM simulator [\[43\]](#page-13-19) to evaluate the communication optimization achieved by the Soft Expert-Tensor Parallelism (S-ETP).

#### <span id="page-8-0"></span>5.2 Evaluation of Expert Partition

We conduct experiments to substantiate the benefits of promoting tensor-level sparsity during the post-training phase, using our proposed expert partition methods.

<span id="page-8-2"></span>5.2.1 Model Quality Gains during Fine-tuning. We apply the expert partition (complete transformation) to the Mixtral-8×7B model, partitioning its original 8 experts into 16 ( = 2) and 32 ( = 4) finer-grained experts. As shown in Table [1,](#page-7-0) the partitioned models demonstrate the same downstream accuracy, with only negligible fluctuations. This consistency is attributed to the mathematical equivalence maintained by the partitioning process, although minor variations may arise due to floating-point precision errors. While models with partitioned experts exhibit significantly lower fine-tuning loss curves in Figure [4,](#page-4-0) these partitioned models also achieve higher downstream accuracy after fine-tuning. Notably, even when applying a 1T-Drop with a 23.9% drop rate to the partitioned model ( = 4), this model still achieves a higher average downstream accuracy of 70.88% than the 70.53% accuracy attained by the fine-tuned original model.

