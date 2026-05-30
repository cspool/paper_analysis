# 3 Method

#### 3.1 Redesign of Parallelsim

The primary bottleneck for MoE (Mixture of Experts) serving is memory constraints due to the duplication of parameters. These parameters include those used for Attention, Normalization, and Shared Expert components. In the traditional Expert Parallelism (EP) mechanism, each machine replicates all of these shared parameters, resulting in significant memory usage. This duplication restricts the ability to handle longer contexts and larger batch sizes during inference.

Our observation highlights a key distinction between MoE training and inference: the nature of communication overhead. During inference, communication typically occurs between machines within the same node, as opposed to the more costly multi-node communication during training. Thus, IFMoE employs a combined Expert Parallelism (EP) and Tensor Parallelism (TP) approach for inference shown at figure [1.](#page-1-0) TP is used for shared parameters, while EP is retained for expert-specific

parameters. This choice is based on the observation that the size of individual experts is relatively small and load balancing across fine-grained MoE experts is generally efficient. Instead of the traditional All-to-All operation used in classic EP mechanisms, IFMoE adopts a double All-Gather operation, which will not introduce higher communication cost. Meanwhile, this hybrid EP+TP parallelism approach provides additional memory for computation and kv-cache storage, ultimately enabling higher throughput in MoE inference. The details are available in Appendix A.

## 3.2 Draft-Decoding and KV-cache revision

Similar to traditional Transformer architectures, fine-grained MoE (Mixture of Experts) models consist of Attention and MLP component within each layer. In the MLP layer, the computation is divided into two phases: routing expert (RE) calculation and shared expert (SE) calculation. During the RE phase, a fusion operation kernel called GroupedGEMM (Grouped General Matrix Multiplication) is employed to accelerate the computation of expert outputs for each token. A detailed discussion of the GroupedGEMM kernel can be found in Appendix B.

The figure illustrates the proportion of latency attributed to various operations during inference, highlighting that the GroupedGEMM kernel accounts for a significant portion of the overall latency. The slow performance arises from two main factors. First, GroupedGEMM is a memory-bound operation. Although the memory footprint for a single expert is relatively small, the number of activated experts grows nearly linearly as batch size increases, leading to heightened memory pressure until all experts are activated. Second, the dynamic control flows present in MoE models further contribute to the performance bottleneck. Specifically, the routing expert (RE) calculations do not benefit from optimizations such as Torch Compile and CUDA Graphs [\[1\]](#page-4-7), thus slowing down the computation.

To address these challenges, we introduce the concept of Speculative Decoding (SP). We observe that fine-grained MoE models can maintain strong performance with fewer activated experts thus instead of using a separate, smaller draft model, we utilize the finegrained MoE model with fewer experts itself as the draft model. Since fewer experts are activated during the GroupedGEMM operation, the decoding process is significantly faster compared to using the full model. In contrast to traditional speculative decoding algorithms, we accept the entire output from the draft model but only update the kv-cache for the generated tokens during the verification stage. The complete algorithm is provided in Algorithm [1.](#page-2-0)

```
Input: α, encode_topk Ek, decode_topk Dk,
fine-grained MoE model M
Initialize: terminate = False
buffer = []
while not terminate do
  for each step in α do
    buffer.append(M.decode(topk = Dk))
  end for
  # Revise KV Cache
  M.encode(buffer, topk = Ek)
```

terminate = detect\_terminate()

<span id="page-2-0"></span>Algorithm 1 IFMoE Decoding

buffer = []

end while

The key insight for algorithm [1](#page-2-0) is that both the draft model and the full model share the same kv-cache during inference. This modification not only improves the efficiency of the entire decoding process but also ensures minimal impact on overall performance. .

<span id="page-2-1"></span>Table 1: Downstream performance is evaluated for both the full model and IFMoE variants. DL refers to the Deepseek-Lite-Chat model, while Qwen2 denotes the Qwen2-57B-A14B-Instruct model. For the hyperparameter settings, we apply α = 10, encode topk E<sup>k</sup> = 6, and decode topk D<sup>k</sup> = 2.

| Task           | DL   | QWEN2 | DL-IFMoE | Qwen2-IFMoE |
|----------------|------|-------|----------|-------------|
| XSum           | 12.6 | 13.7  | 12.7     | 13.5        |
| GSM8K          | 67.7 | 75.4  | 63.8     | 71.1        |
| TruthfulQA-Gen | 43.6 | 47.2  | 43.0     | 45.9        |
| IFEval         | 42.9 | 65.7  | 42.3     | 64.8        |

