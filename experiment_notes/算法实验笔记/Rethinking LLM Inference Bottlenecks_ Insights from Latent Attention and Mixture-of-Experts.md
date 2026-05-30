## Rethinking LLM Inference Bottlenecks: Insights from Latent Attention and Mixture-of-Experts

- 属于算法pipeline的实现是什么？实验比较什么？
  - 实现：本论文并非提出新的算法模型，而是对 MLA（Multi-head Latent Attention）和 MoE 的算法特性进行系统级算术强度（ArI）分析。核心分析的算法优化是 **MLA 的 layer reordering（层重排）**：MLA 中 Q、K、V 经过低秩联合压缩（compressed latent space，$d_{\rm KVco}=512$），使用 decoupled RoPE 消除 Q 和 K 之间的非线性后，利用矩阵乘法结合律将 Score 层从 $\mathbf{S}_i = (\mathbf{C}_{\rm Q} \cdot \mathbf{W}_{\rm DQ_i}) \cdot (\mathbf{C}_{\rm KV} \cdot \mathbf{W}_{\rm DK_i})^T$ 重排为 $\mathbf{S}_i = (\mathbf{Q}_i \cdot \mathbf{W}_{\rm DK_i}^T) \cdot \mathbf{C}_{\rm KV}^T$，使得 decode 阶段 K 解压缩的代价降低 L 倍（只消与 $\mathbf{W}_{\rm DK}$ 相乘而非整个 $\mathbf{C}_{\rm KV}$ 解压缩），同时 Score 层读取的是压缩后的 $\mathbf{C}_{\rm KV}$ 而非完整解压缩的 KV$，将核心注意力层的 ArI 从 $\approx 1$ 提升到 $\approx 100$ Op/B（FlashMLA 优化后 $\approx 200$ Op/B），逼近现代加速器 ridge point。
  - 实验比较：(a) **MLA with/without reordering 的延迟与 ArI 对比**：prefill 和 decode 阶段的 K decompress 和 Score 层的 FLOPs、Memory Access、ArI 分析（Table III）；(b) **Attention block latency 对比**：reordering vs non-reordering，不同 batch size 和 sequence length 下各层执行时间占比（Figure 6）；(c) **核心注意力层 roofline 分析**：GPT-3 (MHA)、Llama4-Maverick (GQA)、DeepSeek-R1 (MLA) 各层在 H100 上的 ArI 与 ridge point 对比（Figure 3）；(d) **Layer reordering 对 prefill/decode 的不同影响**：decode 阶段延迟降低最多 103.12×，prefill 阶段延迟增加最多 2.21×；(e) **TP 对 reordered MLA 的影响**：因所有 head 共享 $\mathbf{C}_{\rm KV}$，TP 降低 ArI 为 $1/deg_{\rm TP}$，分析不同 $deg_{\rm TP}$ 和 batch size 下 attention block 的延迟（Figure 8）；(f) **MoE FC 层的 ArI 与 batch size 关系**：推导 $B_{\rm MoE} = RP_{\rm acc} \cdot n_e / n_k$ 公式。

- 硬件平台是什么，配置是什么。
  - 加速器：NVIDIA B200 GPU（BF16 2250 TFLOPS, 8000 GB/s 内存带宽, 192 GB HBM, ridge point 281.25 Op/B）作为主要评估平台。对比加速器包括 V100 SXM2 (125 TFLOPS, 900 GB/s, ridge point 138.89)、A100 SXM4 (312 TFLOPS, 2039 GB/s, ridge point 153.02)、H200 SXM5 (989.5 TFLOPS, 4800 GB/s, ridge point 206.15)、TPU v5P (459 TFLOPS, 2765 GB/s, ridge point 166)、TPU v7 (2307 TFLOPS, 7400 GB/s, ridge point 320.42)、MI325X (1307.4 TFLOPS, 6000 GB/s, ridge point 217.9)。
  - 真实硬件验证：DGX H100 系统。
  - 默认假设：32 B200 GPU 系统，NVLink 5th Gen 全互联（1.8 TB/s 双向带宽），遵循 NVL72 拓扑。

- 模型是什么。数据集和bench分别是什么。
  论文未使用任何模型或数据集。

- 开源情况。
  论文未使用任何模型或数据集。
