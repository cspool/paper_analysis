## Rethinking LLM Inference Bottlenecks: Insights from Latent Attention and Mixture-of-Experts

- baseline方法是什么？
  Baseline 是传统 transformer LLM（GPT-3、Llama4-Maverick）的 serving 部署，使用 MHA 或 GQA + dense FFN。Baseline 的核心缺陷：(1) **MHA/GQA 的核心注意力层 ArI 极低（≈1 Op/B）**——每个请求的 KV$ 和 attention score 值独占，无法跨批次请求共享，导致 decode 阶段始终 memory-bound，计算资源严重利用不足；(2) **大 KV$ 容量限制 batch size**——GPT-3 的 KV$ 单请求可达 9 GB（L=2048），限制了 $B_{\rm cap}$ 和 $B_{\rm SLO}$，使得 FC 层无法达到足够 batch size 来接近 ridge point，FC 层也处于 memory-bound；(3) **dense FFN 计算成本随参数量线性增长**——每个 token 需要计算所有参数，Scaling 受限。传统研究焦点：设计专用硬件（如 attention-specialized PIM）来缓解 attention 的内存带宽瓶颈。

  全栈执行例子（Baseline: GPT-3, 32 B200 GPU, L=4096, B=128）：
  - **算法Pipeline层**：MHA, 每 head 独立计算 Score = Q_i @ K_i^T → Softmax → Context = P_i @ V_i，ArI=1 Op/B（memory-bound）。Q/K/V 投影为 FC GEMM，无 KV 压缩。
  - **Serving/系统框架层**：Disaggregated prefill/decode, $deg_{\rm TP}=8$, $deg_{\rm DP}=4$。KV$ 每 token 4.5 MB(=12288×96×2×2B)，B=128 时 KV$ 总量 54 GB/GPU，接近 HBM 容量上限。无法扩大 batch size 来使 FC 层达到 ridge point。
  - **编译框架层**：论文未明确说明。
  - **Kernel调度层**：FlashAttention kernel 优化 attention 计算，但 ArI 仍 ≈1。FC 层使用 GEMM kernel，但因 batch size 受限，GEMV-like 特性导致 memory-bound。
  - **硬件架构层**：GPU HBM 提供 8000 GB/s 带宽，但因 ArI=1，实际吞吐远低于峰值。此前工作提倡 PIM 加速 attention（高 BW 匹配低 ArI）。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  论文方法不是提出新系统，而是对 MLA + MoE 架构进行系统级 ArI 分析，揭示新的瓶颈并给出三条设计原则：(1) **MLA + layer reordering 将核心注意力层的 ArI 从 ≈1 提升到 ≈100-200 Op/B**——通过低秩 KV 压缩（$d_{\rm KVco}=512$ vs $d_{\rm dec}=16384$）+ 矩阵乘法结合律重排+ decoupled RoPE，消除 decode 阶段的 K 解压缩开销（减少 L 倍），Score 层读取压缩的 $\mathbf{C}_{\rm KV}$ 而非完整解压缩的 K，内存访问量急剧降低；FlashMLA 进一步复用 Score 层加载的 $\mathbf{C}_{\rm KV}$ 到 Context 层，ArI 逼近 ridge point。(2) **MLA 的极小 KV$ + MoE 的稀疏专家激活形成 Synergy**——MLA 将每 token KV$ 从 4.5 MB 降至 68.6 KB（67× 缩小），释放内存容量用于扩大 batch size（$B_{\rm cap}$ 增大 60×），使 MoE 各 expert 的 FC 层获得足够 token 达到 ridge point。(3) **瓶颈从 memory bandwidth 转移至互联带宽和 expert 负载均衡**——互联带宽（NVLink vs InfiniBand）直接决定 all-to-all dispatch/combine 通信延迟；expert 分布偏斜（Zipfian s=0.8）导致热 expert 饱和造成吞吐量下降和延迟上升，小粒度部署（32 GPU×8）比大粒度（256 GPU monolithic）更能缓解偏斜。

  全栈执行例子（论文方法: DeepSeek-R1, 32 B200 GPU, L=4096, B=128）：
  - **算法Pipeline层**：MLA 低秩压缩 $\mathbf{C}_{\rm KV} \in \mathbb{R}^{L\times 512}$ 替代完整 KV$。Layer reordering: Score = $(Q_i W_{\rm DK_i}^T) @ C_{\rm KV}^T$ 替换 $Q_i @ (C_{\rm KV} W_{\rm DK_i})^T$，Context = $(\text{Softmax} @ C_{\rm KV}) @ W_{\rm DV_i}$。ArI：Score/Context ≈100 Op/B（FlashMLA 加倍至 ≈200），不再 memory-bound。MoE：每 token 仅激活 8/256 experts，计算量/参数比降至 37B/671B。
  - **Serving/系统框架层**：$deg_{\rm TP}=1$（reordered MLA 中 TP 无益），$deg_{\rm DP}=32$, $deg_{\rm EP}=32$。KV$ 每 token 68.6 KB，B=128 时 KV$ 总量仅 0.54 GB/GPU。$B_{\rm cap}=7360$（vs GPT-3 的 124），可在 $B_{\rm RP} \approx 32 \times 281.25$ 下使 attention FC 层达到 ridge point。MoE $B_{\rm MoE} = 281.25 \times 256/8 = 9000$（32 GPU 中每 GPU 处理 8 experts）。
  - **编译框架层**：论文未明确说明。
  - **Kernel调度层**：FlashMLA kernel 实现 reordered MLA decode，$\mathbf{C}_{\rm KV}$ 在 Score/Context 间复用。MoE 使用 fused expert kernel（gate/up/down projection fused）。all-to-all 通信使用 DeepEP 库。
  - **硬件架构层**：PIM 不再必要——MLA+MoE 使核心注意力层 ArI 接近 GPU ridge point，高 memory BW 的优势被高 compute 需求取代。仅低 batch 场景下 PIM 仍有优势。互联带宽是关键：高 BW NVLink（1.8 TB/s）vs 低 BW InfiniBand（100 GB/s）直接影响 MoE token dispatch/combine 延迟。小粒度部署（32 GPU×8）比大粒度（256 GPU monolithic）在 expert 偏斜时保持更高吞吐量（s=0.8 时 $\Gamma_{imb}^{acc}$ 低 6.13×）。
