## V2Drop: Variation-aware Vision Token Dropping for Faster Large Vision-Language Models

- 属于算法pipeline的实现是什么？实验比较什么？
  实现：**V2Drop**，一种基于 token 变异性（variation）的视觉 token 渐进式压缩方法。核心思路：(1) 不再依赖 attention weights 等外部信号判断 token 重要性，而是直接测量 visual token 在相邻 LLM 层之间的表示变化（L2 distance）；(2) 在 3 个预选层（如 LLaVA-1.5-7B 上的 layers 3, 17, 22）进行多阶段渐进式剪枝（progressive dropping），每层按 variation score 降序排序后保留 top-K 高变异性 token，丢弃低变异性的"惰性 token"；(3) 消除 attention-based 方法的位置偏见（positional bias），且天然兼容 FlashAttention 等高效算子。
  实验比较：V2Drop vs. ToMe、FastV、HiRED、LLaVA-PruMerge、SparseVLM、PDrop、DART、DyCoke 在多个 image/video benchmark 上的性能保留率、生成延迟、吞吐量、GPU 峰值显存。

- 硬件平台是什么，配置是什么。
  NVIDIA A100-PCIe-80GB GPU。软件：Python 3.10、PyTorch 2.1.2、CUDA 12.1。

- 模型是什么。数据集和bench分别是什么。
  模型：LLaVA-1.5-7B、Qwen2-VL-7B、LLaVA-OV-7B。
  Benchmarks：GQA、MMBench、MME、POPE、ScienceQA、TextVQA、AI2D、MMStar（图像理解）；MVBench、VideoMME（视频理解）。
  指标：benchmark 性能保留率（以原始模型为 100%）、LLM Generation Latency、Model Generation Latency、Total Latency、GPU Peak Memory、Throughput。

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  已开源：https://github.com/xuyang-liu16/V2Drop（Apache-2.0，CVPR 2026）。
  算法流程（见论文 Algorithm 1）：
  输入：视觉 token F^v ∈ R^{M×D'}，剪枝层 L={l_1, l_2, l_3}，压缩目标 {K_a, K_b, K_c}
  M_curr ← M
  for l = 1 to L do:
      if l ∈ L:
          // Step 1: Variation Computation
          for i = 1 to M_curr:
              s_i^{(l)} ← ||f_i^{(l)} - f_i^{(l-1)}||_2  // L2 距离
          S^{(l)} ← {s_1, ..., s_{M_curr}}
          // Step 2: Token Ranking & Selection
          indices ← argsort(S^{(l)}, descending)
          F̂_l^v ← {f_indices[j] : j=1,...,K_l}  // 保留 top-K
          F_curr^v ← F̂_l^v, M_curr ← K_l
      // Step 3: 继续前向传播
      F_curr^v ← TransformerLayer(F_curr^v)
  return F_curr^v

  张量计算开销：对 M 个 D' 维 token 计算 L2 距离需 3MD' FLOPs（LLaVA-1.5 中约 7M FLOPs），仅为单层 attention（32B FLOPs）的 0.022%。三层总开销约 21M FLOPs，占完整前向传播的 0.002%。
  渐进式压缩 schedule 示例（LLaVA-1.5-7B, retain 192 tokens）：M=576 → layer 3 保留 top-50% → layer 17 保留 top-30% → layer 22 保留指定数量的所有 token。
