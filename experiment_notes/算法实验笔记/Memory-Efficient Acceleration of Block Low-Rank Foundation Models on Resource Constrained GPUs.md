## Memory-Efficient Acceleration of Block Low-Rank Foundation Models on Resource Constrained GPUs

- 属于算法pipeline的实现是什么？实验比较什么？
  实现是以Block Low-Rank (BLR) 压缩技术——Monarch和BLAST——替换transformer模型中的dense线性层（Q/K/V/Oproj、gate/upproj、downproj等），将权重矩阵分解为多个小块的低秩因子表示。Monarch将dense权重划分为b₁×b₂块，每块独立做低秩分解W_{l,k}=V_{l,k}U_{l,k}，参数从i×o降至b₁b₂r'(p+q)，FLOP降至nb₁b₂r'(p+q)。BLAST进一步共享V_l和U_k并引入per-block对角矩阵S_{l,k}（W_{l,k}=V_l S_{l,k} U_k），参数为r(p+q+b²)，FLOP为nr(p+q+b²)，渐进复杂度与低秩相同但具有更高的表达能力和准确率。
  
  实验比较的算法baseline包括：Dense、Low-Rank (LR，标准SVD分解)、Monarch（Dao et al. 2022）、BLAST（Lee et al. 2024），全部在相同压缩比（CF=1.85×至3×）下对比。准确率评估：语言模型用WikiText-103/2 perplexity和zero-shot commonsense reasoning（PIQA, HellaSwag, Winogrande, BoolQ, ARC, OpenBookQA）accuracy；视觉模型用ImageNet classification accuracy；扩散模型用DDPM sampler生成图像后计算FID/sFID/IS vs 50,000 ImageNet validation images。

- 硬件平台是什么，配置是什么。
  NVIDIA A40（40GB显存，6MB共享L2 cache，数据中心GPU）和NVIDIA Jetson Orin Nano（8GB显存，4-6MB L2 cache，DDR DRAM，边缘GPU）。A40用于中大规模模型（Llama-7B、DiT-XL/2、Llama-3.2-1B），Jetson用于中小规模模型（Llama-3.2-1B、DiT-XL/2、GPT2-S、ViT-B）。软件：A40用Python 3.12.8、PyTorch 2.8.0、Triton 3.4.0、CUDA 12.6.3；Jetson用JetPack 6.2、L4T 36.4.3、CUDA 12.6.11、PyTorch 2.6.0、Triton 3.2.0。所有benchmarking用Triton do_bench()和PyTorch benchmarking utilities，torch.compile() + CUDA graph capture消除CPU dispatch overhead。

- 模型是什么。数据集和bench分别是什么。
  模型：Llama-7B（32层，i=o=4096，CF=2×）、Llama-3.2-1B（32层，i=o=2048，CF=2×）、GPT2-S（12层，i=o=768，CF=1.85×）、DiT-XL/2（28层，i=1152, o=3456/4608/6912，CF=2×）、ViT-B（12层，i=o=768，CF=3×）。数据集：WikiText-103/2（perplexity）、PIQA/HellaSwag/Winogrande/BoolQ/ARC/OpenBookQA（zero-shot accuracy）、ImageNet（分类accuracy）、SlimPajama-6B subset（re-training, 4000 steps）。所有替换层的具体配置（rank, blocks, i/o dimensions）记录在Table 3。

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  开源链接：https://github.com/pabillam/mem-efficient-blr

  BLAST算法pipeline（以Llama-7B QKVproj层为例，i=o=4096, r=1024, b=16, b₁=b₂=16, n=1024 tokens）：

  ```
  # BLAST权重参数化
  V ∈ R^{b₁ × p × r} = R^{16 × 256 × 1024}     # p = i/b₁ = 256
  S ∈ R^{b₁ × b₂ × r} = R^{16 × 16 × 1024}
  U ∈ R^{b₂ × r × q}   = R^{16 × 1024 × 256}   # q = o/b₂ = 256

  # 离线压缩（训练后/微调前）：
  # 1. 对dense权重 W[256:256, 256:256] 的每个 block (l,k) 执行preconditioned gradient descent
  #    300步求解: argmin ||W_{l,k} - V_l S_{l,k} U_k||_F
  #    SVD初始化 V_l, U_k, S_{l,k}=I
  # 2. 替代方案: SVD用于Low-Rank；block-wise SVD用于Monarch

  # 在线推理（BLAST线性层前向）：
  def blast_forward(X: [n, i]) -> Y: [n, o]:
      # Step 1: 对所有b₁个输入块并行计算 X_l @ V_l
      X_blocks = X.reshape(n, b₁, p)                    # [n, 16, 256]
      Z_l = batched_bmm(X_blocks, V)                    # [b₁, n, r] = [16, 1024, 1024]
      
      # Step 2: 对每个输出块k，计算加权求和
      for k in range(b₂):  # k = 0..15
          Y_k = zeros(n, q)                              # [1024, 256]
          for l in range(b₁):  # l = 0..15
              # Hadamard product with diagonal S_{l,k}
              Z_lk = Z_l[l] * S[l, k]                    # [n, r] ⊙ [r] → [n, r]
              Y_k += Z_lk @ U[k]                         # [n, r] @ [r, q] → [n, q]
      Y = concat([Y_0, ..., Y_{15}], dim=-1)            # [n, o]
  ```

  Monarch算法pipeline（同层配置，r'=r/b=64）：
  ```
  # Monarch权重参数化
  V ∈ R^{b₁ × (r'b₂) × p} = R^{16 × 1024 × 256}  # r'·b₂ = 64·16 = 1024
  U ∈ R^{b₂ × q × (b₁r')} = R^{16 × 256 × 1024}   # b₁·r' = 64·16 = 1024

  def monarch_forward(X: [n, i]) -> Y: [n, o]:
      X_blocks = X.reshape(n, b₁, p)                    # [n, 16, 256]
      # Step 1: 第一批bmm
      Z = batched_bmm(X_blocks, V.transpose(-1, -2))    # [b₁, n, r'b₂]
      # Step 2: 两次permutation (r'↔b₂, 然后 b₂↔b₁)
      Z = Z.reshape(b₁, n, b₂, r').transpose(0, 2, 1, 3)  # [b₂, n, b₁r']
      # Step 3: 第二批bmm
      Y_k = Z[k] @ U[k] for k in range(b₂)             # [n, q] each
      # Step 4: 最终permutation (b₂, n, q) → (n, q, b₂)
      Y = permute(Y, ...)
  ```

- 关键实验结果：
  - 准确率（Table 1）：BLAST在多数模型上取得最优accuracy-efficiency tradeoff。Llama-7B CF=2×: BLAST WikiText-2 PPL=14.21 vs Monarch 19.54 vs LR 26.33（Dense=9.37）；ViT-B CF=3×: BLAST ImageNet=79.3% vs Dense 78.7%（BLAST甚至略高于Dense）。
  - Roofline分析：多token推理(n=1024)下，Monarch和BLAST从compute-bound落入memory-bound，因为block结构产生大量中间数据移动。Monarch 1.14-1.68× slower than dense，BLAST 2.63-4.31× slower（在未优化的PyTorch实现下）。
