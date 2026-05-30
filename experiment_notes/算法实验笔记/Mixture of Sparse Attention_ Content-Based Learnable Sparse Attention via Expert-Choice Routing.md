## Mixture of Sparse Attention: Content-Based Learnable Sparse Attention via Expert-Choice Routing

- 属于算法pipeline的实现是什么？实验比较什么？
  MoSA（Mixture of Sparse Attention）是一种基于 Expert-Choice Routing 的内容感知可学习稀疏注意力算法。核心创新：每个 attention head 配有一个可学习的 router（权重矩阵 W^r ∈ R^h），通过 sigmoid 计算每个 token 的 selection score，再用 TopK 选出每个 head 专属的 k 个 token，仅对这些 token 计算 Q、K、V 投影和 attention 矩阵。复杂度从 dense attention 的 O(T²) 降至 O(k²+T)。节省的 FLOPs 用于增加注意力头数，提升 head 专业化程度。关键细节：(1) router 输出 r_topk 通过 diag(r_topk)·A 乘到 attention 输出上，使 router 可通过梯度下降学习；(2) casual mask 适配 token 原始位置索引 I；(3) RoPE 旋转角度也基于原始位置而非子集位置；(4) 混合模型：4 个 dense head 保持训练稳定性，其余 head 用 MoSA 替换；(5) IsoFLOP 实验中每个 head 始终包含序列的第一个 token（attention sink）。

  实验比较 MoSA vs Dense baseline、Fixed Sparse Attention（基于位置的固定稀疏，stride=ρ，k=T/ρ 个 token）、Routing Transformer（online K-means 聚类，ρ 个簇各含 k 个 token）。共四个模型规模：Tiny(28M)、Small(113M)、Medium(210M)、Large(516M)。在 IsoFLOP 设定（逐步增加 sparsity ρ=T/k，用节省 FLOPs 增加 head 数）下评估 perplexity；在 perplexity-matched 设定下评估 wall-clock time、GPU memory、KV-cache 大小。还测试了长序列（T=8192，结合 local attention）和 6 个下游 zero-shot 任务。

- 硬件平台是什么，配置是什么。
  Tiny/Small/Medium 模型：单张 NVIDIA A100 GPU。Large 模型：两张 A100 GPU。纯 PyTorch 实现（无专用 CUDA kernel），使用 einsum、scatter、gather 操作。训练使用 Adam optimizer，lr=0.00025，gradient clipping norm=0.25，linear warmup 4k steps。

- 模型是什么。数据集和bench分别是什么。
  模型：Tiny（6 layers, h=512, FFN=2048, 9 heads, 28M params）、Small（9 layers, h=1024, FFN=4096, 9 heads, 113M）、Medium（18 layers, h=1024, FFN=4096, 9 heads, 210M）、Large（27 layers, h=1280, FFN=5120, 16 heads, 516M）。所有模型 head hidden size=64，基于 Pre-LN Transformer + RoPE。

  数据集：C4 训练集，100k batches，batch size=64，sequence length T=1024（约 6.5B tokens）。Tokenizer: SentencePiece，vocab size=8000，基于 sub-word units。

  Benchmark：C4 测试集 perplexity（主指标）；下游 zero-shot：LAMBADA、WinoGrande、BLiMP、HellaSwag、PIQA、AI2ARC（6 个任务）。

- 开源情况。基于开源文档和论文，使用例子解释，算法pipeline，至少具体到伪代码或张量计算。
  开源：https://github.com/piotrpiekos/MoSA。基于 PyTorch 实现，使用 einsum/scatter/gather。

  **MoSA 单层前向 pipeline（张量级）**：

  ```
  输入: X ∈ R^{T×h}  (T=sequence length, h=hidden dim)
  参数: router W^r_i ∈ R^h, Q/K/V projections W^Q_i/W^K_i/W^V_i ∈ R^{h×h'}, W^O_i ∈ R^{h'×h}
        h' = head dim (默认 64), k = 每个 head 选择的 token 数

  对于每个 head i (i=1..H):
    Step 1 - Token Selection:
      r = σ(X @ W^r_i)              # r ∈ R^T, sigmoid 逐元素
      r_topk, I = TopK(r, k)        # r_topk ∈ R^k, I ∈ {0..T-1}^k

    Step 2 - Gather selected tokens:
      X^s = X[I]                     # X^s ∈ R^{k×h}, 按索引 gather

    Step 3 - Q/K/V projections (仅对 k 个 token):
      Q = X^s @ W^Q_i               # Q ∈ R^{k×h'}
      K = X^s @ W^K_i               # K ∈ R^{k×h'}
      V = X^s @ W^V_i               # V ∈ R^{k×h'}

    Step 4 - Causal mask (基于原始位置索引 I):
      M_{a,b} = 0 if I_a ≥ I_b else -∞    # M ∈ R^{k×k}

    Step 5 - Sparse Attention:
      A = softmax(Q @ K^T / √h' + M) @ V  # A ∈ R^{k×h'}

    Step 6 - Router gate and output projection:
      X^o = diag(r_topk) @ A @ W^O_i       # X^o ∈ R^{k×h}, router gradient 通过乘法传递

    Step 7 - Scatter back to full sequence:
      Y_j = X^o_{idx} if j = I_{idx}, else 0   # Y ∈ R^{T×h}

  最终输出: Y = Σ_{i=1..H} Y_i
  ```

  **FLOPs 对比**：
  - Dense head: FLOP = 8hh'T + 4h'T²
  - MoSA head: FLOP = 8hh'k + 4h'k² + 2hT + h'k  (routing overhead: 2hT from scoring + h'k from gating)
  - Fixed sparse head: FLOP = 8hh'k + 4h'k²
  - Routing Transformer head: FLOP = ρ(6hh'k + 4h'k²) + 2h'T  (must compute all Q/K/V/O for all T tokens, Q=K in auto-regressive)

  **混合模型构建**：保持 4 个 dense head，其余 head 用 MoSA 替换。MoSA head 数量 = max H 使得 total FLOPs ≤ baseline FLOPs。sparsity ρ = T/k。例如 Tiny 模型 ρ=64 时：4 dense heads + 505 MoSA heads, 总参数 423M, perplexity 16.39（vs dense 22.46, -27%）。

  **Perplexity-matched 资源优化**：固定 ρ=32（Large: ρ=16），逐步增加 MoSA head 数直到 perplexity 匹配 dense baseline。结果：wall-clock time -7.3%~-12.9%，memory -1.6%~-10.0%，KV-cache -51.1%~-69.5%。
